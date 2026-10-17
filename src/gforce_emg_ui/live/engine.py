"""Live inference engine (Qt-free).

  DataHub (source thread) -> on_emg: copy block, append to a bounded inbox, return
  worker thread            -> validate samples, trailing contiguous run, a window of
                              N samples every H new samples, frozen predict_windows()
  latest-result snapshot   <- read by the GUI refresh (no per-result Qt events)
  watchdog thread          -> expires output when input or the prediction is stale

N and H come from the loaded model (sample counts, never milliseconds). Only the 8 EMG
values enter windows; indices, masks, receipt times and stream/segment IDs are used for
validity and diagnostics only. Inbox overflow drops the OLDEST blocks (counted); the
worker sees the sequence gap and warms up again instead of stitching across lost input.
Windows whose newest sample is already older than stale_s are not classified.
Every Start / Stop / reset / model change / input staleness bumps the generation:
queued work and results from an older generation are discarded.
"""

import collections
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import numpy as np

from ..config import EMG_CHANNELS

HUB_CHANNELS = tuple(f"ch{i + 1}" for i in range(EMG_CHANNELS))   # EmgBlock.data row order

NO_MODEL, READY, WARMING_UP, RUNNING, STALE, INVALID_INPUT, STOPPED, ERROR = (
    "NO_MODEL", "READY", "WARMING_UP", "RUNNING", "STALE", "INVALID_INPUT", "STOPPED", "ERROR")
ACTIVE = (WARMING_UP, RUNNING, STALE, INVALID_INPUT)

# Reset reasons. INVALID ones show INVALID_INPUT until valid samples follow.
INVALID_REASONS = {"lost_sample", "invalid_channel", "nonfinite_value", "malformed_block"}


@dataclass(frozen=True)
class LiveSettings:
    stale_s: float = 0.5            # engineering default, to verify on hardware
    inbox_samples: int = 1024       # bounded input queue (queued samples; newest block kept)
    watchdog_s: float = 0.05        # 0 = no watchdog thread (tests call check())
    history: int = 2000             # points kept for the history plot
    timing_samples: int = 500


class LiveResult(NamedTuple):
    pred_id: int
    run_id: str
    generation: int
    stream_id: int
    segment: int
    window_start_index: int
    window_end_index: int
    first_receipt_mono: float       # receipt of the batch holding the window's first sample
    newest_receipt_mono: float      # receipt of the batch holding its newest sample
    newest_receipt_wall: float
    result_mono: float              # inference finished
    inference_ms: float             # feature + model time of the whole batch call
    batch_size: int
    prediction: Any                 # emg_classifier.Prediction

    @property
    def valid(self) -> bool:
        return bool(self.prediction.valid)

    def age(self, now: float) -> float:
        return now - self.newest_receipt_mono


@dataclass(frozen=True)
class LiveSnapshot:
    version: int
    state: str
    reason: str
    result: Optional[LiveResult]    # newest published result of the current run, if not cleared
    run_id: Optional[str]
    generation: int
    warm: Tuple[int, int]           # (samples in the current contiguous run, N)
    stale_s: float
    last_input_mono: Optional[float]
    counters: dict

    def current(self, now: float) -> Optional[LiveResult]:
        """The result to show as s, or None (shown as s = --). Freshness rechecked here."""
        r = self.result
        if self.state != RUNNING or r is None or not r.valid or r.age(now) > self.stale_s:
            return None
        return r


class _Item(NamedTuple):
    seq: int
    gen: int
    data: Any
    valid: Any
    index: Any
    lost: Any
    stream_id: int
    segment: int
    rate: float
    host_mono: float
    host_time: float
    error: Optional[str]


class _Window(NamedTuple):
    data: np.ndarray                # (N, 8) in the model's channel order
    start_index: int
    end_index: int
    first_receipt: float
    newest_receipt: float
    newest_wall: float
    stream_id: int
    segment: int


def _pct(values, q) -> Optional[float]:
    return float(np.percentile(values, q)) if len(values) else None


class LiveEngine:
    """Thread roles: start/stop/reset/set_model/snapshot from the GUI thread; _on_emg on
    the source thread; _process on the worker; check() on the watchdog (or a test)."""

    SUB_NAME = "live-classifier"

    def __init__(self, hub, settings: Optional[LiveSettings] = None,
                 clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time):
        self.hub = hub
        self.settings = settings or LiveSettings()
        self._clock = clock
        self._wall = wall
        self._lock = threading.Lock()              # state, result, counters, generation
        self._cv = threading.Condition()           # inbox
        self._inbox: collections.deque = collections.deque()
        self._closing = threading.Event()
        self._threads: List[threading.Thread] = []
        self._model = None
        self._order: Optional[List[int]] = None
        self._gen = 0
        self._accepting = False
        self._subscribed = False
        self._fn = self._on_emg                    # one callable: subscribe == unsubscribe
        self._state, self._reason = NO_MODEL, ""
        self._result: Optional[LiveResult] = None
        self._version = 0
        self._run_id: Optional[str] = None
        self._run_meta: dict = {}
        self._started_mono: Optional[float] = None
        self._input_stale = False
        self._log = None
        self._pred_id = 0
        self._warm = (0, 0)
        self._counts = collections.Counter()
        self._resets = collections.Counter()
        self._history: collections.deque = collections.deque(maxlen=self.settings.history)
        n = self.settings.timing_samples
        self._t_infer = collections.deque(maxlen=n)        # ms per batch call
        self._t_infer_win = collections.deque(maxlen=n)    # ms per window
        self._t_ready = collections.deque(maxlen=n)        # newest receipt -> result ms
        # source thread (under _cv)
        self._seq = 0
        self._inbox_n = 0                          # samples queued in the inbox
        self._last_input_mono: Optional[float] = None
        self._dropped_blocks = 0
        self._dropped_samples = 0
        # worker thread only
        self._w_gen = -1
        self._w_run = None
        self._run_rate: Optional[float] = None    # reported rate at the run's first block
        self._w_counts = collections.Counter()    # merged into _counts under _lock
        self._w_reset()

    # -- GUI thread --------------------------------------------------------------
    @property
    def model(self):
        return self._model

    @property
    def running(self) -> bool:
        return self._accepting

    def set_model(self, model):
        """Install a verified model (or None). Stops and invalidates any current run."""
        self.stop("model_replaced")
        order = None
        if model is not None:
            missing = [c for c in model.channel_names if c not in HUB_CHANNELS]
            if missing:
                raise ValueError(f"model channels {missing} are not in the EMG stream "
                                 f"{HUB_CHANNELS}")
            order = [HUB_CHANNELS.index(c) for c in model.channel_names]
        with self._lock:
            self._model, self._order = model, order
            self._gen += 1
            self._result = None
            self._set_state(READY if model is not None else NO_MODEL, "")

    def start(self, run_meta: Optional[dict] = None, log=None,
              run_id: Optional[str] = None) -> str:
        with self._lock:
            if self._model is None:
                raise RuntimeError("no model loaded")
            if self._accepting:
                return self._run_id
            self._gen += 1
            self._run_id = run_id or uuid.uuid4().hex[:10]
            self._run_meta = dict(run_meta or {})
            self._started_mono = self._clock()
            self._input_stale = False
            self._log = log
            self._result = None
            self._counts.clear()
            self._resets.clear()
            self._warm = (0, self._model.window_samples)
            self._gap_locked(self._started_mono)
            self._set_state(WARMING_UP, "collecting the first window")
            with self._cv:
                self._inbox.clear()
                self._inbox_n = 0
                self._last_input_mono = None
                self._dropped_blocks = self._dropped_samples = 0
            self._accepting = True
            run_id, gen = self._run_id, self._gen
        self._ensure_threads()
        if not self._subscribed:
            self.hub.subscribe_emg(self._fn, name=self.SUB_NAME)
            self._subscribed = True
        self._event("start", f"generation {gen}")
        return run_id

    def stop(self, reason: str = "user_stop"):
        """Stop classifying (never touches the band, recorder or collection)."""
        with self._lock:
            was = self._accepting
            self._stop_locked(reason)
        if self._subscribed:
            self.hub.unsubscribe_emg(self._fn)
            self._subscribed = False
        if was:
            self._close_log(reason)

    def reset(self, reason: str = "user_reset"):
        """Clear the buffer and warm up again (run continues)."""
        with self._lock:
            if not self._accepting:
                return
            self._gen += 1
            self._resets[reason] += 1
            self._result = None
            self._gap_locked(self._clock())
            self._set_state(WARMING_UP, reason)
        self._event("reset", reason)

    def snapshot(self) -> LiveSnapshot:
        with self._cv:
            last_in = self._last_input_mono
            dropped = (self._dropped_blocks, self._dropped_samples)
        with self._lock:
            counters = dict(self._counts)
            counters.update(dropped_blocks=dropped[0], dropped_samples=dropped[1],
                            resets=dict(self._resets), predictions=self._pred_id)
            return LiveSnapshot(self._version, self._state, self._reason, self._result,
                                self._run_id, self._gen, self._warm, self.settings.stale_s,
                                last_in, counters)

    def history(self) -> np.ndarray:
        """(k, 3) rows of (newest receipt mono, s_discrete, s_expected); NaN rows = gaps."""
        with self._lock:
            return np.array(self._history, dtype=float).reshape(-1, 3)

    def timings(self) -> dict:
        """Host-side p50/p95 (ms) over the last results; None until measured."""
        with self._lock:
            sets = {"inference_batch_ms": list(self._t_infer),
                    "inference_window_ms": list(self._t_infer_win),
                    "receipt_to_result_ms": list(self._t_ready)}
        return {k: {"p50": _pct(v, 50), "p95": _pct(v, 95), "n": len(v)}
                for k, v in sets.items()}

    def log_display(self, pred_id: int, gui_mono: float, receipt_to_gui_ms: float):
        log = self._log
        if log is not None:
            log.put_display({"pred_id": pred_id, "gui_applied_mono": gui_mono,
                             "receipt_to_gui_ms": receipt_to_gui_ms})

    def shutdown(self, timeout: float = 0.5):
        """Stop, then let the threads exit (bounded joins; they are daemons)."""
        self.stop("shutdown")
        self._closing.set()
        with self._cv:
            self._cv.notify_all()
        for t in self._threads:
            t.join(timeout)
        self._threads = [t for t in self._threads if t.is_alive()]

    @property
    def queued_samples(self) -> int:
        with self._cv:
            return self._inbox_n

    @property
    def threads(self) -> List[threading.Thread]:
        return [t for t in self._threads if t.is_alive()]

    # -- watchdog ----------------------------------------------------------------
    def check(self, now: Optional[float] = None):
        """Expire stale input / predictions. Runs on the watchdog thread every watchdog_s."""
        now = self._clock() if now is None else now
        last_in = self._last_input_mono            # float written by the source thread
        stale = self.settings.stale_s
        event = None
        with self._lock:
            if not self._accepting:
                return
            ref = last_in if last_in is not None else self._started_mono
            if ref is not None and now - ref > stale:
                if not self._input_stale:
                    self._input_stale = True
                    self._gen += 1                 # input stale: buffer restarts
                    self._resets["input_stale"] += 1
                    event = f"no EMG received for {now - ref:.2f} s"
            elif (self._state != STALE and self._result is not None
                  and self._result.age(now) > stale):
                event = (f"prediction expired: newest contributing sample received "
                         f"{self._result.age(now):.2f} s ago")
            if event:
                self._result = None
                self._counts["stale_events"] += 1
                self._gap_locked(now)
                self._set_state(STALE, event)
        if event:
            self._event("stale", event)

    # -- source thread -----------------------------------------------------------
    def _on_emg(self, block):
        if not self._accepting:
            return
        gen = self._gen
        try:
            parts = (np.array(block.data), np.array(block.valid), np.array(block.sample_index),
                     np.array(block.lost), int(block.stream_id), int(block.segment),
                     float(block.sample_rate), float(block.host_mono), float(block.host_time),
                     None)
        except Exception as e:                    # malformed block: reported by the worker
            mono = getattr(block, "host_mono", None)
            parts = ((None,) * 7 + (mono if isinstance(mono, float) else None, None,
                                    f"{type(e).__name__}: {e}"))
        n = 0 if parts[2] is None else int(np.size(parts[2]))
        with self._cv:
            self._seq += 1
            item = _Item(self._seq, gen, *parts)
            while self._inbox and self._inbox_n + n > self.settings.inbox_samples:
                old = self._inbox.popleft()        # freshness first: drop the oldest
                m = 0 if old.index is None else int(np.size(old.index))
                self._inbox_n -= m
                self._dropped_blocks += 1
                self._dropped_samples += m
            self._inbox.append(item)
            self._inbox_n += n
            if item.host_mono is not None:
                self._last_input_mono = item.host_mono
            self._cv.notify()

    # -- worker thread -----------------------------------------------------------
    def _ensure_threads(self):
        if self._threads and all(t.is_alive() for t in self._threads):
            return
        self._closing.clear()
        self._threads = [threading.Thread(target=self._work, name="live-worker", daemon=True)]
        if self.settings.watchdog_s > 0:
            self._threads.append(threading.Thread(target=self._watch, name="live-watchdog",
                                                  daemon=True))
        for t in self._threads:
            t.start()

    def _watch(self):
        while not self._closing.wait(self.settings.watchdog_s):
            try:
                self.check()
            except Exception:                      # never let the watchdog die silently
                traceback.print_exc()

    def _work(self):
        while True:
            with self._cv:
                while not self._inbox and not self._closing.is_set():
                    self._cv.wait(0.5)
                if self._closing.is_set():
                    return
                items = list(self._inbox)
                self._inbox.clear()
                self._inbox_n = 0
            try:
                self._process(items)
            except Exception as e:                 # bug or model failure: stop, show ERROR
                traceback.print_exc()
                with self._lock:
                    self._counts["errors"] += 1
                    self._stop_locked(f"{type(e).__name__}: {e}", state=ERROR)
                self._close_log("error")

    def _w_reset(self):
        self._tail = None                         # (k <= N-1, 8) newest samples of the run
        self._tail_index = np.empty(0, np.int64)
        self._tail_receipt = np.empty(0)
        self._count = 0                           # samples in the current contiguous run
        self._stream = self._segment = None
        self._last_index: Optional[int] = None
        self._last_receipt: Optional[float] = None
        self._expected_seq: Optional[int] = None

    def _process(self, items: List[_Item]):
        with self._lock:
            gen, model, order, run = self._gen, self._model, self._order, self._run_id
            if not self._accepting or model is None:
                return
        if run != self._w_run:
            self._w_run, self._run_rate = run, None
            self._w_counts.clear()
        if gen != self._w_gen:
            self._w_reset()
            self._w_gen = gen
        self._N, self._H, self._ord = model.window_samples, model.hop_samples, order
        events: list = []
        windows: List[_Window] = []
        old = 0
        for it in items:
            if it.gen != gen:
                old += 1
                continue
            if self._ingest(it, events, windows):
                break                             # fatal: stream configuration changed
        now = self._clock()
        todo = [k for k, w in enumerate(windows) if now - w.newest_receipt <= self.settings.stale_s]
        preds, infer_s = {}, 0.0
        if todo:
            t0 = time.perf_counter()
            out = model.clf.predict_windows([windows[k].data for k in todo],
                                            channel_names=model.channel_names)
            infer_s = time.perf_counter() - t0
            preds = dict(zip(todo, out))
        self._publish(gen, items, old, events, windows, preds, infer_s, len(windows) - len(todo))

    def _ingest(self, it: _Item, events: list, windows: list) -> bool:
        """Validate one block and extend the run. True = fatal (stop the run)."""
        stale = self.settings.stale_s
        if it.seq != self._expected_seq and self._expected_seq is not None:
            self._reset(events, "input_dropped")
            self._last_index = None               # no continuity across dropped blocks
        self._expected_seq = it.seq + 1
        bad = it.error or _malformed(it)
        if bad:
            self._reset(events, "malformed_block", bad)
            self._last_index = None
            return False
        if self._run_rate is None:
            self._run_rate = it.rate
        elif it.rate != self._run_rate:
            events.append(("fatal", f"reported stream rate changed {self._run_rate:g} -> "
                                    f"{it.rate:g} Hz: review compatibility and Start again"))
            return True
        if self._stream is not None and (it.stream_id, it.segment) != (self._stream,
                                                                       self._segment):
            self._reset(events, "segment_change" if it.stream_id == self._stream
                        else "stream_change")
            self._last_index = None
        self._stream, self._segment = it.stream_id, it.segment
        if self._last_receipt is not None and it.host_mono - self._last_receipt > stale:
            self._reset(events, "input_gap")
        self._last_receipt = it.host_mono

        data, idx = it.data, it.index.astype(np.int64)
        n = data.shape[1]
        breaks = np.zeros(n, bool)
        breaks[1:] = np.diff(idx) != 1
        prev = self._last_index
        breaks[0] = prev is not None and int(idx[0]) != prev + 1
        ok = it.valid.all(axis=0) & ~it.lost & np.isfinite(data).all(axis=0)
        p = 0
        for c in np.flatnonzero(breaks | ~ok):
            if c > p:
                self._extend(it, p, c, windows, events)
            if breaks[c]:
                d = int(idx[c]) - (int(idx[c - 1]) if c else prev)
                self._reset(events, "duplicate_index" if d == 0 else
                            "backward_index" if d < 0 else "index_gap")
            if not ok[c]:
                self._reset(events, "lost_sample" if it.lost[c] else
                            "invalid_channel" if not it.valid[:, c].all() else "nonfinite_value")
                self._w_counts["rejected_samples"] += 1
                p = c + 1
            else:
                p = c
        if p < n:
            self._extend(it, p, n, windows, events)
        self._last_index = int(idx[-1])
        return False

    def _reset(self, events: list, reason: str, detail: str = ""):
        if self._count or not events or events[-1][:2] != ("reset", reason):
            events.append(("reset", reason, detail, self._count > 0))
        self._count = 0
        self._tail = None
        self._tail_index = np.empty(0, np.int64)
        self._tail_receipt = np.empty(0)

    def _extend(self, it: _Item, a: int, b: int, windows: list, events: list):
        """Append samples a..b-1 (all valid, contiguous); cut every due window."""
        N, H = self._N, self._H
        rows = it.data[self._ord, a:b].T                       # (m, 8): model channel order
        idx = it.index[a:b].astype(np.int64)
        rc = np.full(b - a, it.host_mono)
        if self._tail is not None and len(self._tail):
            rows = np.concatenate((self._tail, rows))
            idx = np.concatenate((self._tail_index, idx))
            rc = np.concatenate((self._tail_receipt, rc))
        k = len(rows) - (b - a)                               # tail samples in front
        c0, c1 = self._count, self._count + (b - a)
        first = N if c0 < N else N + ((c0 - N) // H + 1) * H
        for c in range(first, c1 + 1, H):
            e = c - c0 + k - 1
            s = e - N + 1
            windows.append(_Window(rows[s:e + 1].copy(), int(idx[s]), int(idx[e]),
                                   float(rc[s]), float(rc[e]), it.host_time,
                                   it.stream_id, it.segment))
            events.append(("window", len(windows) - 1))
        self._count = c1
        keep = min(N - 1, len(rows))
        self._tail = rows[len(rows) - keep:].copy()
        self._tail_index = idx[len(idx) - keep:].copy()
        self._tail_receipt = rc[len(rc) - keep:].copy()

    def _publish(self, gen, items, old, events, windows, preds, infer_s, skipped):
        rows = []
        log_events = []
        with self._lock:
            c = self._counts
            c.update(self._w_counts)
            self._w_counts.clear()
            if gen != self._gen or not self._accepting:
                c["discarded_results"] += len(preds)
                c["discarded_blocks"] += len(items)
                return
            c["blocks"] += len(items) - old
            c["discarded_blocks"] += old
            c["windows"] += len(windows)
            c["skipped_stale_windows"] += skipped
            now = self._clock()                   # freshness rechecked before publishing
            stale = self.settings.stale_s
            changed = bool(items)
            if preds:
                self._t_infer.append(infer_s * 1000)
                self._t_infer_win.append(infer_s * 1000 / len(preds))
            for ev in events:
                if ev[0] == "reset":
                    _, reason, detail, cleared = ev
                    if cleared:
                        self._resets[reason] += 1
                        log_events.append(("reset", f"{reason} {detail}".strip()))
                    self._result = None
                    self._gap_locked(now)
                    self._set_state(INVALID_INPUT if reason in INVALID_REASONS else WARMING_UP,
                                    f"{reason}: {detail}" if detail else reason)
                elif ev[0] == "fatal":
                    self._stop_locked(ev[1])
                    log_events.append(("fatal", ev[1]))
                    break
                else:
                    k = ev[1]
                    if k not in preds:
                        continue
                    w, p = windows[k], preds[k]
                    self._pred_id += 1
                    r = LiveResult(self._pred_id, self._run_id, gen, w.stream_id, w.segment,
                                   w.start_index, w.end_index, w.first_receipt, w.newest_receipt,
                                   w.newest_wall, now, infer_s * 1000, len(preds), p)
                    fresh = r.age(now) <= stale
                    rows.append((r, fresh))
                    if not fresh:
                        c["expired_before_publish"] += 1
                        continue
                    self._t_ready.append(r.age(now) * 1000)
                    self._result = r
                    if p.valid:
                        c["valid_results"] += 1
                        self._history.append((w.newest_receipt, p.s_discrete, p.s_expected))
                        self._set_state(RUNNING, "")
                    else:
                        c["invalid_results"] += 1
                        self._gap_locked(w.newest_receipt)
                        self._set_state(INVALID_INPUT, f"{p.status}: {p.reason}")
            if self._accepting:
                self._warm = (self._count, self._N)
                if self._state == STALE and changed and self._count < self._N:
                    self._set_state(WARMING_UP, "input resumed")
            if changed:
                self._input_stale = False
                self._version += 1
            counters = self._log_counters()
            log = self._log
            run_id = self._run_id
        if log is not None:
            for r, fresh in rows:
                log.put_prediction(_row(r, fresh, counters))
            for kind, detail in log_events:
                log.put_event(_event_row(self._clock(), self._wall(), run_id, gen, kind, detail))
            if any(kind == "fatal" for kind, _ in log_events):
                self._close_log("stopped")

    # -- helpers (callers hold _lock) --------------------------------------------
    def _set_state(self, state: str, reason: str):
        if (state, reason) != (self._state, self._reason):
            self._state, self._reason = state, reason
            self._version += 1

    def _gap_locked(self, t: float):
        if self._history and not np.isnan(self._history[-1][1]):
            self._history.append((t, np.nan, np.nan))

    def _stop_locked(self, reason: str, state: str = STOPPED):
        if self._accepting:
            self._accepting = False
            self._gen += 1
            self._result = None
            self._gap_locked(self._clock())
            self._set_state(state, reason)
        elif state == ERROR:
            self._set_state(ERROR, reason)
        with self._cv:
            self._inbox.clear()
            self._inbox_n = 0

    def _log_counters(self) -> dict:
        return {"dropped_blocks": self._dropped_blocks, "dropped_samples": self._dropped_samples,
                "skipped_stale_windows": self._counts["skipped_stale_windows"],
                "expired_before_publish": self._counts["expired_before_publish"],
                "resets": sum(self._resets.values())}

    def _event(self, kind: str, detail: str):
        log = self._log
        if log is not None:
            log.put_event(_event_row(self._clock(), self._wall(), self._run_id, self._gen,
                                     kind, detail))

    def _close_log(self, reason: str):
        log, self._log = self._log, None
        if log is not None:
            snap = self.snapshot()
            log.put_event(_event_row(self._clock(), self._wall(), snap.run_id, snap.generation,
                                     "stop", reason))
            log.close({"stop_reason": reason, "counters": snap.counters,
                       "timings": self.timings()})


def _malformed(it: _Item) -> Optional[str]:
    d, v, i, lost = it.data, it.valid, it.index, it.lost
    if d.ndim != 2 or d.shape[0] != len(HUB_CHANNELS) or d.shape[1] < 1:
        return f"data shape {d.shape}, expected ({len(HUB_CHANNELS)}, n>0)"
    n = d.shape[1]
    if not np.issubdtype(d.dtype, np.floating) and not np.issubdtype(d.dtype, np.integer):
        return f"data dtype {d.dtype}"
    if v.shape != d.shape or v.dtype != bool:
        return f"valid mask shape {v.shape} / dtype {v.dtype}"
    if i.shape != (n,) or not np.issubdtype(i.dtype, np.integer):
        return f"sample_index shape {i.shape} / dtype {i.dtype}"
    if lost.shape != (n,) or lost.dtype != bool:
        return f"lost shape {lost.shape} / dtype {lost.dtype}"
    return None


PREDICTION_FIELDS = [
    "pred_id", "run_id", "generation", "stream_id", "segment", "window_start_index",
    "window_end_index", "first_receipt_mono", "newest_receipt_mono", "newest_receipt_wall",
    "result_mono", "inference_ms", "batch_size", "published", "valid", "status", "reason",
    "class_id", "label", "p_low", "p_medium", "p_high", "s_discrete", "s_expected",
    "max_probability", "dropped_blocks", "dropped_samples", "skipped_stale_windows",
    "expired_before_publish", "resets"]
EVENT_FIELDS = ["t_mono", "t_wall", "run_id", "generation", "event", "detail"]
DISPLAY_FIELDS = ["pred_id", "gui_applied_mono", "receipt_to_gui_ms"]


def _row(r: LiveResult, published: bool, counters: dict) -> dict:
    p = r.prediction
    probs = p.probabilities or {}
    return {"pred_id": r.pred_id, "run_id": r.run_id, "generation": r.generation,
            "stream_id": r.stream_id, "segment": r.segment,
            "window_start_index": r.window_start_index, "window_end_index": r.window_end_index,
            "first_receipt_mono": r.first_receipt_mono,
            "newest_receipt_mono": r.newest_receipt_mono,
            "newest_receipt_wall": r.newest_receipt_wall, "result_mono": r.result_mono,
            "inference_ms": r.inference_ms, "batch_size": r.batch_size,
            "published": int(published), "valid": int(bool(p.valid)), "status": p.status,
            "reason": p.reason or ("" if published else "expired before publishing"),
            "class_id": p.class_id, "label": p.label, "p_low": probs.get("low"),
            "p_medium": probs.get("medium"), "p_high": probs.get("high"),
            "s_discrete": p.s_discrete, "s_expected": p.s_expected,
            "max_probability": p.max_probability, **counters}


def _event_row(mono, wall, run_id, gen, kind, detail) -> dict:
    return {"t_mono": mono, "t_wall": wall, "run_id": run_id, "generation": gen,
            "event": kind, "detail": detail}
