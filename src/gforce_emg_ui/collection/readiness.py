"""Stream and electrode-contact readiness, and per-trial stream-quality capture.

Electrode contact uses the existing impedance-based contact_level() thresholds:
GOOD (green, <= 500 kOhm) and MARGINAL (orange, <= 999 kOhm) are accepted; POOR
(red) blocks. Missing, NaN, infinite, zero or negative impedance is UNKNOWN and
blocks too (no reading is not a green or orange signal). EMG amplitude is never
used: low-effort trials are meant to have small signals.

Freshness comes from received EMG batches (their host receipt time), never from
the time an impedance value last changed: the SDK copies its latest impedance
into every EMG sample and exposes no measurement time or age for it, so an
unchanged good value on fresh EMG batches is still current, while the last
values shown after a disconnect are not (no fresh batches -> not ready).

StreamTracker.on_emg runs on the data thread (hub subscriber): it stores a tiny
per-block record and returns. Analysis (summaries, flags) runs on the GUI thread.
"""

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, NamedTuple, Optional, Tuple

import numpy as np

from ..config import EMG_CHANNELS
from ..contact import GOOD, MARGINAL, UNKNOWN, contact_level, format_kohm
from ..sources import STREAMING


ACCEPTED_LEVELS = (GOOD, MARGINAL)          # green + orange; red / unknown block


def channel_levels(impedance) -> Tuple[str, ...]:
    """contact_level() per channel; invalid or missing values are UNKNOWN."""
    out = []
    vals = [] if impedance is None else list(np.asarray(impedance, dtype=float).ravel())
    for ch in range(EMG_CHANNELS):
        v = vals[ch] if ch < len(vals) else float("nan")
        out.append(contact_level(v) if math.isfinite(v) and v > 0 else UNKNOWN)
    return tuple(out)


def _imp_list(impedance) -> list:
    """Reported impedance per channel (Ohm); None where missing or not finite."""
    vals = [] if impedance is None else [float(v) for v in np.asarray(impedance, float).ravel()]
    vals = (vals + [float("nan")] * EMG_CHANNELS)[:EMG_CHANNELS]
    return [v if math.isfinite(v) else None for v in vals]


class BlockRecord(NamedTuple):            # NamedTuple: cheap to build on the data thread
    host_mono: float          # batch receipt time (not a per-sample acquisition time)
    host_time: float
    segment: int
    stream_id: int
    first_index: int
    last_index: int
    n: int
    n_lost: int               # samples the SDK flagged lost
    n_invalid: int            # samples with at least one invalid channel
    channels_present: bool    # every channel has at least one valid sample in the batch
    levels: Tuple[str, ...]
    impedance: Tuple[Optional[float], ...]

    @property
    def contact_ok(self) -> bool:
        return all(lv in ACCEPTED_LEVELS for lv in self.levels)


class StreamTracker:
    """Subscribe with hub.subscribe_emg(tracker.on_emg). Thread-safe snapshot reads."""

    def __init__(self, freshness_s: float = 0.5, max_records: int = 200_000):
        self.freshness_s = freshness_s
        self._max = max_records
        self._lock = threading.Lock()
        self._last: Optional[BlockRecord] = None
        self._good_since: Optional[float] = None
        self._capture: Optional[list] = None
        self._capture_prev: Optional[BlockRecord] = None
        self._capture_truncated = False
        self._prev_imp = None
        self._prev_levels: Tuple[str, ...] = (UNKNOWN,) * EMG_CHANNELS
        self._prev_imp_list: tuple = (None,) * EMG_CHANNELS

    # -- data thread -------------------------------------------------------------
    def on_emg(self, block):
        imp = block.impedance
        key = None if imp is None else imp.tobytes()      # NaN-safe, cheap equality
        if key is None or key != self._prev_imp:
            self._prev_imp = key
            self._prev_levels = channel_levels(imp)
            self._prev_imp_list = tuple(_imp_list(imp))
        valid = block.valid
        rec = BlockRecord(
            host_mono=block.host_mono, host_time=block.host_time, segment=block.segment,
            stream_id=block.stream_id, first_index=int(block.sample_index[0]),
            last_index=int(block.sample_index[-1]), n=int(block.data.shape[1]),
            n_lost=int(np.count_nonzero(block.lost)),
            n_invalid=int(np.count_nonzero(~valid.all(axis=0))),
            channels_present=valid.shape[0] >= EMG_CHANNELS and bool(valid.any(axis=1).all()),
            levels=self._prev_levels, impedance=self._prev_imp_list)
        ok = rec.contact_ok and rec.channels_present
        with self._lock:
            prev = self._last
            restart = (prev is None or prev.segment != rec.segment
                       or rec.host_mono - prev.host_mono > self.freshness_s)
            if not ok:
                self._good_since = None
            elif restart or self._good_since is None:
                self._good_since = rec.host_mono
            self._last = rec
            cap = self._capture
            if cap is not None:
                if len(cap) < self._max:
                    cap.append(rec)
                else:
                    self._capture_truncated = True

    # -- GUI thread ---------------------------------------------------------------
    def snapshot(self) -> Tuple[Optional[BlockRecord], Optional[float]]:
        """(newest block record, receipt time since which every block was good)."""
        with self._lock:
            return self._last, self._good_since

    @property
    def last(self) -> Optional[BlockRecord]:
        return self._last

    def start_capture(self):
        with self._lock:
            self._capture = []
            self._capture_prev = self._last
            self._capture_truncated = False

    def stop_capture(self) -> Tuple[Optional[BlockRecord], List[BlockRecord], bool]:
        """(record before the capture started, captured records, truncated)."""
        with self._lock:
            cap, self._capture = self._capture or [], None
            return self._capture_prev, cap, self._capture_truncated

    def capture_so_far(self) -> List[BlockRecord]:
        with self._lock:
            return list(self._capture or [])


@dataclass
class Readiness:
    ready: bool                   # stream + contact OK and stable for required_s
    stream_contact_ok: bool       # OK right now (before the stability period)
    stable_for_s: float
    required_s: float
    blockers: List[str]
    levels: Tuple[str, ...]
    impedance: Tuple[Optional[float], ...]
    emg_age_s: Optional[float]
    segment: Optional[int]
    source_state: str

    def snapshot(self) -> dict:
        return {"ready": self.ready, "stream_contact_ok": self.stream_contact_ok,
                "stable_for_s": self.stable_for_s, "required_s": self.required_s,
                "blockers": list(self.blockers),
                "contact_levels": {f"ch{i + 1}": lv for i, lv in enumerate(self.levels)},
                "impedance_ohm": {f"ch{i + 1}": v for i, v in enumerate(self.impedance)},
                "emg_age_s": self.emg_age_s, "segment": self.segment,
                "source_state": self.source_state}


class ReadinessMonitor:
    """Evaluates readiness on the GUI thread from the tracker's newest record."""

    def __init__(self, tracker: StreamTracker, clock: Callable[[], float] = time.monotonic):
        self.tracker = tracker
        self._clock = clock
        self._blocked_at: Optional[float] = None
        self._segment: Optional[int] = None

    def reset(self):
        """Require the full stability period again (e.g. after every attempt)."""
        self._blocked_at = self._clock()

    def evaluate(self, source_state: str, readiness_s: float, freshness_s: float) -> Readiness:
        now = self._clock()
        self.tracker.freshness_s = freshness_s
        last, good_since = self.tracker.snapshot()
        blockers = []
        levels: Tuple[str, ...] = (UNKNOWN,) * EMG_CHANNELS
        imp: tuple = (None,) * EMG_CHANNELS
        age = None
        segment = None
        if source_state != STREAMING:
            blockers.append(f"device not streaming ({source_state})")
        if last is None:
            blockers.append("no EMG received yet")
        else:
            age = now - last.host_mono
            segment = last.segment
            fresh = source_state == STREAMING and age <= freshness_s
            if age > freshness_s:
                blockers.append(f"EMG stale: last batch {age:.1f} s ago (limit {freshness_s:g} s)")
            if not last.channels_present:
                blockers.append("not all 8 EMG channels delivered valid data")
            if not fresh:
                blockers.append("contact unknown (no fresh EMG)")
            else:
                levels, imp = last.levels, last.impedance   # only trusted on fresh data
                bad = [f"CH{i + 1} {lv}" + (f" ({format_kohm(imp[i])})" if imp[i] is not None
                                            else "")
                       for i, lv in enumerate(levels) if lv not in ACCEPTED_LEVELS]
                if bad:
                    blockers.append("contact not OK: " + ", ".join(bad))
        if segment != self._segment:
            self._segment = segment
            self._blocked_at = now          # reconnect / new segment: start over
        ok_now = not blockers and good_since is not None
        if not ok_now:
            self._blocked_at = now
            stable = 0.0
        else:
            since = good_since if self._blocked_at is None else max(good_since, self._blocked_at)
            stable = max(0.0, now - since)
        ready = ok_now and stable >= readiness_s
        if ok_now and not ready:
            # whole seconds: the text (and the panel layout) changes once per second, not per tick
            blockers.append(f"waiting for stable contact: {math.ceil(readiness_s - stable):d} s")
        return Readiness(ready, ok_now, stable, readiness_s, blockers, levels, imp, age,
                         segment, source_state)


def summarize_window(records: List[BlockRecord], prev: Optional[BlockRecord],
                     t0: float, t1: float, freshness_s: float) -> dict:
    """Stream quality for batches received in [t0, t1] (host monotonic receipt time).

    prev is the last batch received before the capture started (for gaps at the
    start). Gaps, lost and invalid samples are reported, never filled.
    """
    before = [r for r in records if r.host_mono < t0]
    inside = [r for r in records if t0 <= r.host_mono <= t1]
    ref = before[-1] if before else prev
    out = {"window_host_mono": [t0, t1], "batches": len(inside),
           "samples": sum(r.n for r in inside), "lost_samples": sum(r.n_lost for r in inside),
           "samples_with_invalid_channels": sum(r.n_invalid for r in inside),
           "batches_missing_channels": sum(not r.channels_present for r in inside),
           "index_gaps": 0, "missing_samples": 0, "segment_changes": 0,
           "segments": sorted({r.segment for r in inside}),
           "max_receipt_gap_s": None, "contact_levels_seen": {}, "contact_not_ok": [],
           "contact_marginal": [],
           "first_sample_index": inside[0].first_index if inside else None,
           "last_sample_index": inside[-1].last_index if inside else None}
    levels_in_effect = ([ref] if ref is not None else []) + inside
    gaps = []
    last_t = ref.host_mono if ref is not None else None
    for r in inside:
        if last_t is not None:
            gaps.append(r.host_mono - max(last_t, t0))
        last_t = r.host_mono
    if inside:
        gaps.append(t1 - inside[-1].host_mono)
    elif ref is not None:
        gaps.append(t1 - max(ref.host_mono, t0))
    out["max_receipt_gap_s"] = max(gaps) if gaps else None
    p = ref
    for r in inside:
        if p is not None:
            if r.segment != p.segment:
                out["segment_changes"] += 1
            elif r.first_index != p.last_index + 1:
                out["index_gaps"] += 1
                out["missing_samples"] += max(0, r.first_index - p.last_index - 1)
        p = r
    seen = {}
    for r in levels_in_effect:
        for ch, lv in enumerate(r.levels):
            seen.setdefault(f"ch{ch + 1}", set()).add(lv)
    out["contact_levels_seen"] = {k: sorted(v) for k, v in seen.items()}
    out["contact_not_ok"] = [k for k, v in seen.items() if not v <= set(ACCEPTED_LEVELS)]
    out["contact_marginal"] = [k for k, v in seen.items() if MARGINAL in v]   # info only
    flags = []
    if not inside:
        flags.append("no_emg_received")
    if out["contact_not_ok"]:
        flags.append("contact_not_ok:" + ",".join(c.upper() for c in out["contact_not_ok"]))
    if out["lost_samples"]:
        flags.append(f"lost_samples:{out['lost_samples']}")
    if out["samples_with_invalid_channels"]:
        flags.append(f"invalid_samples:{out['samples_with_invalid_channels']}")
    if out["batches_missing_channels"]:
        flags.append("missing_channels")
    if out["index_gaps"]:
        flags.append(f"index_gaps:{out['index_gaps']}({out['missing_samples']} samples)")
    if out["segment_changes"]:
        flags.append("segment_change")
    if out["max_receipt_gap_s"] is not None and out["max_receipt_gap_s"] > freshness_s:
        flags.append(f"receipt_gap:{out['max_receipt_gap_s']:.2f}s")
    out["flags"] = flags
    return out


def contact_events(records: List[BlockRecord], prev: Optional[BlockRecord]) -> List[dict]:
    """Contact-level changes during the capture (first entry = state at start)."""
    events = []
    last = prev.levels if prev is not None else None
    if prev is not None:
        events.append({"host_mono": prev.host_mono, "segment": prev.segment,
                       "levels": list(prev.levels), "impedance_ohm": list(prev.impedance),
                       "initial": True})
    for r in records:
        if r.levels != last:
            events.append({"host_mono": r.host_mono, "segment": r.segment,
                           "levels": list(r.levels), "impedance_ohm": list(r.impedance)})
            last = r.levels
    return events
