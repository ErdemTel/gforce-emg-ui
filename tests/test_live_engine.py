"""Live Classifier engine: SOFTWARE tests with a test-double classifier (not the trained
model, no accuracy claims). Windowing, validity, staleness, overload and lifecycle."""

import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest

from conftest import make_imu, wait_until
from gforce_emg_ui.hub import DataHub, EmgBlock
from gforce_emg_ui.live.engine import (HUB_CHANNELS, INVALID_INPUT, RUNNING, STALE, STOPPED,
                                       WARMING_UP, LiveEngine, LiveSettings)
from gforce_emg_ui.live.log import LogRootError, PredictionLog, check_root
from gforce_emg_ui.recorder import SAVED, Recorder

LABELS = ("low", "medium", "high")


@dataclass(frozen=True)
class FakePred:
    valid: bool
    status: str
    reason: Optional[str] = None
    class_id: Optional[int] = None
    label: Optional[str] = None
    probabilities: Optional[dict] = None
    s_discrete: Optional[float] = None
    s_expected: Optional[float] = None
    max_probability: Optional[float] = None


class FakeClf:
    """Test double: records every window; class = newest ch1 sample index mod 3."""

    def __init__(self, delay=0.0, gate: Optional[threading.Event] = None, invalid=False):
        self.windows, self.calls = [], 0
        self.delay, self.gate, self.invalid = delay, gate, invalid
        self.active = threading.Event()

    def predict_windows(self, windows, *, channel_names):
        assert tuple(channel_names) == HUB_CHANNELS
        self.active.set()
        if self.gate is not None:
            self.gate.wait(10)
        if self.delay:
            time.sleep(self.delay)
        self.calls += 1
        out = []
        for w in windows:
            w = np.asarray(w)
            self.windows.append(w.copy())
            if self.invalid:
                out.append(FakePred(False, "unusable_window", "test double says invalid"))
                continue
            cid = int(round((w[-1, 0]) / 10)) % 3
            p = {l: float(k == cid) for k, l in enumerate(LABELS)}
            out.append(FakePred(True, "ok", None, cid, LABELS[cid], p, cid / 2, cid / 2, 1.0))
        return out


def fake_model(N=10, H=4, **kw):
    return SimpleNamespace(clf=FakeClf(**kw), window_samples=N, hop_samples=H,
                           channel_names=HUB_CHANNELS)


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def block(start, n, t, stream_id=1, rate=1000.0, lost=None, valid=None, data=None, imp=None):
    """Channel-coded EMG: value = sample_index * 10 + channel (exact in float32)."""
    idx = np.arange(start, start + n, dtype=np.int64)
    if data is None:
        data = (idx[None, :] * 10 + np.arange(8)[:, None]).astype(np.float32)
    lost = np.zeros(n, bool) if lost is None else np.asarray(lost, bool)
    valid = np.ones((8, n), bool) if valid is None else np.asarray(valid, bool)
    valid = valid & ~lost[None, :]
    data = np.where(valid, data, np.nan).astype(np.float32)
    return EmgBlock(data=data, valid=valid, sample_index=idx, device_time=idx / rate, lost=lost,
                    host_time=1.7e9 + t, host_mono=t, sample_rate=rate,
                    impedance=np.full(8, 3e5) if imp is None else imp, stream_id=stream_id)


def make_engine(model=None, clock=None, **settings):
    hub = DataHub()
    clock = clock or Clock()
    eng = LiveEngine(hub, LiveSettings(watchdog_s=0, **settings), clock=clock)
    eng.set_model(model or fake_model())
    return hub, eng, clock


def settle(eng, pushed, timeout=5):
    def done():
        c = eng.snapshot().counters
        return c.get("blocks", 0) + c.get("discarded_blocks", 0) + c["dropped_blocks"] >= pushed
    assert wait_until(done, timeout)
    time.sleep(0.02)


def window_bounds(model):
    """(first, last) sample index of each window the classifier received."""
    return [(int(round((w[0, 0]) / 10)), int(round((w[-1, 0]) / 10))) for w in model.clf.windows]


def assert_contiguous(model, N):
    for w in model.clf.windows:
        assert w.shape == (N, 8)
        idx = (w - np.arange(8)[None, :]) / 10           # undo the channel code
        assert np.all(idx == idx[:, :1])                  # same sample in every channel: no swap
        assert np.all(np.diff(idx[:, 0]) == 1)            # consecutive: no gap, pad or duplicate


def feed(hub, sizes, total, t=1000.0, start=0):
    pos, k, n_blocks = start, 0, 0
    while pos < start + total:
        n = min(sizes[k % len(sizes)], start + total - pos)
        hub.push_emg(block(pos, n, t))
        pos += n
        k += 1
        n_blocks += 1
    return n_blocks


# -- schedule / packetization ----------------------------------------------------------

def test_warmup_then_one_window_every_hop_samples():
    model = fake_model(N=10, H=4)
    hub, eng, _ = make_engine(model)
    eng.start()
    hub.push_emg(block(0, 9, 1000.0))
    settle(eng, 1)
    s = eng.snapshot()
    assert s.state == WARMING_UP and s.warm == (9, 10) and model.clf.windows == []
    assert s.current(1000.0) is None
    hub.push_emg(block(9, 1, 1000.0))                     # 10th fresh sample -> first window
    settle(eng, 2)
    assert window_bounds(model) == [(0, 9)]
    n = 2 + feed(hub, [1], 30, start=10)
    settle(eng, n)
    assert window_bounds(model) == [(s0, s0 + 9) for s0 in range(0, 31, 4)]
    assert_contiguous(model, 10)
    snap = eng.snapshot()
    assert snap.state == RUNNING and snap.current(1000.0).window_end_index == 37
    eng.shutdown()


@pytest.mark.parametrize("sizes", [[1], [3], [7], [10], [16], [33], [100], [300],
                                   [5, 1, 17, 2, 64]])
def test_block_size_does_not_change_windows_or_results(sizes):
    model = fake_model(N=10, H=4)
    hub, eng, _ = make_engine(model)
    eng.start()
    settle(eng, feed(hub, sizes, 300))
    assert eng.snapshot().counters["dropped_blocks"] == 0          # no overload
    assert window_bounds(model) == [(s0, s0 + 9) for s0 in range(0, 291, 4)]
    assert_contiguous(model, 10)
    assert eng.snapshot().current(1000.0).window_end_index == 297
    eng.shutdown()


def test_start_after_samples_needs_n_new_samples():
    model = fake_model(N=10, H=4)
    hub, eng, _ = make_engine(model)
    feed(hub, [16], 64)                                   # before Start: never used
    eng.start()
    settle(eng, feed(hub, [16], 32, start=64))
    assert window_bounds(model)[0] == (64, 73)
    eng.reset("user_reset")
    settle(eng, 2 + feed(hub, [5], 15, start=96))
    assert window_bounds(model)[-1][0] >= 96              # reset: N new samples again
    eng.shutdown()


# -- faults --------------------------------------------------------------------------

FAULTS = {
    "index_gap": lambda: block(35, 5, 1000.0),            # 30..34 missing
    "duplicate_index": lambda: block(29, 5, 1000.0),
    "backward_index": lambda: block(10, 5, 1000.0),
    "lost_sample": lambda: block(30, 5, 1000.0, lost=[0, 0, 1, 0, 0]),
    "invalid_channel": lambda: block(30, 5, 1000.0, valid=np.r_[np.ones((7, 5)),
                                                               [[1, 1, 0, 1, 1]]].astype(bool)),
    "nonfinite_value": lambda: block(30, 5, 1000.0, data=np.where(
        np.arange(5) == 2, np.inf, (np.arange(30, 35) * 10)[None, :] + np.arange(8)[:, None])),
    "stream_change": lambda: block(0, 5, 1000.0, stream_id=2),
    "malformed_block": lambda: EmgBlock(
        data=np.zeros((7, 5), np.float32), valid=np.ones((7, 5), bool),
        sample_index=np.arange(30, 35), device_time=np.zeros(5), lost=np.zeros(5, bool),
        host_time=0.0, host_mono=1000.0, sample_rate=1000.0, stream_id=1),
}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_fault_resets_and_never_bridges(fault):
    model = fake_model(N=10, H=4)
    hub, eng, _ = make_engine(model)
    eng.start()
    n = feed(hub, [5], 30)                                # samples 0..29
    settle(eng, n)
    n_before = len(model.clf.windows)
    bad = FAULTS[fault]()
    if fault == "malformed_block":
        bad.segment = 1
        eng._on_emg(bad)                                  # straight in: 7 rows break the hub
    else:
        hub.push_emg(bad)
    settle(eng, n + 1)
    snap = eng.snapshot()
    assert snap.current(1000.0) is None
    assert snap.state in (INVALID_INPUT, WARMING_UP)
    if fault in ("lost_sample", "invalid_channel", "nonfinite_value", "malformed_block"):
        assert fault in snap.reason or snap.state == WARMING_UP
    first_after = int(bad.sample_index[-1]) + 1
    stream = bad.stream_id
    pos, k = first_after, n + 1
    for _ in range(6):
        hub.push_emg(block(pos, 5, 1000.0, stream_id=stream))
        pos += 5
        k += 1
    settle(eng, k)
    assert_contiguous(model, 10)
    bounds = window_bounds(model)
    before, after = bounds[:n_before], bounds[n_before:]
    assert before == [(s0, s0 + 9) for s0 in range(0, 21, 4)]
    first = {"lost_sample": 33, "invalid_channel": 33, "nonfinite_value": 33,  # valid suffix
             "malformed_block": 35}.get(fault, int(bad.sample_index[0]))      # or new run
    assert after[0] == (first, first + 9)                 # exactly N fresh samples later
    assert eng.snapshot().counters["resets"]
    eng.shutdown()


def test_repeated_receipt_times_are_valid_and_channels_never_swap():
    model = fake_model(N=10, H=4)
    hub, eng, _ = make_engine(model)
    eng.start()
    for k in range(10):                                   # 5 batches share each receipt time
        hub.push_emg(block(k * 16, 16, 1000.0 + (k // 5) * 0.15))
    settle(eng, 10)
    assert eng.snapshot().counters["resets"] == {}
    assert_contiguous(model, 10)
    assert window_bounds(model) == [(s0, s0 + 9) for s0 in range(0, 151, 4)]
    eng.shutdown()


def test_imu_and_impedance_never_change_predictions():
    outs = []
    for imp, imu in ((np.full(8, 2e5), False), (np.full(8, 5e6), True)):
        model = fake_model(N=10, H=4)
        hub, eng, _ = make_engine(model)
        eng.start()
        for k in range(8):
            hub.push_emg(block(k * 16, 16, 1000.0, imp=imp * (k + 1)))
            if imu:
                hub.push_imu(make_imu(k * 3, 3))
        settle(eng, 8)
        outs.append(np.stack(model.clf.windows))
        eng.shutdown()
    np.testing.assert_array_equal(outs[0], outs[1])


def test_reported_rate_change_stops_the_run():
    hub, eng, _ = make_engine()
    eng.start()
    hub.push_emg(block(0, 16, 1000.0))
    eng._on_emg(block(16, 16, 1000.0, rate=500.0))
    settle(eng, 2)
    s = eng.snapshot()
    assert s.state == STOPPED and "rate changed" in s.reason and not eng.running
    eng.stop()
    eng.shutdown()


def test_invalid_model_output_is_never_s0():
    model = fake_model(invalid=True)
    hub, eng, _ = make_engine(model)
    eng.start()
    feed(hub, [16], 64)
    settle(eng, 4)
    s = eng.snapshot()
    assert s.state == INVALID_INPUT and s.current(1000.0) is None
    assert s.result is not None and s.result.prediction.s_discrete is None
    assert s.counters["invalid_results"] > 0 and s.counters.get("valid_results", 0) == 0
    eng.shutdown()


# -- staleness / generations ---------------------------------------------------------

def test_output_expires_without_callbacks_and_buffer_restarts():
    model = fake_model(N=10, H=4)
    hub, eng, clock = make_engine(model)
    eng.start()
    settle(eng, feed(hub, [16], 32, t=clock.t))
    assert eng.snapshot().state == RUNNING
    clock.t += 0.4
    eng.check()
    assert eng.snapshot().state == RUNNING                # 0.4 s < 0.5 s
    clock.t += 0.2
    eng.check()                                           # no callbacks arrived meanwhile
    s = eng.snapshot()
    assert s.state == STALE and "no EMG" in s.reason and s.current(clock.t) is None
    assert s.result is None
    n = len(model.clf.windows)
    settle(eng, 2 + feed(hub, [16], 32, t=clock.t, start=32))   # contiguous indices resume
    assert window_bounds(model)[n][0] == 32               # but warm-up starts over
    assert eng.snapshot().state == RUNNING
    eng.shutdown()


def test_prediction_expires_when_inference_stalls_and_late_results_stay_hidden():
    gate = threading.Event()
    model = fake_model(N=10, H=4)
    hub, eng, clock = make_engine(model)
    eng.start()
    settle(eng, feed(hub, [16], 16, t=clock.t))
    assert eng.snapshot().state == RUNNING
    model.clf.gate = gate                                 # inference now blocks
    model.clf.active.clear()
    hub.push_emg(block(16, 16, clock.t + 0.1))
    assert model.clf.active.wait(5)
    for k in range(3):                                    # fresh input keeps arriving
        hub.push_emg(block(32 + 16 * k, 16, clock.t + 0.2 + 0.1 * k))
    clock.t += 0.7
    eng.check()
    s = eng.snapshot()
    assert s.state == STALE and "prediction expired" in s.reason and s.current(clock.t) is None
    model.clf.gate = None
    gate.set()                                            # the stalled call finishes ...
    settle(eng, 5)
    s = eng.snapshot()
    assert s.counters["expired_before_publish"] >= 1      # ... its results expired unseen
    cur = s.current(clock.t)                              # later, still-fresh input may show
    assert cur is None or (cur.window_end_index > 31 and cur.age(clock.t) <= 0.5)
    hub.push_emg(block(80, 16, clock.t))                  # newer input -> fresh output
    settle(eng, 6)
    assert eng.snapshot().current(clock.t).window_end_index == 93     # last 9 + 4k <= 95
    eng.shutdown()


def test_input_staleness_is_still_detected_after_a_prediction_expired():
    gate = threading.Event()
    model = fake_model(N=10, H=4)
    hub, eng, clock = make_engine(model)
    eng.start()
    settle(eng, feed(hub, [16], 16, t=clock.t))
    model.clf.gate = gate                                 # stalled worker
    model.clf.active.clear()
    hub.push_emg(block(16, 16, clock.t + 0.3))
    assert model.clf.active.wait(5)
    clock.t += 0.6
    eng.check()
    assert "prediction expired" in eng.snapshot().reason
    clock.t += 0.3                                        # input now 0.6 s old as well
    eng.check()
    s = eng.snapshot()
    assert s.state == STALE and "no EMG" in s.reason and s.counters["resets"]["input_stale"] == 1
    eng.check()
    assert eng.snapshot().counters["resets"]["input_stale"] == 1   # counted once
    model.clf.gate = None
    gate.set()
    eng.shutdown()


@pytest.mark.parametrize("action", ["stop", "reset", "model"])
def test_old_generation_results_never_revive(action):
    gate = threading.Event()
    model = fake_model(N=10, H=4, gate=gate)
    hub, eng, clock = make_engine(model)
    eng.start()
    hub.push_emg(block(0, 16, clock.t))
    assert model.clf.active.wait(5)                       # inference in flight
    if action == "stop":
        eng.stop()
    elif action == "reset":
        eng.reset()
    else:
        eng.set_model(fake_model())
    gate.set()
    time.sleep(0.2)
    s = eng.snapshot()
    assert s.result is None and s.current(clock.t) is None
    assert s.state == {"stop": STOPPED, "reset": WARMING_UP, "model": "READY"}[action]
    assert s.counters.get("discarded_results", 0) >= 1 or action == "model"
    eng.shutdown()


# -- overload ---------------------------------------------------------------------------

def test_overload_is_bounded_never_stitches_and_keeps_recorder_whole(tmp_path):
    model = fake_model(N=10, H=4, delay=0.05)             # slow classifier
    hub = DataHub()
    eng = LiveEngine(hub, LiveSettings(watchdog_s=0, inbox_samples=128))
    eng.set_model(model)
    rec = Recorder(hub, tmp_path)
    session = rec.start("overload")
    eng.start()
    hub.push_emg(block(0, 16, time.monotonic()))
    assert model.clf.active.wait(5)                       # worker busy in a slow call
    peak = 0
    t0 = time.perf_counter()
    for k in range(1, 300):                               # bursts far faster than inference
        hub.push_emg(block(k * 16, 16, time.monotonic()))
        peak = max(peak, eng.queued_samples)
    push_s = time.perf_counter() - t0
    settle(eng, 300, timeout=10)
    session.request_stop()
    assert session.wait(10) and session.state == SAVED
    assert session.counts["emg"]["written"] == session.counts["emg"]["received"] == 300 * 16
    c = eng.snapshot().counters
    assert peak <= 128 and c["dropped_blocks"] > 0 and c["dropped_samples"] == 16 * c["dropped_blocks"]
    assert c["resets"].get("input_dropped", 0) >= 1      # run restarted after the drop
    assert window_bounds(model)[-1][0] >= 300 * 16 - 128  # recovered to the recent tail
    assert push_s < 1.0                                   # the data thread never waited
    assert_contiguous(model, 10)                          # no window across dropped blocks
    eng.shutdown()


def test_overload_with_real_clock_publishes_only_fresh_results(tmp_path):
    model = fake_model(N=100, H=25, delay=0.35)
    hub = DataHub()
    eng = LiveEngine(hub, LiveSettings(watchdog_s=0.05, inbox_samples=96))
    eng.set_model(model)
    log = PredictionLog(tmp_path, "overload", {"test": "software"}, simulation=True)
    eng.start(log=log, run_id="overload")
    t_end = time.monotonic() + 3.0
    k = 0
    while time.monotonic() < t_end:                       # ~505 samples/s in ~150 ms bursts
        for _ in range(5):
            hub.push_emg(block(k * 16, 16, time.monotonic()))
            k += 1
        time.sleep(0.15)
    eng.stop()
    assert log.wait(5)
    import csv
    rows = list(csv.DictReader(open(log.folder / "predictions.csv")))
    published = [r for r in rows if r["published"] == "1"]
    assert published
    for r in published:
        assert float(r["result_mono"]) - float(r["newest_receipt_mono"]) <= 0.5
    c = eng.snapshot().counters
    assert c["skipped_stale_windows"] + c.get("expired_before_publish", 0) + c["dropped_blocks"] > 0
    eng.shutdown()


# -- lifecycle / logging ------------------------------------------------------------------

def test_repeated_lifecycle_has_one_subscription_and_bounded_threads():
    hub = DataHub()
    eng = LiveEngine(hub, LiveSettings(watchdog_s=0.01))
    before = threading.active_count()
    worst = 0.0
    for k in range(30):
        t0 = time.perf_counter()
        eng.set_model(fake_model())
        eng.start()
        eng.start()                                       # second Start is a no-op
        hub.push_emg(block(k * 16, 16, time.monotonic()))
        eng.reset()
        eng.stop()
        eng.stop()
        worst = max(worst, time.perf_counter() - t0)
        names = [n for n, _ in hub._emg_subscribers]
        assert names.count(LiveEngine.SUB_NAME) == 0
        assert len(eng.threads) <= 2
    eng.start()
    assert [n for n, _ in hub._emg_subscribers].count(LiveEngine.SUB_NAME) == 1
    eng.shutdown()
    assert eng.threads == [] and hub._emg_subscribers == []
    assert threading.active_count() <= before
    assert worst < 0.1                                    # GUI-thread calls never block


def test_engine_module_is_qt_free():
    import gforce_emg_ui.live.compat as c
    import gforce_emg_ui.live.engine as e
    import gforce_emg_ui.live.model as m
    for mod in (c, e, m):
        assert "PyQt5" not in open(mod.__file__).read()


def test_prediction_log_writes_new_folder_and_counts_drops(tmp_path):
    model = fake_model(N=10, H=4)
    hub, eng, clock = make_engine(model)
    log = PredictionLog(tmp_path / "logs", "r1", {"model": {"sha256": "x"}}, simulation=True)
    eng.start(log=log, run_id="r1")
    settle(eng, feed(hub, [16], 160, t=clock.t))
    eng.log_display(1, clock.t, 1.0)
    eng.stop()
    assert log.wait(5) and not log.error
    import csv
    import json
    assert log.folder.parent == tmp_path / "logs" and "sim_live_r1" in log.folder.name
    rows = list(csv.DictReader(open(log.folder / "predictions.csv")))
    assert len(rows) == eng.snapshot().counters["predictions"] == len(model.clf.windows)
    assert rows[0]["window_start_index"] == "0" and rows[0]["window_end_index"] == "9"
    events = [r["event"] for r in csv.DictReader(open(log.folder / "events.csv"))]
    assert events[0] == "start" and events[-1] == "stop"
    assert len(list(csv.DictReader(open(log.folder / "display.csv")))) == 1
    meta = json.load(open(log.folder / "run.json"))
    assert meta["log"]["complete"] and meta["end"]["stop_reason"] == "user_stop"
    small = PredictionLog(tmp_path / "logs", "r2", {}, simulation=False, maxsize=1)
    for _ in range(1000):
        small.put_prediction({"pred_id": 1})              # never blocks
    small.close()
    assert small.wait(5) and small.dropped > 0
    eng.shutdown()


def test_log_root_refuses_recording_and_model_folders(tmp_path):
    trial = tmp_path / "S1" / "trial_a"
    trial.mkdir(parents=True)
    (trial / "trial.json").write_text("{}")
    with pytest.raises(LogRootError):
        check_root(trial / "logs")
    run = tmp_path / "runs" / "lda"
    run.mkdir(parents=True)
    with pytest.raises(LogRootError):
        check_root(run / "x", model_path=run / "model.joblib")
    assert check_root(tmp_path / "live_logs") == (tmp_path / "live_logs").resolve()
