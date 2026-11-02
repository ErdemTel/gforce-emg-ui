"""Output stabilization: SOFTWARE tests with synthetic prediction streams, fake clocks and
test-double classifiers. No accuracy claim: without ground truth a rejected transition is
not an "error corrected", and synthetic data says nothing about human EMG."""

import csv
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import wait_until
from gforce_emg_ui.hub import DataHub
from gforce_emg_ui.live.engine import (HUB_CHANNELS, PREDICTION_FIELDS, RUNNING, STALE,
                                       LiveEngine, LiveSettings)
from gforce_emg_ui.live.log import PredictionLog
from gforce_emg_ui.live.stabilization import (CONFIRMING, GATE_NOT_MET, HOLDING, LABELS, OFF,
                                              PENDING, UNSUPPORTED, WAITING, Observation,
                                              SettingsError, Stabilizer, StabilizerSettings,
                                              describe)
from test_live_engine import Clock, FakePred, block, settle

ON = StabilizerSettings(enabled=True)                    # 100 / 100 / 100 ms, gate off
RAW_PREDICTION_FIELDS = [                                # predictions.csv before this feature
    "pred_id", "run_id", "generation", "stream_id", "segment", "window_start_index",
    "window_end_index", "first_receipt_mono", "newest_receipt_mono", "newest_receipt_wall",
    "result_mono", "inference_ms", "batch_size", "published", "valid", "status", "reason",
    "class_id", "label", "p_low", "p_medium", "p_high", "s_discrete", "s_expected",
    "max_probability", "dropped_blocks", "dropped_samples", "skipped_stale_windows",
    "expired_before_publish", "resets"]


# -- pure state machine ----------------------------------------------------------------

def probs_for(label, p=1.0):
    return {l: (p if l == label else (1 - p) / 2) for l in LABELS}


class Seq:
    """Feeds observations with supplied receipt times; consecutive windows (hop 25)."""

    def __init__(self, settings=ON):
        self.st = Stabilizer(settings)
        self.pid = 0
        self.end = 100
        self.rows = []

    def feed(self, label, t, p=1.0, probs=None, valid=True, result=None, end=None, gen=1,
             pid=None):
        if pid is None:
            self.pid += 1
            pid = self.pid
        self.end = self.end + 25 if end is None else end
        ob = Observation(pid, "r", gen, 1, 0, self.end, 25, t,
                         t + 0.002 if result is None else result, valid,
                         label if valid else None,
                         (probs or probs_for(label, p)) if valid else None)
        row = self.st.observe(ob, now=t + 0.003)
        self.rows.append(row)
        return self.st.state()

    def hold(self, label, t0, t1, step=0.05):
        """Continuous evidence for label every step s from t0 to t1."""
        for k in range(int(round((t1 - t0) / step)) + 1):
            s = self.feed(label, round(t0 + k * step, 6))
        return s


def test_stabilization_module_is_qt_free():
    import gforce_emg_ui.live.stabilization as m
    assert "PyQt5" not in open(m.__file__).read()


def test_settings_defaults_and_validation():
    d = StabilizerSettings()
    assert not d.enabled and (d.up_ms, d.down_ms, d.startup_ms) == (100, 100, 100)
    assert not d.gate and (d.min_probability, d.min_margin) == (0.60, 0.10)
    assert (d.max_gap_ms, d.max_unsupported_ms) == (250, 1000)
    assert d.errors(500) == []
    bad = {"up_ms": 1001, "down_ms": -1, "min_probability": 1.5, "max_gap_ms": 501,
           "max_unsupported_ms": 99}
    for k, v in bad.items():
        assert StabilizerSettings(**{k: v}).errors(500), k
    assert StabilizerSettings(max_gap_ms=300).errors(250)        # above the stale limit
    assert StabilizerSettings(enabled=1).errors(500)              # bool required
    assert StabilizerSettings(up_ms=True).errors(500)
    assert StabilizerSettings(up_ms=float("nan")).errors(500)
    assert StabilizerSettings(up_ms=100.5).errors(500)
    assert "expire first" in StabilizerSettings(up_ms=900, max_unsupported_ms=1000).errors(500)[0]
    good = d.to_dict()
    assert StabilizerSettings.from_dict({**good, "up_ms": 200.0}, 500).up_ms == 200
    for broken in ({**good, "up_ms": 2000}, {k: v for k, v in good.items() if k != "gate"},
                   {**good, "extra": 1}, ["not", "a", "dict"]):
        with pytest.raises(SettingsError):
            StabilizerSettings.from_dict(broken, 500)


def test_off_mode_is_the_raw_output_including_validity():
    st = Stabilizer(StabilizerSettings(enabled=False, gate=True, min_probability=0.99))
    rng = np.random.default_rng(0)
    t = 10.0
    for pid in range(1, 300):
        t += float(rng.choice([0.0, 0.0003, 0.15, 0.4]))
        valid = rng.random() > 0.1
        label = str(rng.choice(LABELS))
        ob = Observation(pid, "r", 1, 1, 0, 100 + 25 * pid, 25, t, t, valid,
                         label if valid else None, probs_for(label, 0.34) if valid else None)
        st.observe(ob, t)
        s = st.state()
        assert s.status == OFF and s.pending is None
        assert s.accepted == (label if valid else None)          # gate bypassed when OFF


def test_brief_excursion_does_not_switch():
    q = Seq()
    q.feed("low", 0.00)
    assert q.st.state().accepted is None                         # startup: s = --
    q.feed("low", 0.05)
    assert q.feed("low", 0.10).accepted == "low"
    s = q.feed("medium", 0.15)
    assert s.accepted == "low" and s.status == PENDING and s.pending_elapsed_ms == 0
    s = q.feed("medium", 0.20)                                   # 50 ms excursion
    assert s.accepted == "low" and s.pending_elapsed_ms == pytest.approx(50)
    s = q.feed("low", 0.25)
    assert s.accepted == "low" and s.pending is None and s.status == HOLDING
    assert q.rows[-1]["reason"] == "returned_to_accepted" and q.rows[2]["reason"] == "confirmed"
    c = s.counters
    assert c.get("accepted_changes", 0) == 0 and c["cancelled:returned_to_accepted"] == 1
    assert c["raw_changes"] == 2


def test_persistent_change_accepted_at_first_observation_reaching_the_threshold():
    q = Seq()
    q.hold("low", 0.0, 0.95)
    s = q.feed("medium", 1.0)
    for t in (1.05, 1.0999):
        s = q.feed("medium", t)
        assert s.accepted == "low"
    s = q.feed("medium", 1.1)                                    # exactly 100 ms
    assert s.accepted == "medium" and s.accepted_pred_id == q.pid
    assert s.accepted_start_mono == 1.0 and s.accepted_evidence_mono == 1.1
    assert s.accepted_decision_mono == pytest.approx(1.103)      # not backdated to 1.0
    assert q.rows[-1]["transition"] == "low->medium"


def test_upward_and_downward_durations_are_independent():
    q = Seq(StabilizerSettings(enabled=True, up_ms=50, down_ms=200, startup_ms=0))
    assert q.hold("low", 0.0, 0.95).accepted == "low"
    q.feed("high", 1.0)
    assert q.feed("high", 1.05).accepted == "high"               # up: 50 ms
    q.hold("high", 1.1, 1.95)
    q.feed("low", 2.0)
    for t in (2.05, 2.1, 2.15, 2.199):
        assert q.feed("low", t).accepted == "high"
    assert q.feed("low", 2.2).accepted == "low"                  # down: 200 ms
    q.hold("low", 2.25, 2.95)
    q.feed("medium", 3.0)
    assert q.feed("medium", 3.05).accepted == "medium"           # low -> medium is up


def test_direct_low_high_changes_obey_confirmation():
    q = Seq(StabilizerSettings(enabled=True, startup_ms=0))
    q.hold("low", 0.0, 0.95)
    assert q.feed("high", 1.0).accepted == "low"                 # HIGH never bypasses
    assert q.feed("high", 1.1).accepted == "high"
    assert q.rows[-1]["transition"] == "low->high"               # no MEDIUM step
    q.hold("high", 1.15, 1.95)
    q.feed("low", 2.0)
    assert q.feed("low", 2.1).accepted == "low"
    assert q.rows[-1]["transition"] == "high->low"
    assert q.st.state().counters["accepted_changes"] == 2


def test_startup_and_reset_are_unavailable_until_confirmed():
    q = Seq(StabilizerSettings(enabled=True, startup_ms=200))
    s = q.st.state()
    assert s.accepted is None and s.status == WAITING and s.s is None
    for t in (0.0, 0.1, 0.19):
        s = q.feed("low", t)
        assert s.accepted is None and s.status == CONFIRMING     # never LOW by default
    assert q.feed("low", 0.2).accepted == "low"
    q.st.reset("user_reset", 0.3)
    s = q.st.state()
    assert s.accepted is None and s.status == WAITING and s.reason == "user_reset"
    assert q.feed("low", 0.31).accepted is None                  # confirm again


def test_candidate_change_and_disconnected_bursts_restart_from_zero():
    q = Seq()
    q.hold("low", 0.0, 0.95)
    q.feed("medium", 1.0)
    q.feed("medium", 1.06)
    s = q.feed("high", 1.08)                                     # different candidate
    assert s.pending == "high" and s.pending_elapsed_ms == 0
    q.feed("medium", 1.12)
    s = q.feed("medium", 1.21)
    assert s.accepted == "low" and s.pending_elapsed_ms == pytest.approx(90)
    assert q.feed("medium", 1.22).accepted == "medium"
    q2 = Seq()                                                   # MED, LOW, MED: no sum
    q2.hold("low", 0.0, 0.95)
    q2.feed("medium", 1.0)
    q2.feed("low", 1.05)
    q2.feed("medium", 1.08)
    s = q2.feed("medium", 1.15)
    assert s.accepted == "low" and s.pending_elapsed_ms == pytest.approx(70)


def test_probability_gate_resets_pending_and_uses_margin_only_with_an_accepted_class():
    g = StabilizerSettings(enabled=True, gate=True, min_probability=0.6, min_margin=0.1)
    q = Seq(g)
    s = q.feed("low", 0.0, p=0.55)                               # below minimum
    assert s.status == GATE_NOT_MET and s.accepted is None and s.pending is None
    q.feed("low", 0.05, probs={"low": 0.62, "medium": 0.37, "high": 0.01})
    assert q.feed("low", 0.15, p=0.7).accepted == "low"          # no margin at startup
    q.hold("low", 0.2, 0.95)
    q.feed("medium", 1.0, p=0.7)
    s = q.feed("medium", 1.05, p=0.55)                           # gate fails -> cleared
    assert s.pending is None and s.status == GATE_NOT_MET and s.accepted == "low"
    q.feed("medium", 1.08, p=0.7)
    q.feed("medium", 1.15, p=0.7)
    s = q.feed("medium", 1.17, probs={"low": 0.32, "medium": 0.64, "high": 0.04})
    assert s.accepted == "low" and s.pending == "medium"          # margin 0.32 ok
    s = q.feed("medium", 1.18, probs={"low": 0.36, "medium": 0.44, "high": 0.2})
    assert s.pending is None                                     # p < 0.6 and margin 0.08
    q.feed("medium", 1.2, probs={"low": 0.0, "medium": 0.65, "high": 0.35})
    assert q.feed("medium", 1.3, p=0.9).accepted == "medium"
    assert q.st.state().counters["gate_failures"] == 3
    assert "probability gate not met" in describe(Seq(g).feed("high", 0.0, p=0.4)).lower()


def test_zero_durations():
    q = Seq(StabilizerSettings(enabled=True, up_ms=0, down_ms=0, startup_ms=0))
    for label, t in (("low", 0.0), ("high", 0.0), ("medium", 0.0), ("low", 0.01)):
        assert q.feed(label, t).accepted == label                # immediate, no gate
    q = Seq(StabilizerSettings(enabled=True, up_ms=0, down_ms=0, startup_ms=0, gate=True))
    assert q.feed("low", 0.0, p=0.5).accepted is None            # gate still applies


def test_reads_duplicates_and_out_of_order_results_add_no_evidence():
    q = Seq()
    q.hold("low", 0.0, 0.95)
    q.feed("medium", 1.0)
    before = q.st.state()
    for _ in range(100):                                         # GUI refreshes
        assert q.st.state().pending_elapsed_ms == before.pending_elapsed_ms == 0
    s = q.feed("medium", 1.5, pid=q.pid)                         # duplicate pred_id
    assert s.pending_elapsed_ms == 0 and q.rows[-1]["kind"] == "ignored"
    s = q.feed("medium", 1.5, pid=q.pid - 1)                     # older pred_id
    assert s.pending_elapsed_ms == 0 and s.accepted == "low"
    assert s.counters["ignored_duplicate"] == 2


def test_equal_receipt_times_and_slow_inference_add_no_time():
    q = Seq()
    q.hold("low", 0.0, 0.95)
    for k in range(10):                                          # one burst, one receipt time
        s = q.feed("medium", 1.0, result=1.0 + 0.4 * k)          # inference done much later
        assert s.accepted == "low" and s.pending_elapsed_ms == 0
    s = q.feed("medium", 1.05, result=3.0)
    assert s.pending_elapsed_ms == pytest.approx(50) and s.accepted == "low"


def test_evidence_gap_and_window_discontinuity_restart_pending():
    q = Seq()
    q.hold("low", 0.0, 0.95)
    q.feed("medium", 1.0)
    s = q.feed("medium", 1.3)                                    # 300 ms > 250 ms gap
    assert s.accepted == "low" and s.pending_elapsed_ms == 0 and s.pending_start_mono == 1.3
    assert q.rows[-1]["reason"] == "evidence_gap"
    s = q.feed("medium", 1.35, end=q.end + 50)                   # a window is missing
    assert s.pending_start_mono == 1.35 and q.rows[-1]["reason"] == "window_discontinuity"
    q.st.break_evidence("expired_before_publish", 1.36)
    assert q.st.state().pending is None
    assert q.feed("medium", 1.40).pending_start_mono == 1.40


def test_unsupported_hold_expires_and_is_not_stale():
    q = Seq(StabilizerSettings(enabled=True, startup_ms=0))
    q.feed("low", 0.0)
    t, labels = 0.0, ["medium", "high"]
    for k in range(40):                                          # never one class 100 ms
        t += 0.05
        s = q.feed(labels[k % 2], t)
        if t <= 1.0:
            assert s.accepted == "low"
        if s.accepted is None:
            break
    assert s.accepted is None and s.status == UNSUPPORTED and t > 1.0
    assert s.reason == "unsupported_hold_expired" and s.counters["unsupported_expiries"] == 1
    assert describe(s).startswith("Output unavailable: unsupported hold expired")
    assert q.rows[-1]["reason"] == "unsupported_hold_expired"
    s = q.feed("high", t + 0.05)                                  # startup rule again (0 ms)
    assert s.accepted == "high" and s.status == HOLDING


def test_apply_boundary_survives_later_engine_resets():
    q = Seq(StabilizerSettings(enabled=True, startup_ms=0))
    q.feed("low", 1.0)
    q.st.configure(StabilizerSettings(enabled=True, startup_ms=0), 1.05)
    q.st.reset("index_gap", 1.06)                                # in-worker reset after Apply
    s = q.feed("high", 1.04)                                     # queued before Apply
    assert s.accepted is None and q.rows[-1]["kind"] == "ignored"
    assert q.feed("high", 1.07).accepted == "high"
    q.st.reset("start", 2.0, new_run=True)
    assert q.feed("low", 1.5).accepted == "low"                  # new run: no old boundary


def test_invalid_and_context_change_clear_the_state():
    q = Seq(StabilizerSettings(enabled=True, startup_ms=0))
    assert q.feed("low", 0.0).accepted == "low"
    s = q.feed("low", 0.05, valid=False)
    assert s.accepted is None and s.reason == "invalid_prediction"
    assert q.feed("low", 0.1).accepted == "low"
    s = q.feed("low", 0.15, probs={"low": float("nan"), "medium": 0.0, "high": 0.0})
    assert s.accepted is None                                    # non-finite = invalid
    q.feed("low", 0.2)
    s = q.feed("high", 0.25, gen=2)                               # new generation: restart
    assert s.accepted == "high" and q.st.state().counters["reset:context_change"] == 1


def test_describe_texts():
    q = Seq()
    assert describe(q.feed("low", 0.0)) == "Confirming initial class: LOW pending: 0 / 100 ms"
    q.feed("low", 0.1)
    assert describe(q.st.state()) == "Holding LOW"
    q.feed("medium", 0.2)
    assert describe(q.feed("medium", 0.25)) == "Holding LOW; MEDIUM pending: 50 / 100 ms"
    assert describe(Stabilizer().state()) == "Stabilization OFF"


def test_replay_demo_counts_raw_versus_accepted_changes(capsys):
    """Known sequence in ~150 ms bursts of 3 windows (like the band's dongle backend):
    brief excursions and sustained changes. A software check, not physiological evidence."""
    script = [("low", 8), ("medium", 1), ("low", 5), ("high", 6), ("low", 1), ("high", 4),
              ("medium", 6), ("low", 0.34), ("medium", 4)]       # 0.34 = one window
    q = Seq()
    t, labels = 1.0, []
    for label, bursts in script:
        n = max(1, int(round(bursts * 3)))
        for k in range(n):
            labels.append(label)
            q.feed(label, t + (k % 3) * 0.0004)
            if k % 3 == 2:
                t += 0.15
        if n % 3:
            t += 0.15
    c = q.st.state().counters
    raw_changes = sum(a != b for a, b in zip(labels, labels[1:]))
    accepted = [r["transition"] for r in q.rows if r["transition"]]
    with capsys.disabled():
        print(f"\n[replay demo] {len(labels)} windows: raw class changes {raw_changes}, "
              f"accepted changes {c['accepted_changes']} (+{c['initial_acceptances']} initial: "
              f"{accepted}), cancelled candidates {c['cancelled_candidates']}")
    assert c["raw_changes"] == raw_changes == 8
    assert accepted == ["none->low", "low->high", "high->medium"]
    assert c["cancelled_candidates"] == 3


# -- engine integration ----------------------------------------------------------------

class ScriptClf:
    """Test double: newest sample's ch1 = class id, ch2 = winner probability x 1000,
    ch3 != 0 -> invalid output."""

    def __init__(self):
        self.gate = None
        self.active = threading.Event()

    def predict_windows(self, windows, *, channel_names):
        self.active.set()
        if self.gate is not None:
            self.gate.wait(10)
        out = []
        for w in windows:
            cid, pw, bad = np.asarray(w)[-1, :3]
            if bad:
                out.append(FakePred(False, "unusable_window", "scripted"))
                continue
            cid = int(cid)
            p = probs_for(LABELS[cid], pw / 1000)
            out.append(FakePred(True, "ok", None, cid, LABELS[cid], p, cid / 2,
                                p["medium"] / 2 + p["high"], max(p.values())))
        return out


class Rig:
    """Engine with a fake clock; one 5-sample block -> one window (N=10, H=5)."""

    def __init__(self, settings=None, log=None, warm=None):
        self.hub = DataHub()
        self.clock = Clock(1.0)
        self.eng = LiveEngine(self.hub, LiveSettings(watchdog_s=0), clock=self.clock)
        self.clf = ScriptClf()
        self.eng.set_model(SimpleNamespace(clf=self.clf, window_samples=10, hop_samples=5,
                                           channel_names=HUB_CHANNELS))
        if settings is not None:
            self.eng.set_stabilization(settings)
        self.idx = self.pushed = 0
        self.eng.start(log=log, run_id="rig")
        if warm:
            self.push(warm, 0.95)                                  # first 5 samples

    def push(self, label, t, p=1.0, bad=False, wait=True, skip=0):
        self.clock.t = t
        self.idx += skip
        data = np.zeros((8, 5), np.float32)
        data[0], data[1], data[2] = LABELS.index(label), p * 1000, float(bad)
        self.hub.push_emg(block(self.idx, 5, t, data=data))
        self.idx += 5
        self.pushed += 1
        if wait:
            settle(self.eng, self.pushed)

    def st(self):
        return self.eng.snapshot().stabilization


def test_engine_off_mode_matches_raw_current_exactly():
    rig = Rig(warm="low")
    script = [("low", 1.0, {}), ("medium", 1.05, {}), ("high", 1.1, {}), ("high", 1.1, {}),
              ("low", 1.2, {"bad": True}), ("low", 1.25, {}), ("medium", 1.3, {"skip": 3}),
              ("medium", 1.35, {}), ("high", 1.4, {}), ("low", 1.45, {})]
    for label, t, kw in script:
        rig.push(label, t, **kw)
        for now in (t, t + 0.3, t + 0.6):                        # freshness at read time
            snap = rig.eng.snapshot()
            raw, stab = snap.current(now), snap.stable(now)
            assert (raw is None) == (stab is None), (label, t, now)
            if raw is not None:
                assert stab.accepted == raw.prediction.label and stab.s == raw.prediction.s_discrete
    rig.clock.t = 2.1
    rig.eng.check()                                              # watchdog: stale
    snap = rig.eng.snapshot()
    assert snap.current(2.1) is None and snap.stable(2.1) is None
    rig.eng.shutdown()


def test_engine_processes_every_result_of_a_batch_in_order():
    rig = Rig(ON, warm="low")
    rig.push("low", 1.0)
    rig.push("low", 1.1)
    assert rig.st().accepted == "low"
    gate = threading.Event()
    rig.clf.gate = gate
    rig.clf.active.clear()
    rig.push("medium", 1.20, wait=False)
    assert rig.clf.active.wait(5)                                # inference in flight
    for label, t in (("medium", 1.25), ("high", 1.28), ("medium", 1.31), ("medium", 1.35)):
        rig.push(label, t, wait=False)
    rig.clf.gate = None
    gate.set()
    settle(rig.eng, rig.pushed)
    snap = rig.eng.snapshot()
    assert snap.result.batch_size == 4                           # one batch of 4 windows
    st = snap.stabilization
    assert st.accepted == "low" and st.pending == "medium"       # HIGH was not skipped
    assert st.pending_start_mono == pytest.approx(1.31) and st.pending_elapsed_ms == pytest.approx(40)
    assert st.counters["cancelled:candidate_changed"] == 2
    rig.eng.shutdown()


def test_engine_equal_receipts_and_late_results_do_not_fabricate_time():
    rig = Rig(ON, warm="low")
    rig.push("low", 1.0)
    rig.push("low", 1.1)
    gate = threading.Event()
    rig.clf.gate = gate
    rig.clf.active.clear()
    rig.push("medium", 1.2, wait=False)
    assert rig.clf.active.wait(5)
    for _ in range(3):
        rig.push("medium", 1.2, wait=False)                      # same receipt time
    rig.clock.t = 1.35                                           # results 150 ms later
    rig.clf.gate = None
    gate.set()
    settle(rig.eng, rig.pushed)
    st = rig.st()
    assert st.accepted == "low" and st.pending_elapsed_ms == 0
    assert rig.eng.snapshot().result.result_mono == 1.35
    for _ in range(20):                                          # watchdog / reads: nothing
        rig.eng.check(1.35)
        rig.eng.snapshot()
    assert rig.st().pending_elapsed_ms == 0 and rig.st().accepted == "low"
    rig.push("medium", 1.4)                                      # 200 ms of receipt time
    assert rig.st().accepted == "medium"
    rig.eng.shutdown()


@pytest.mark.parametrize("how", ["invalid", "stale", "disconnect", "link_lost", "index_gap",
                                 "old_generation"])
def test_engine_invalidations_clear_and_never_restore(how):
    rig = Rig(ON, warm="low")
    rig.push("low", 1.0)
    rig.push("low", 1.1)
    assert rig.st().accepted == "low"
    if how == "invalid":
        rig.push("low", 1.15, bad=True)
    elif how == "stale":
        snap = rig.eng.snapshot()
        assert snap.stable(1.7) is None                          # read-time freshness first
        rig.eng.check(1.7)
        assert rig.eng.snapshot().state == STALE
    elif how == "disconnect":
        rig.eng.stop("disconnected")
    elif how == "link_lost":
        rig.eng.reset("link_lost")
    elif how == "index_gap":
        rig.push("low", 1.15, skip=2)
    else:
        gate = threading.Event()
        rig.clf.gate = gate
        rig.clf.active.clear()
        rig.push("low", 1.15, wait=False)
        assert rig.clf.active.wait(5)
        rig.eng.reset("user_reset")                              # in-flight result is old
        rig.clf.gate = None
        gate.set()
        settle(rig.eng, rig.pushed)
    snap = rig.eng.snapshot()
    assert snap.stabilization.accepted is None and snap.stable(snap.result.newest_receipt_mono
                                                               if snap.result else 1.2) is None
    if how == "disconnect":
        assert not rig.eng.running
    else:
        for t in (1.2, 1.25, 1.3):                               # warm-up / startup again
            rig.push("low", t)
            st = rig.st()
            if st.accepted is not None:
                break
        assert st.accepted is None or st.accepted_evidence_mono - st.accepted_start_mono >= 0.1
    rig.eng.shutdown()


def test_engine_unsupported_expiry_is_separate_from_stale_input():
    rig = Rig(StabilizerSettings(enabled=True, gate=True, startup_ms=0), warm="low")
    rig.push("low", 1.0)
    assert rig.st().accepted == "low"
    t = 1.0
    while t < 2.2:
        t = round(t + 0.1, 3)
        rig.push("medium", t, p=0.5)                             # fresh, but gate never met
    snap = rig.eng.snapshot()
    assert snap.state == RUNNING and snap.current(t).prediction.label == "medium"
    st = snap.stabilization
    assert st.accepted is None and st.status == UNSUPPORTED and snap.stable(t) is None
    assert st.reason == "unsupported_hold_expired"
    rig.eng.check(t + 0.6)                                       # now the input is stale
    st = rig.st()
    assert rig.eng.snapshot().state == STALE and st.status == WAITING and st.reason == "stale"
    rig.eng.shutdown()


def test_apply_mid_run_is_atomic_and_never_reuses_old_results():
    rig = Rig(StabilizerSettings(enabled=True, up_ms=0, down_ms=0, startup_ms=0), warm="low")
    rig.push("low", 1.0)
    rev0 = rig.eng.stabilizer_revision
    assert rig.st().accepted == "low"
    gate = threading.Event()
    rig.clf.gate = gate
    rig.clf.active.clear()
    rig.push("high", 1.10, wait=False)
    assert rig.clf.active.wait(5)                                # result computed meanwhile
    rig.clock.t = 1.12
    new = StabilizerSettings(enabled=True, up_ms=300, down_ms=300)
    assert rig.eng.set_stabilization(new) == rev0 + 1
    st = rig.st()
    assert rig.eng.stabilizer_settings == new and st.revision == rev0 + 1
    assert st.accepted is None and st.pending is None and st.reason == "settings_applied"
    rig.clf.gate = None
    gate.set()
    settle(rig.eng, rig.pushed)
    snap = rig.eng.snapshot()
    assert snap.current(1.12).prediction.label == "high"         # raw output unaffected
    assert snap.stabilization.accepted is None and snap.stabilization.pending is None
    assert snap.stabilization.counters["ignored_before_boundary"] == 1
    rig.push("high", 1.15)
    assert rig.st().accepted is None and rig.st().pending == "high"
    rig.push("high", 1.25)                                       # startup 100 ms after 1.15
    assert rig.st().accepted == "high" and rig.eng.running
    raw, marks = rig.eng.history(), rig.eng.stabilized_history()[1]
    sh = rig.eng.stabilized_history()[0]
    assert 1.12 in marks
    assert not np.isnan(raw[:, 1]).any()                         # raw history unbroken
    i = int(np.flatnonzero(sh[:, 0] == 1.12)[0])
    assert np.isnan(sh[i, 1]) and sh[i - 1, 1] == 0.0            # break, old history kept
    with pytest.raises(ValueError):
        rig.eng.set_stabilization(StabilizerSettings(max_gap_ms=600))
    assert rig.eng.stabilizer_revision == rev0 + 1               # refused: nothing changed
    rig.eng.shutdown()


def test_logs_keep_raw_predictions_and_record_settings_transitions_and_resets(tmp_path):
    log = PredictionLog(tmp_path, "rig", {"test": "software"}, simulation=True)
    rig = Rig(ON, log=log, warm="low")
    for label, t in (("low", 1.0), ("low", 1.1), ("medium", 1.2), ("medium", 1.25),
                     ("low", 1.3), ("high", 1.4), ("high", 1.45), ("high", 1.5)):
        rig.push(label, t)
    rig.clock.t = 1.55
    rig.eng.set_stabilization(StabilizerSettings(enabled=True, startup_ms=0))
    rig.push("low", 1.6)
    rig.clock.t = 1.62
    rig.eng.reset_stabilization()
    rig.push("low", 1.65)
    rig.eng.stop()
    assert log.wait(5) and not log.error
    rd = lambda name: list(csv.DictReader(open(log.folder / name)))
    preds = rd("predictions.csv")
    with open(log.folder / "predictions.csv") as f:
        assert next(csv.reader(f)) == RAW_PREDICTION_FIELDS == PREDICTION_FIELDS
    stab = rd("stabilization.csv")
    obs = [r for r in stab if r["kind"] == "observe"]
    assert [r["pred_id"] for r in obs] == [r["pred_id"] for r in preds]
    by_id = {r["pred_id"]: r for r in preds}
    for r in obs:
        p = by_id[r["pred_id"]]
        assert (r["raw_label"], r["p_low"], r["p_high"]) == (p["label"], p["p_low"], p["p_high"])
        assert r["run_id"] == p["run_id"] == "rig"
    assert [r["transition"] for r in obs if r["transition"]] == [
        "none->low", "low->high", "none->low", "none->low"]
    first = obs[0]
    assert first["output_valid"] == "0" and first["accepted_label"] == "" \
        and first["accepted_s"] == "" and first["status"] == CONFIRMING
    hold = [r for r in obs if r["status"] == HOLDING and not r["transition"]][0]
    assert hold["accepted_pred_id"] != hold["pred_id"]           # established earlier
    settings = [r for r in stab if r["kind"] == "settings"]
    assert [r["reason"] for r in settings] == ["active", "settings_applied"]
    assert int(settings[1]["revision"]) == int(settings[0]["revision"]) + 1
    assert json.loads(settings[1]["settings"])["startup_ms"] == 0
    assert {r["revision"] for r in obs} == {settings[0]["revision"], settings[1]["revision"]}
    resets = [r["reason"] for r in stab if r["kind"] == "reset"]
    assert resets[0] == "start" and "user_stabilization_reset" in resets \
        and resets[-1] == "user_stop"
    ignored = [r for r in stab if r["kind"] == "ignored"]
    assert all(r["reason"] == "evidence_before_revision_boundary" for r in ignored)
    events = rd("events.csv")
    kinds = [e["event"] for e in events]
    assert kinds.count("stabilization_settings") == 2 and "stabilization_reset" in kinds
    assert json.loads(events[kinds.index("stabilization_settings")]["detail"])["algorithm"]
    meta = json.load(open(log.folder / "run.json"))
    end = meta["end"]["stabilization"]
    assert end["algorithm"] == "persistence_confirmation" and end["counters"]["observations"]
    rig.eng.shutdown()


# -- UI --------------------------------------------------------------------------------

def make_dash(tmp_path, store=None):
    from PyQt5 import QtCore
    from gforce_emg_ui.sources import SimSource
    from gforce_emg_ui.ui.live_view import LiveClassifierDashboard
    hub = DataHub()
    store = store or QtCore.QSettings(str(tmp_path / "live.ini"), QtCore.QSettings.IniFormat)
    dash = LiveClassifierDashboard(hub, SimSource(hub, seed=1), log_root=tmp_path / "logs",
                                   store=store)
    return dash, store


def test_default_settings_store_is_isolated_in_tests(qapp, tmp_path):
    from pathlib import Path
    from gforce_emg_ui.ui.live_view import settings_store
    path = Path(settings_store().fileName())
    assert Path.home() / ".config" not in path.parents and "qsettings" in str(path)


def test_ui_draft_apply_persist_and_restore_defaults(qapp, tmp_path):
    from gforce_emg_ui.ui.live_view import STORE_KEY
    dash, store = make_dash(tmp_path)
    try:
        assert not dash.stab_enable.isChecked() and dash.engine.stabilizer_settings == \
            StabilizerSettings()                                 # first run: OFF, defaults
        assert (dash.stab_up.value(), dash.stab_down.value(), dash.stab_startup.value()) == \
            (100, 100, 100)
        assert not dash.stab_gate.isChecked() and dash.stab_minp.value() == 0.6
        assert dash.stab_gap.maximum() == 500 and dash.stab_gap.value() == 250
        assert dash.mode_label.text() == "Raw output — stabilization OFF"
        assert not dash.btn_stab_apply.isEnabled() and store.value(STORE_KEY) is None
        assert "No unapplied changes" in dash.stab_draft_label.text()
        dash.stab_enable.setChecked(True)
        dash.stab_up.setValue(200)
        assert dash.stab_dirty and dash.btn_stab_apply.isEnabled()
        assert "Unapplied changes" in dash.stab_draft_label.text()
        assert not dash.engine.stabilizer_settings.enabled      # draft only
        dash.btn_stab_apply.click()
        active = dash.engine.stabilizer_settings
        assert active.enabled and active.up_ms == 200 and dash.engine.stabilizer_revision == 1
        assert not dash.stab_dirty and "Active (rev 1): ON" in dash.stab_draft_label.text()
        assert json.loads(store.value(STORE_KEY))["up_ms"] == 200
        dash.restore_stabilization_defaults()
        assert dash.stab_up.value() == 100 and dash.stab_enable.isChecked()
        assert dash.stab_dirty and dash.engine.stabilizer_settings.up_ms == 200
        dash.stab_hold.setValue(100)                             # < 100 + 250
        assert not dash.btn_stab_apply.isEnabled() and "Invalid" in dash.stab_draft_label.text()
        assert not dash.apply_stabilization() and dash.engine.stabilizer_revision == 1
    finally:
        dash.shutdown()
    dash2, _ = make_dash(tmp_path, store)                        # persisted, applied at start
    try:
        assert dash2.stab_enable.isChecked() and dash2.stab_up.value() == 200
        assert dash2.engine.stabilizer_settings == active and not dash2.stab_dirty
    finally:
        dash2.shutdown()


@pytest.mark.parametrize("stored", ["{not json", json.dumps({"enabled": True}),
                                    json.dumps({**StabilizerSettings(enabled=True).to_dict(),
                                                "max_gap_ms": 9999})])
def test_ui_ignores_corrupt_stored_settings(qapp, tmp_path, stored):
    from PyQt5 import QtCore
    from gforce_emg_ui.ui.live_view import SCHEMA_KEY, STORE_KEY
    store = QtCore.QSettings(str(tmp_path / "bad.ini"), QtCore.QSettings.IniFormat)
    store.setValue(SCHEMA_KEY, 1)
    store.setValue(STORE_KEY, stored)
    dash, _ = make_dash(tmp_path, store)
    try:
        assert dash.engine.stabilizer_settings == StabilizerSettings()
        assert not dash.stab_enable.isChecked() and "ignored" in dash.stab_draft_label.text()
    finally:
        dash.shutdown()
    store.setValue(SCHEMA_KEY, 99)                               # unknown schema version
    store.setValue(STORE_KEY, json.dumps(StabilizerSettings(enabled=True).to_dict()))
    dash, _ = make_dash(tmp_path, store)
    try:
        assert not dash.engine.stabilizer_settings.enabled
    finally:
        dash.shutdown()


from test_live_model import needs_model                          # noqa: E402
from test_live_ui import load, make, make_compatible, shut       # noqa: E402


@needs_model
def test_ui_stabilized_display_hidden_tab_toggles_and_apply_while_running(qapp, tmp_path):
    win, src, hub = make(qapp, tmp_path)
    live = win.live
    try:
        load(qapp, live)
        make_compatible(qapp, live, src)
        live.stab_enable.setChecked(True)
        live.btn_stab_advanced.setChecked(True)
        live.stab_startup.setValue(0)
        assert live.apply_stabilization()
        live.btn_start.click()
        assert wait_until(lambda: live.engine.snapshot().stabilization.accepted is not None
                          and live.s_label.text() in ("s = 0", "s = 0.5", "s = 1"), 10, qapp)
        assert live.mode_label.text() == "Stabilized command"
        assert live.raw_label.text().startswith("Raw: ")
        assert live.stab_label.text().startswith("Stabilization: ")
        assert wait_until(lambda: live.stab_curve.getData()[0] is not None
                          and len(live.stab_curve.getData()[0]) > 0, 5, qapp)

        win.tabs.setCurrentIndex(0)                               # hidden: still filtering
        assert not live.refresh()
        n0 = live.engine.snapshot().stabilization.counters["observations"]
        assert wait_until(lambda: live.engine.snapshot().stabilization.counters[
            "observations"] > n0 + 5, 5, qapp)
        win.tabs.setCurrentWidget(live)

        live.show_stab.setChecked(False)                          # display only
        live.show_raw.setChecked(False)
        assert not live.stab_curve.isVisible() and not live.s_curve.isVisible()
        n1 = live.engine.snapshot().stabilization.counters["observations"]
        assert wait_until(lambda: live.engine.snapshot().stabilization.counters[
            "observations"] > n1, 5, qapp)
        live.show_stab.setChecked(True)
        live.show_raw.setChecked(True)

        model, rev = live.engine.model, live.engine.stabilizer_revision
        live.stab_up.setValue(300)
        assert live.apply_stabilization() and live.engine.running
        assert live.engine.model is model and live.engine.stabilizer_revision == rev + 1
        live.stab_enable.setChecked(False)
        assert live.apply_stabilization()
        assert wait_until(lambda: live.mode_label.text() == "Raw output — stabilization OFF",
                          5, qapp)
        assert wait_until(lambda: (lambda c: c is not None and live.s_label.text()
                                   == f"s = {c.prediction.s_discrete:g}")(
            live.engine.snapshot().current(live._clock())), 5, qapp)
        assert live.stab_label.text() == "Stabilization OFF"
        assert live.engine.snapshot().counters["resets"] == {}   # buffer never restarted
    finally:
        shut(qapp, win, src)
