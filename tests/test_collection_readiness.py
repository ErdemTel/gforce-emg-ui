"""Readiness gate (stream freshness + impedance contact) and stream-quality
summaries, with a fake monotonic clock and hand-built EMG blocks."""

import numpy as np
import pytest

from conftest import make_emg
from gforce_emg_ui.collection.readiness import (ReadinessMonitor, StreamTracker,
                                                channel_levels, contact_events,
                                                summarize_window)
from gforce_emg_ui.contact import GOOD, MARGINAL, POOR, UNKNOWN
from gforce_emg_ui.hub import DataHub
from gforce_emg_ui.sources import IDLE, RECONNECTING, STREAMING

GOOD_IMP = np.full(8, 250_000.0)


class FakeClock:
    def __init__(self, t=500.0):
        self.t = t

    def __call__(self):
        return self.t


class Rig:
    """Hub + tracker + monitor; push() delivers a block received 'now'."""

    def __init__(self, freshness=0.5, readiness=3.0):
        self.clock = FakeClock()
        self.hub = DataHub()
        self.tracker = StreamTracker(freshness)
        self.hub.subscribe_emg(self.tracker.on_emg)
        self.monitor = ReadinessMonitor(self.tracker, self.clock)
        self.freshness, self.readiness = freshness, readiness
        self.index = 0
        self.stream = 1

    def push(self, imp=GOOD_IMP, valid=None, lost=None, stream_id=None, n=16):
        if stream_id is not None and stream_id != self.stream:
            self.stream, self.index = stream_id, 0
        b = make_emg(self.index, n=n, imp=imp, valid=valid, lost=lost, stream_id=self.stream)
        b.host_mono = self.clock.t
        self.hub.push_emg(b)
        self.index += n
        return b

    def run(self, seconds, step=0.05, state=STREAMING, **push_kw):
        """Deliver a block every `step` s for `seconds`, evaluating after each."""
        r = None
        for _ in range(int(round(seconds / step))):
            self.clock.t += step
            self.push(**push_kw)
            r = self.evaluate(state)
        return r

    def evaluate(self, state=STREAMING):
        return self.monitor.evaluate(state, self.readiness, self.freshness)


def test_channel_levels_use_existing_thresholds_and_invalid_is_unknown():
    imp = [200e3, 500e3, 750e3, 1.5e6, np.nan, 0.0, -5.0, np.inf]
    assert channel_levels(imp) == (GOOD, GOOD, MARGINAL, POOR, UNKNOWN, UNKNOWN, UNKNOWN,
                                   UNKNOWN)
    assert channel_levels(None) == (UNKNOWN,) * 8
    assert channel_levels([200e3] * 6) == (GOOD,) * 6 + (UNKNOWN,) * 2   # missing channels


def test_all_good_becomes_ready_only_after_the_stability_period():
    rig = Rig(readiness=3.0)
    r = rig.run(2.0)
    assert r.stream_contact_ok and not r.ready
    assert r.stable_for_s == pytest.approx(1.95, abs=0.06)
    assert any("waiting for stable contact" in b for b in r.blockers)
    r = rig.run(1.1)
    assert r.ready and r.blockers == []
    assert r.levels == (GOOD,) * 8


def test_no_data_yet_is_not_ready():
    rig = Rig()
    r = rig.evaluate()
    assert not r.ready and "no EMG received yet" in r.blockers


@pytest.mark.parametrize("ch,value,word", [(7, 1.4e6, "poor"), (5, 1.0e6, "poor"),
                                           (2, np.nan, "unknown"), (0, 0.0, "unknown"),
                                           (4, -1.0, "unknown"), (3, np.inf, "unknown")])
def test_one_bad_or_invalid_channel_blocks_readiness(ch, value, word):
    rig = Rig(readiness=0.5)
    imp = GOOD_IMP.copy()
    imp[ch] = value
    r = rig.run(2.0, imp=imp)
    assert not r.ready and not r.stream_contact_ok
    text = " ".join(r.blockers)
    assert f"CH{ch + 1} {word}" in text
    assert r.stable_for_s == 0.0


def test_orange_contact_is_accepted():
    rig = Rig(readiness=0.5)
    imp = GOOD_IMP.copy()
    imp[[1, 5]] = (700e3, 999e3)                     # orange (<= 999 kOhm), no red
    r = rig.run(1.0, imp=imp)
    assert r.ready and r.blockers == []
    assert r.levels[1] == r.levels[5] == MARGINAL
    r = rig.run(1.0, imp=np.full(8, 900e3))          # all orange
    assert r.ready


def test_unchanged_good_impedance_on_fresh_emg_stays_ready():
    """contact.csv logs changes only: an unchanged value is not stale while EMG is fresh."""
    rig = Rig(readiness=1.0)
    r = rig.run(10.0, imp=GOOD_IMP)              # identical impedance for 10 s
    assert r.ready and r.stable_for_s == pytest.approx(9.95, abs=0.06)


def test_stale_stream_is_not_ready_and_stability_restarts():
    rig = Rig(freshness=0.5, readiness=1.0)
    assert rig.run(2.0).ready
    rig.clock.t += 0.6                           # no batches for 0.6 s
    r = rig.evaluate()
    assert not r.ready and any("stale" in b for b in r.blockers)
    assert r.levels == (UNKNOWN,) * 8            # old values are not shown as current
    r = rig.run(0.5)                             # data resumes (same segment)
    assert not r.ready and r.stable_for_s < 0.6
    assert rig.run(0.6).ready


def test_frozen_green_values_after_disconnect_do_not_count():
    rig = Rig(readiness=0.5)
    assert rig.run(1.0).ready
    for state in (IDLE, RECONNECTING):
        r = rig.evaluate(state)                  # last block was all-good, but no link
        assert not r.ready
        assert any("not streaming" in b for b in r.blockers)
        assert r.levels == (UNKNOWN,) * 8


def test_reconnect_new_segment_requires_full_stability_again():
    rig = Rig(readiness=1.0)
    assert rig.run(2.0).ready
    rig.clock.t += 0.05
    rig.push(stream_id=2)                         # reconnection: new stream -> new segment
    r = rig.evaluate()
    assert rig.hub.emg_segment == 2
    assert not r.ready and r.stable_for_s < 0.1
    assert not rig.run(0.9).ready
    assert rig.run(0.2).ready


def test_bad_block_between_evaluations_restarts_stability():
    rig = Rig(readiness=1.0)
    assert rig.run(2.0).ready
    bad = GOOD_IMP.copy()
    bad[3] = 2e6
    rig.clock.t += 0.02
    rig.push(imp=bad)                             # blip never seen by evaluate()
    rig.clock.t += 0.02
    rig.push()
    r = rig.evaluate()
    assert not r.ready and r.stable_for_s < 0.05


def test_missing_channel_blocks_readiness():
    rig = Rig(readiness=0.2)
    valid = np.ones((8, 16), bool)
    valid[6] = False
    r = rig.run(1.0, valid=valid)
    assert not r.ready
    assert any("not all 8 EMG channels" in b for b in r.blockers)


def test_low_amplitude_emg_is_never_a_blocker():
    rig = Rig(readiness=0.5)
    for _ in range(30):
        rig.clock.t += 0.05
        b = make_emg(rig.index, value=0.0, imp=GOOD_IMP)    # flat, tiny signal
        b.host_mono = rig.clock.t
        rig.hub.push_emg(b)
        rig.index += 16
        r = rig.evaluate()
    assert r.ready


def test_monitor_reset_requires_the_period_again():
    rig = Rig(readiness=1.0)
    assert rig.run(2.0).ready
    rig.monitor.reset()
    assert not rig.run(0.5).ready
    assert rig.run(0.6).ready


# -- stream quality ---------------------------------------------------------------

def _capture(rig, blocks):
    rig.tracker.start_capture()
    for kw in blocks:
        rig.clock.t += kw.pop("dt", 0.05)
        if "index" in kw:
            rig.index = kw.pop("index")
        rig.push(**kw)
    return rig.tracker.stop_capture()


def test_quality_summary_reports_gaps_lost_invalid_and_contact_without_filling():
    rig = Rig()
    rig.clock.t += 0.05
    rig.push()
    lost = np.zeros(16, bool)
    lost[:3] = True
    marg = GOOD_IMP.copy()
    marg[2] = 800e3
    prev, recs, trunc = _capture(rig, [{}, {"lost": lost}, {"index": 100}, {"imp": marg},
                                       {"dt": 0.7}])
    assert not trunc and len(recs) == 5
    t0, t1 = recs[0].host_mono - 0.01, recs[-1].host_mono
    q = summarize_window(recs, prev, t0, t1, 0.5)
    assert q["batches"] == 5 and q["samples"] == 80
    assert q["lost_samples"] == 3 and q["samples_with_invalid_channels"] == 3
    assert q["index_gaps"] == 1 and q["missing_samples"] == 100 - 48
    assert q["contact_not_ok"] == [] and q["contact_marginal"] == ["ch3"]   # orange: info
    assert q["max_receipt_gap_s"] == pytest.approx(0.7)
    flags = " ".join(q["flags"])
    for word in ("lost_samples:3", "invalid_samples:3", "index_gaps:1", "receipt_gap"):
        assert word in flags
    assert "contact" not in flags
    ev = contact_events(recs, prev)
    assert ev[0]["initial"] and ev[0]["levels"] == [GOOD] * 8
    assert ev[1]["levels"][2] == MARGINAL
    assert len(ev) == 3 and ev[2]["levels"] == [GOOD] * 8        # recovered


def test_quality_window_only_counts_batches_inside_and_contact_in_effect_at_start():
    rig = Rig()
    poor = GOOD_IMP.copy()
    poor[7] = 3e6
    prev, recs, _ = _capture(rig, [{"imp": poor}, {}, {}, {}])
    # window starts after the poor batch, but the poor level was still in effect at t0
    t0 = recs[0].host_mono + 0.01
    q = summarize_window(recs, prev, t0, recs[-1].host_mono, 0.5)
    assert q["batches"] == 3 and q["contact_not_ok"] == ["ch8"]
    assert "contact_not_ok:CH8" in q["flags"]
    # a window entirely after recovery is clean
    q2 = summarize_window(recs, prev, recs[1].host_mono + 0.01, recs[-1].host_mono, 0.5)
    assert q2["flags"] == []


def test_segment_change_inside_window_is_flagged():
    rig = Rig()
    prev, recs, _ = _capture(rig, [{}, {"stream_id": 2}, {}])
    q = summarize_window(recs, prev, recs[0].host_mono, recs[-1].host_mono, 0.5)
    assert q["segment_changes"] == 1 and "segment_change" in q["flags"]


def test_empty_window_is_flagged_no_data():
    rig = Rig()
    prev, recs, _ = _capture(rig, [{}])
    q = summarize_window(recs, prev, recs[-1].host_mono + 1, recs[-1].host_mono + 2, 0.5)
    assert "no_emg_received" in q["flags"] and any(f.startswith("receipt_gap") for f in q["flags"])
