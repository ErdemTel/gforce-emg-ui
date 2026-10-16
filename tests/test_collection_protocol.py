"""Stiffness-collection protocol: labels, settings validation and the trial
state machine, driven by an injected fake monotonic clock (no Qt, no sleeps)."""

import math

import pytest

from gforce_emg_ui.collection.protocol import (ABORTED, DONE, HOLD, PREPARE, REST, SETTLE,
                                               TARGETS, ProtocolSettings, TrialStateMachine,
                                               participant_instructions, phase_instruction,
                                               target_info)


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def machine(clock, **kw):
    return TrialStateMachine(ProtocolSettings(**kw), clock=clock, wall=lambda: 1.7e9 + clock.t)


def run_to_end(m, clock, step=0.01, limit=10_000):
    names = [e.name for e in m.start()]
    for _ in range(limit):
        clock.advance(step)
        names += [e.name for e in m.update()]
        if m.finished:
            return names
    raise AssertionError("did not finish")


ALL = ["PREPARE_START", "SETTLE_START", "HOLD_START", "HOLD_END", "REST_START", "REST_END",
       "TRIAL_END"]


def test_default_protocol_is_28_s_with_hold_5_s():
    s = ProtocolSettings()
    assert (s.prepare_s, s.settle_s, s.hold_s, s.rest_s) == (5.0, 3.0, 5.0, 15.0)
    assert s.total_s == 28.0
    assert s.readiness_s == 3.0 and s.emg_freshness_s == 0.5 and s.rest_extension_s == 15.0
    assert s.validate() == []


def test_phase_order_each_event_once_and_exact_deadlines():
    clock = FakeClock()
    m = machine(clock)
    t0 = clock.t
    assert [e.name for e in m.start()] == ["PREPARE_START"]
    assert m.phase == PREPARE
    clock.advance(4.999)
    assert m.update() == []
    clock.advance(0.001)
    assert [e.name for e in m.update()] == ["SETTLE_START"]
    clock.advance(3.0)
    assert [e.name for e in m.update()] == ["HOLD_START"]
    assert m.phase == HOLD
    clock.advance(5.0)
    assert [e.name for e in m.update()] == ["HOLD_END", "REST_START"]
    clock.advance(15.0)
    assert [e.name for e in m.update()] == ["REST_END", "TRIAL_END"]
    assert m.phase == DONE and m.finished
    assert [e.name for e in m.events] == ALL
    assert [iv["phase"] for iv in m.intervals] == [PREPARE, SETTLE, HOLD, REST]
    assert all(iv["completed"] for iv in m.intervals)
    starts = [iv["start_host_mono"] - t0 for iv in m.intervals]
    assert starts == pytest.approx([0, 5, 8, 13])
    assert m.elapsed() == pytest.approx(28.0)
    assert not [f for f in m.flags if f["type"] == "overrun"] and m.abort_reason is None
    # this test ticks only at the deadlines: the long gaps between ticks are recorded
    assert {f["type"] for f in m.flags} == {"tick_gap"}
    for _ in range(3):                     # nothing more happens after the end
        clock.advance(10)
        assert m.update() == []


def test_configurable_durations():
    clock = FakeClock()
    m = machine(clock, prepare_s=1.0, settle_s=2.0, hold_s=3.0, rest_s=4.0)
    names = run_to_end(m, clock, step=0.05)
    assert names == ALL
    actual = [iv["actual_s"] for iv in m.intervals]
    assert actual == pytest.approx([1.0, 2.0, 3.0, 4.0], abs=0.051)
    assert [iv["planned_s"] for iv in m.intervals] == [1.0, 2.0, 3.0, 4.0]


def test_monotonic_deadlines_not_decrement_counters():
    """Irregular tick intervals do not change phase lengths beyond one tick."""
    clock = FakeClock()
    m = machine(clock, prepare_s=1.0, settle_s=1.0, hold_s=1.0, rest_s=1.0)
    m.start()
    for dt in [0.013, 0.071, 0.002, 0.049] * 200:
        clock.advance(dt)
        m.update()
        if m.finished:
            break
    for iv in m.intervals:
        assert 1.0 <= iv["actual_s"] < 1.0 + 0.072


def test_delayed_tick_does_not_backdate_or_skip_a_phase():
    clock = FakeClock()
    m = machine(clock, prepare_s=1.0, settle_s=0.2, hold_s=0.2, rest_s=1.0,
                overrun_flag_s=0.1, overrun_abort_s=1.0)
    m.start()
    clock.advance(1.0)
    m.update()                              # SETTLE starts exactly on time
    settle_start = clock.t
    clock.advance(0.5)                      # GUI stalled past the SETTLE *and* HOLD deadlines
    events = m.update()
    assert [e.name for e in events] == ["HOLD_START"]      # one transition, HOLD not skipped
    assert m.phase == HOLD
    hold = m.intervals[-1]
    assert hold["start_host_mono"] == clock.t               # starts when shown, not backdated
    assert m.intervals[-2]["overrun_s"] == pytest.approx(0.3)
    assert m.intervals[-2]["actual_s"] == pytest.approx(clock.t - settle_start)
    assert any(f["type"] == "overrun" and f["phase"] == SETTLE for f in m.flags)
    clock.advance(0.1)
    assert m.update() == []                 # HOLD still runs its full 0.2 s
    clock.advance(0.1)
    assert [e.name for e in m.update()] == ["HOLD_END", "REST_START"]
    assert m.intervals[2]["actual_s"] == pytest.approx(0.2)


def test_large_overrun_before_or_in_hold_aborts_instead_of_faking_hold():
    for stall_phase, waits in ((SETTLE, [1.0]), (HOLD, [1.0, 0.2])):
        clock = FakeClock()
        m = machine(clock, prepare_s=1.0, settle_s=0.2, hold_s=0.2, rest_s=1.0,
                    overrun_abort_s=1.0)
        m.start()
        for w in waits:
            clock.advance(w)
            m.update()
        assert m.phase == stall_phase
        clock.advance(3.0)                  # GUI frozen for 3 s
        names = [e.name for e in m.update()]
        assert names == ["ABORT", "TRIAL_END"]
        assert m.phase == ABORTED and m.abort_reason == "timing_overrun"
        assert m.events[-2].detail["phase"] == stall_phase
        if stall_phase == SETTLE:
            assert "HOLD_START" not in [e.name for e in m.events]
        assert not m.intervals[-1]["completed"]
        assert m.consume_stop_request() and not m.consume_stop_request()


def test_overrun_at_end_of_rest_is_recorded_not_aborted():
    clock = FakeClock()
    m = machine(clock, prepare_s=0.1, settle_s=0.1, hold_s=0.1, rest_s=0.1)
    m.start()
    for _ in range(3):
        clock.advance(0.1)
        m.update()
    assert m.phase == REST
    clock.advance(2.0)
    assert [e.name for e in m.update()] == ["REST_END", "TRIAL_END"]
    assert m.phase == DONE
    assert any(f["type"] == "overrun" and f["phase"] == REST for f in m.flags)


def test_rest_extension_only_in_rest_and_logged_once_per_click():
    clock = FakeClock()
    m = machine(clock, prepare_s=1, settle_s=1, hold_s=1, rest_s=2, rest_extension_s=15)
    m.start()
    assert m.extend_rest() == []            # PREPARE: not allowed
    clock.advance(1)
    m.update()
    clock.advance(1)
    m.update()
    assert m.phase == HOLD and m.extend_rest() == []
    clock.advance(1)
    m.update()
    assert m.phase == REST
    ev = m.extend_rest()
    assert [e.name for e in ev] == ["REST_EXTENDED"]
    assert ev[0].detail == {"extension_s": 15, "count": 1}
    assert m.intervals[-1]["planned_s"] == 17
    clock.advance(2.5)
    assert m.update() == []                 # deadline moved by 15 s
    assert m.phase_remaining() == pytest.approx(14.5)
    m.extend_rest()
    assert m.rest_extensions == 2
    clock.advance(29.5)
    assert [e.name for e in m.update()] == ["REST_END", "TRIAL_END"]
    assert [e.name for e in m.events].count("REST_EXTENDED") == 2
    assert m.intervals[-1]["actual_s"] == pytest.approx(32.0)


def test_abort_ends_protocol_once_and_requests_stop_once():
    clock = FakeClock()
    m = machine(clock, prepare_s=1, settle_s=1, hold_s=5, rest_s=1)
    m.start()
    clock.advance(1)
    m.update()
    clock.advance(1)
    m.update()
    clock.advance(2)
    m.update()
    assert m.phase == HOLD
    ev = m.abort("user_abort")
    assert [e.name for e in ev] == ["ABORT", "TRIAL_END"]
    assert ev[0].detail["reason"] == "user_abort" and ev[0].detail["during"] == HOLD
    hold = m.intervals[-1]
    assert hold["phase"] == HOLD and not hold["completed"] and hold["actual_s"] == pytest.approx(2)
    assert m.abort("again") == [] and m.update() == [] and m.extend_rest() == []
    assert m.consume_stop_request() is True
    assert m.consume_stop_request() is False
    assert m.abort_reason == "user_abort"


def test_exactly_one_stop_request_after_normal_end():
    clock = FakeClock()
    m = machine(clock, prepare_s=0.1, settle_s=0.1, hold_s=0.1, rest_s=0.1)
    run_to_end(m, clock)
    requests = sum(m.consume_stop_request() for _ in range(5))
    m.abort("late")
    clock.advance(1)
    m.update()
    requests += sum(m.consume_stop_request() for _ in range(5))
    assert requests == 1


def test_start_twice_is_an_error():
    clock = FakeClock()
    m = machine(clock)
    m.start()
    with pytest.raises(RuntimeError):
        m.start()


@pytest.mark.parametrize("field", ["prepare_s", "settle_s", "hold_s", "rest_s"])
@pytest.mark.parametrize("bad", [0, -1.0, math.nan, math.inf, -math.inf, "5", None, True, 1e9])
def test_invalid_durations_are_rejected(field, bad):
    s = ProtocolSettings(**{field: bad})
    assert s.validate(), (field, bad)
    with pytest.raises(ValueError):
        TrialStateMachine(s)


def test_abort_threshold_must_exceed_flag_threshold():
    assert ProtocolSettings(overrun_flag_s=1.0, overrun_abort_s=0.5).validate()
    assert ProtocolSettings(readiness_s=0.0).validate() == []


def test_settings_are_frozen_at_trial_start():
    clock = FakeClock()
    s = ProtocolSettings(prepare_s=1.0)
    m = TrialStateMachine(s, clock=clock)
    s.prepare_s = 99.0                       # later edits do not affect a running trial
    m.start()
    clock.advance(1.0)
    assert [e.name for e in m.update()] == ["SETTLE_START"]


def test_explicit_label_mapping_and_source():
    assert {k: (v["class_id"], v["s"]) for k, v in TARGETS.items()} == {
        "low": (0, 0.0), "medium": (1, 0.5), "high": (2, 1.0)}
    assert target_info("medium") == {"target_label": "medium", "target_class_id": 1,
                                     "target_s": 0.5, "label_source": "instructed_effort"}


def test_instructions_follow_the_protocol_wording():
    assert phase_instruction(PREPARE, "low").startswith("Position your hand/object.")
    assert phase_instruction(HOLD, "high") == ("Maintain HIGH effort. Keep your hand and "
                                               "wrist position steady.")
    assert phase_instruction(REST, "low").startswith("Release the effort and rest.")
    assert "Do not open the hand" in phase_instruction(SETTLE, "low")
    assert "between low and high" in phase_instruction(SETTLE, "medium")
    high = phase_instruction(SETTLE, "high")
    assert "not maximal" in high and "discomfort or fatigue" in high
    for target in TARGETS:
        for text in participant_instructions(target).values():
            assert "as hard as possible" not in text.lower()
            assert "50%" not in text
    # SETTLE is not described as the steady hold; HOLD is
    assert "Maintain" not in phase_instruction(SETTLE, "medium")
