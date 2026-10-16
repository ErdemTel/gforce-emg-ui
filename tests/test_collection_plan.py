"""200-trial plan, progress from persisted trial.json records, and candidate
selection (no Qt, no hardware)."""

import collections
import json

import pytest

from gforce_emg_ui.collection.plan import (DEFAULT_PLAN_COUNTS, EMPTY_HAND, TrialPlan,
                                           eligible_trials,
                                           attempt_category, compute_progress,
                                           counts_toward_plan, generate_order, max_target_run,
                                           merge_objects, participant_root, pick_trial,
                                           scan_participant, target_counts, training_candidate)
from gforce_emg_ui.collection.storage import (atomic_write_text, attempt_name, dumps,
                                              ensure_within, jsonable, s_tag, valid_id)


def test_default_counts_total_200_and_67_67_66():
    assert sum(sum(c) for c in DEFAULT_PLAN_COUNTS.values()) == 200
    assert target_counts(DEFAULT_PLAN_COUNTS) == {"low": 67, "medium": 67, "high": 66}
    assert DEFAULT_PLAN_COUNTS[EMPTY_HAND] == (13, 13, 14)
    assert {k: sum(v) for k, v in DEFAULT_PLAN_COUNTS.items()} == {
        EMPTY_HAND: 40, "obj01": 32, "obj02": 32, "obj03": 32, "obj04": 32, "obj05": 32}
    assert DEFAULT_PLAN_COUNTS["obj04"] == (11, 10, 11)
    assert DEFAULT_PLAN_COUNTS["obj05"] == (10, 11, 11)


def test_plan_has_exact_cell_counts_and_postures():
    plan = TrialPlan.create("P01", "real", seed=7)
    assert plan.total == 200
    cells = collections.Counter((t["object_id"], t["target_label"]) for t in plan.trials)
    for obj, (lo, me, hi) in DEFAULT_PLAN_COUNTS.items():
        assert (cells[(obj, "low")], cells[(obj, "medium")], cells[(obj, "high")]) == (lo, me, hi)
    assert [t["trial_id"] for t in plan.trials] == [f"t{i:03d}" for i in range(1, 201)]
    assert all(t["posture_id"] == "fist" for t in plan.trials if t["object_id"] == EMPTY_HAND)
    assert all(t["posture_id"] == "grasp" for t in plan.trials if t["object_id"] != EMPTY_HAND)


def test_plan_order_is_reproducible_from_the_seed():
    a = generate_order(DEFAULT_PLAN_COUNTS, 12345)
    b = generate_order(DEFAULT_PLAN_COUNTS, 12345)
    c = generate_order(DEFAULT_PLAN_COUNTS, 12346)
    assert a == b and a != c
    p1, p2 = TrialPlan.create("P01", "real", seed=99), TrialPlan.create("P01", "real", seed=99)
    assert p1.trials == p2.trials and p1.plan_id != p2.plan_id    # same order, distinct IDs


@pytest.mark.parametrize("seed", range(40))
def test_no_run_of_more_than_two_identical_targets(seed):
    order = generate_order(DEFAULT_PLAN_COUNTS, seed)
    assert max_target_run(order) <= 2


def test_targets_are_spread_over_the_whole_plan():
    for seed in range(10):
        order = generate_order(DEFAULT_PLAN_COUNTS, seed)
        for start in range(0, 200, 50):          # every quarter has every target
            block = collections.Counter(t for _, t in order[start:start + 50])
            assert min(block[t] for t in ("low", "medium", "high")) >= 8, (seed, start, block)


def test_plan_round_trips_through_json(tmp_path):
    plan = TrialPlan.create("P01", "simulation", seed=3)
    path = tmp_path / plan.filename
    atomic_write_text(path, dumps(plan.to_dict()))
    loaded = TrialPlan.from_dict(json.loads(path.read_text()))
    assert loaded.trials == plan.trials and loaded.seed == 3 and loaded.plan_id == plan.plan_id
    d = json.loads(path.read_text())
    assert d["counts_low_medium_high"]["obj04"] == [11, 10, 11] and d["total"] == 200
    with pytest.raises(ValueError):
        TrialPlan.from_dict({"schema": "something else"})


# -- synthetic trial.json records --------------------------------------------------

def rec(plan, trial_id, attempt, *, status="reviewed", protocol="completed", recorder="saved",
        metadata="ok", review="accepted", quality="clean", role="development",
        source=None, target=None, t=0.0):
    planned = plan.trial(trial_id)
    source = source or plan.data_source
    return {"attempt_id": attempt, "trial_id": trial_id, "plan_id": plan.plan_id,
            "participant_id": plan.participant_id, "dataset_role": role,
            "data_source": source, "is_simulation": source == "simulation", "status": status,
            "condition": {"object_id": planned["object_id"],
                          "motion_condition": planned.get("motion_condition")},
            "label": {"target_label": target or planned["target_label"], "target_class_id": 0,
                      "target_s": 0.0, "label_source": "instructed_effort"},
            "outcome": {"protocol": protocol, "recorder": recorder, "metadata": metadata},
            "review": {"status": review},
            "training_candidate": {"phase": "HOLD", "quality_status": quality,
                                   "start_host_mono": 10.0, "end_host_mono": 15.0},
            "timing": {"start": {"host_time": t}}}


def test_progress_counts_each_planned_trial_once_and_keeps_categories_apart():
    plan = TrialPlan.create("P01", "real", seed=1)
    t1, t2, t3, t4 = (plan.trials[i]["trial_id"] for i in range(4))
    records = {r["attempt_id"]: r for r in [
        rec(plan, t1, "a", review="rejected", t=1),            # rejected, then repeated
        rec(plan, t1, "b", t=2),                               # accepted
        rec(plan, t1, "c", t=3),                               # duplicate accept: once only
        rec(plan, t2, "d", protocol="aborted", review="rejected", t=4),       # invalid
        rec(plan, t2, "e", status="recording", review="pending", t=5),        # interrupted
        rec(plan, t3, "f", review="pending", status="finalized", t=6),        # unreviewed
        rec(plan, t3, "g", recorder="error", t=7),             # error: invalid even if accepted
        rec(plan, t4, "h", quality="flagged", t=8),            # flagged: never counts
        rec(plan, t4, "i", metadata="error: disk full", t=9),  # metadata failed: invalid
    ]}
    p = compute_progress(plan, records)
    assert p.completed == {t1: "b"}
    assert (p.accepted, p.remaining, p.attempted) == (1, 199, 9)
    assert (p.rejected, p.invalid, p.unreviewed) == (1, 4, 1)
    planned = plan.trial(t1)
    assert p.by_cell[(planned["object_id"], planned["target_label"])][0] == 1
    assert sum(v[0] for v in p.by_cell.values()) == 1
    assert sum(v[1] for v in p.by_cell.values()) == 200
    assert p.attempts_by_trial[t1] == 3


def test_simulation_practice_and_mismatched_records_never_count():
    plan = TrialPlan.create("P01", "real", seed=1)
    t1 = plan.trials[0]["trial_id"]
    other = "high" if plan.trials[0]["target_label"] != "high" else "low"
    for bad in (rec(plan, t1, "s", source="simulation"),
                rec(plan, t1, "p", role="practice"),
                rec(plan, t1, "x", target=other)):
        assert attempt_category(bad) == "accepted"
        assert not counts_toward_plan(bad, plan)
        assert compute_progress(plan, {bad["attempt_id"]: bad}).accepted == 0
    sim_plan = TrialPlan.create("P01", "simulation", seed=1)
    sim = rec(sim_plan, t1, "z")
    assert counts_toward_plan(sim, sim_plan) and not counts_toward_plan(sim, plan)


def test_live_attempt_is_in_progress_not_interrupted():
    plan = TrialPlan.create("P01", "real", seed=1)
    r = rec(plan, "t001", "live", status="recording", review="pending")
    assert attempt_category(r) == "invalid"
    assert attempt_category(r, live={"live"}) == "in_progress"
    assert compute_progress(plan, {"live": r}, live={"live"}).in_progress == 1


def test_only_available_objects_and_empty_hand_are_eligible():
    plan = TrialPlan.create("P01", "real", seed=1)
    p = compute_progress(plan, {})
    idx = eligible_trials(plan, p, {"obj02"})
    assert {plan.trials[i]["object_id"] for i in idx} == {"obj02", EMPTY_HAND}
    assert len(idx) == 32 + 40
    assert {plan.trials[i]["object_id"] for i in eligible_trials(plan, p, set())} == {EMPTY_HAND}
    empty = [t["trial_id"] for t in plan.trials if t["object_id"] == EMPTY_HAND]
    done = {f"a{k}": rec(plan, tid, f"a{k}") for k, tid in enumerate(empty)}
    p_done = compute_progress(plan, done)                 # empty hand complete: drops out
    assert eligible_trials(plan, p_done, set()) == []
    assert pick_trial(plan, p_done, set()) is None
    assert {plan.trials[i]["object_id"] for i in eligible_trials(plan, p_done, {"obj05"})} == {"obj05"}


def test_pick_is_random_reproducible_and_limited_to_available():
    plan = TrialPlan.create("P01", "real", seed=3)
    p = compute_progress(plan, {})
    picks = [pick_trial(plan, p, {"obj01", "obj04"}, key=str(k)) for k in range(60)]
    assert {plan.trials[i]["object_id"] for i in picks} == {"obj01", "obj04", EMPTY_HAND}
    assert len(set(picks)) > 30                                    # spread, not plan order
    assert picks == [pick_trial(plan, p, {"obj01", "obj04"}, key=str(k)) for k in range(60)]
    first_incomplete = 0
    assert sum(i == first_incomplete for i in picks) < 5


def test_pick_avoids_a_third_identical_target_and_the_rejected_trial():
    plan = TrialPlan.create("P01", "real", seed=3)
    p = compute_progress(plan, {})
    for k in range(50):
        i = pick_trial(plan, p, {"obj01", "obj02"}, recent_targets=["high", "high"], key=str(k))
        assert plan.trials[i]["target_label"] != "high"
        j = pick_trial(plan, p, {"obj01", "obj02"}, key=str(k), exclude=i)
        assert j != i
    # only one eligible trial left: it is picked even if excluded or a third in a row
    ids = [t["trial_id"] for t in plan.trials]
    keep = next(k for k, t in enumerate(plan.trials) if t["object_id"] == "obj03")
    done = {f"a{k}": rec(plan, tid, f"a{k}") for k, tid in enumerate(ids) if k != keep}
    p1 = compute_progress(plan, done)
    target = plan.trials[keep]["target_label"]
    assert pick_trial(plan, p1, {"obj03"}, recent_targets=[target] * 2, exclude=keep) == keep


def test_training_candidate_is_hold_only_and_requires_clean_accepted():
    plan = TrialPlan.create("P01", "real", seed=1)
    ok = rec(plan, "t001", "a")
    c = training_candidate(ok)
    assert c["start_host_mono"] == 10.0 and c["end_host_mono"] == 15.0
    assert c["label_source"] == "instructed_effort"
    for bad in (rec(plan, "t001", "b", protocol="aborted"), rec(plan, "t001", "c", quality="flagged"),
                rec(plan, "t001", "d", review="rejected"), rec(plan, "t001", "e", status="recording")):
        assert training_candidate(bad) is None
    unreviewed = rec(plan, "t001", "f", review="pending", status="finalized")
    assert training_candidate(unreviewed) is None
    assert training_candidate(unreviewed, require_accepted=False) is not None
    rest = rec(plan, "t001", "g")
    rest["training_candidate"]["phase"] = "REST"           # a REST interval is never returned
    assert training_candidate(rest) is None


def test_scan_reads_plans_and_attempts_and_reports_unreadable(tmp_path):
    proot = participant_root(tmp_path, "P01", simulated=False)
    plan = TrialPlan.create("P01", "real", seed=5)
    proot.mkdir(parents=True)
    atomic_write_text(proot / plan.filename, dumps(plan.to_dict()))
    folder = proot / "S01" / "20261015_120000_obj01_low_s0p0_t001_a01_abcd1234"
    folder.mkdir(parents=True)
    atomic_write_text(folder / "trial.json", dumps(rec(plan, "t001", "abcd1234")))
    broken = proot / "S01" / "20261015_120001_broken"
    broken.mkdir()
    (broken / "trial.json").write_text("{not json")
    res = scan_participant(proot)
    assert [p.plan_id for p in res["plans"]] == [plan.plan_id]
    assert list(res["records"]) == ["abcd1234"]
    assert len(res["unreadable"]) == 1
    assert participant_root(tmp_path, "P01", simulated=True) == tmp_path / "simulated" / "P01"


def test_object_catalog_keeps_stable_ids():
    objs = merge_objects([("obj03", "Mug"), ("obj06", "Ball (test)"), ("Bad ID", "x"),
                          ("empty_hand", "")])
    ids = [o for o, _ in objs]
    assert ids[:6] == [EMPTY_HAND, "obj01", "obj02", "obj03", "obj04", "obj05"]
    assert dict(objs)["obj03"] == "Mug" and dict(objs)[EMPTY_HAND] == "Empty hand"
    assert ids[6:] == ["obj06"]


def test_names_are_sanitized_and_traversal_rejected(tmp_path):
    assert s_tag(0.0) == "s0p0" and s_tag(0.5) == "s0p5" and s_tag(1.0) == "s1p0"
    assert attempt_name("obj03", "high", 1.0, "t042", 1, "3fa2b1c4") == \
        "obj03_high_s1p0_t042_a01_3fa2b1c4"
    assert attempt_name("empty_hand", "low", 0.0, "m007", 2, "abcd", "practice") == \
        "practice_empty_hand_low_s0p0_m007_a02_abcd"
    for bad in ("../x", "a/b", "", ".", "..", "x" * 40, "simulated", "SIMULATED", "a b"):
        assert not valid_id(bad), bad
    with pytest.raises(ValueError):
        attempt_name("../obj", "low", 0.0, "t001", 1, "abcd")
    with pytest.raises(ValueError):
        ensure_within(tmp_path / "root", tmp_path / "root" / ".." / "elsewhere")
    assert ensure_within(tmp_path, tmp_path / "P01" / "S01") == (tmp_path / "P01" / "S01").resolve()


def test_json_is_strict_nan_becomes_null():
    import numpy as np
    text = dumps({"imp": np.array([1.0, np.nan]), "x": float("inf"), "n": np.int64(3)})
    assert json.loads(text) == {"imp": [1.0, None], "x": None, "n": 3}
    assert jsonable({"s": {2, 1}}) == {"s": [1, 2]}
