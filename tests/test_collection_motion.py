"""Movement-varied trials: instructions, metadata, plan coverage, v1 compatibility."""

import collections
import hashlib
import json

import pytest

from conftest import wait_until
from gforce_emg_ui.collection.controller import REVIEW_REASONS, TrialSpec
from gforce_emg_ui.collection.plan import (DEFAULT_PLAN_COUNTS, PLAN_MOTIONS, TrialPlan,
                                           compute_progress, counts_toward_plan,
                                           generate_motion_order, max_target_run, motion_of,
                                           motion_split, participant_root, scan_participant,
                                           target_counts, training_candidate)
from gforce_emg_ui.collection.protocol import (HOLD, MOTION_INSTRUCTION_VERSION, MOTION_ORDER,
                                               MOTIONS, PHASES, PREPARE, REST, SETTLE,
                                               STATIONARY, TARGET_INSTRUCTIONS, TARGETS,
                                               ProtocolSettings, motion_info,
                                               participant_instructions, phase_instruction)
from gforce_emg_ui.collection.storage import atomic_write_text, dumps
from test_collection_plan import rec
from test_collection_workflow import (FAST, close_rig, drive, make_plan, make_rig, make_window,
                                      manual_spec, plan_spec, review, run_trial, sidecar, sim)

MOVING = [m for m in MOTION_ORDER if m != STATIONARY]


# -- instructions ---------------------------------------------------------------

def test_motion_ids_and_versioned_instruction():
    assert MOTION_ORDER == (STATIONARY, "wrist_flex_ext", "forearm_rotation", "reach_sweep")
    assert [MOTIONS[m]["display"] for m in MOTION_ORDER] == [
        "Stationary", "Wrist flexion/extension", "Forearm rotation", "Reach/sweep"]
    assert motion_info(STATIONARY) == {
        "motion_condition": STATIONARY, "display": "Stationary",
        "instruction_version": MOTION_INSTRUCTION_VERSION, "movement_instruction": None,
        "movement_phases": []}
    for m in MOVING:
        info = motion_info(m)
        assert info["instruction_version"] == "motion/1"
        assert info["movement_phases"] == [SETTLE, HOLD]
        assert "comfortable range" in info["movement_instruction"]


@pytest.mark.parametrize("target", TARGETS)
def test_stationary_instructions_are_unchanged(target):
    level = TARGETS[target]["display"]
    for phase in PHASES:
        assert phase_instruction(phase, target) == phase_instruction(phase, target, STATIONARY)
    assert phase_instruction(HOLD, target) == (f"Maintain {level} effort. Keep your hand and "
                                               "wrist position steady.")


@pytest.mark.parametrize("motion", MOVING)
@pytest.mark.parametrize("target", TARGETS)
def test_moving_trials_move_in_settle_and_hold_while_keeping_the_effort(motion, target):
    level, move = TARGETS[target]["display"], MOTIONS[motion]["movement"]
    text = participant_instructions(target, motion)
    assert "starts in SETTLE" in text[PREPARE]
    assert f"Smoothly establish {level} effort and begin the movement: {move}." in text[SETTLE]
    assert TARGET_INSTRUCTIONS[target] in text[SETTLE]
    hold = text[HOLD]
    assert hold.startswith(f"Maintain {level} effort while you continue the movement: {move}.")
    assert "same grasp" in hold
    assert "steady" not in hold and "still" not in hold           # HOLD != stay still
    assert text[REST].startswith("Stop the movement, release the effort")
    for t in text.values():
        assert "as hard as possible" not in t.lower()


def test_phase_durations_do_not_depend_on_motion():
    s = ProtocolSettings()
    assert (s.prepare_s, s.settle_s, s.hold_s, s.rest_s) == (5.0, 3.0, 5.0, 15.0)


# -- plan coverage ---------------------------------------------------------------

def test_motion_split_keeps_cell_totals_and_class_counts():
    split = motion_split(DEFAULT_PLAN_COUNTS)
    assert sum(split.values()) == 200
    for obj, counts in DEFAULT_PLAN_COUNTS.items():
        for i, t in enumerate(("low", "medium", "high")):
            assert sum(split[(obj, t, m)] for m in PLAN_MOTIONS) == counts[i]
    per_target = {t: sum(n for (_, tt, _), n in split.items() if tt == t)
                  for t in ("low", "medium", "high")}
    assert per_target == target_counts(DEFAULT_PLAN_COUNTS) == {"low": 67, "medium": 67,
                                                                "high": 66}


def test_every_motion_meets_every_gain_in_every_condition():
    split = motion_split(DEFAULT_PLAN_COUNTS)
    for obj in DEFAULT_PLAN_COUNTS:
        for t in ("low", "medium", "high"):
            for m in PLAN_MOTIONS:
                assert split[(obj, t, m)] >= 2, (obj, t, m)


def test_motions_are_balanced_per_condition_and_per_target():
    split = motion_split(DEFAULT_PLAN_COUNTS)
    for obj, counts in DEFAULT_PLAN_COUNTS.items():
        per_motion = [sum(split[(obj, t, m)] for t in ("low", "medium", "high"))
                      for m in PLAN_MOTIONS]
        assert per_motion == [sum(counts) // 4] * 4                # 10 (empty hand) / 8
    for t in ("low", "medium", "high"):
        totals = [sum(split[(o, t, m)] for o in DEFAULT_PLAN_COUNTS) for m in PLAN_MOTIONS]
        assert max(totals) - min(totals) <= 1, (t, totals)        # no motion/label confound
    assert sum(n for (_, _, m), n in split.items() if m == STATIONARY) == 50   # retained


@pytest.mark.parametrize("seed", range(20))
def test_new_plans_follow_the_split_and_the_run_rule(seed):
    plan = TrialPlan.create("P01", "real", seed=seed)
    assert plan.schema_version == 2 and plan.motions == PLAN_MOTIONS
    cells = collections.Counter((t["object_id"], t["target_label"], t["motion_condition"])
                                for t in plan.trials)
    assert cells == collections.Counter(motion_split(DEFAULT_PLAN_COUNTS))
    assert max_target_run([(t["object_id"], t["target_label"]) for t in plan.trials]) <= 2


def test_motion_order_is_reproducible_and_saved_with_the_plan(tmp_path):
    assert generate_motion_order(DEFAULT_PLAN_COUNTS, 5) == \
        generate_motion_order(DEFAULT_PLAN_COUNTS, 5)
    plan = TrialPlan.create("P01", "real", seed=5)
    d = plan.to_dict()
    assert d["schema_version"] == 2 and d["motion_conditions"] == list(PLAN_MOTIONS)
    assert d["motion_instruction_version"] == MOTION_INSTRUCTION_VERSION
    assert d["motion_counts"]["obj04"]["medium"] == {
        m: motion_split(DEFAULT_PLAN_COUNTS)[("obj04", "medium", m)] for m in PLAN_MOTIONS}
    path = tmp_path / plan.filename
    atomic_write_text(path, dumps(d))
    loaded = TrialPlan.from_dict(json.loads(path.read_text()))
    assert loaded.trials == plan.trials and loaded.motions == PLAN_MOTIONS


def test_bad_plan_files_are_rejected_not_guessed():
    d = TrialPlan.create("P01", "real", seed=1).to_dict()
    d["trials"][0]["motion_condition"] = "jumping"
    with pytest.raises(ValueError):
        TrialPlan.from_dict(d)
    d = TrialPlan.create("P01", "real", seed=1).to_dict()
    d["schema_version"] = 3
    with pytest.raises(ValueError):
        TrialPlan.from_dict(d)


def v1_plan_dict(seed=3):
    """A plan file as written before motion existed (schema_version 1, no motion)."""
    d = TrialPlan.create("P01", "simulation", seed=seed).to_dict()
    for k in ("motion_conditions", "motion_instruction_version", "motion_counts"):
        d.pop(k)
    d.update(schema_version=1, generator="weighted-random-order/no-target-run-over-2/v1")
    for t in d["trials"]:
        t.pop("motion_condition")
    return d


def test_v1_plan_loads_without_inventing_motion():
    plan = TrialPlan.from_dict(v1_plan_dict())
    assert plan.motions is None and plan.schema_version == 1
    assert all("motion_condition" not in t for t in plan.trials)
    assert plan.motion_of(plan.trials[0]) == STATIONARY              # what new attempts use
    assert compute_progress(plan, {}).by_motion == {}


def test_progress_requires_the_planned_motion_in_v2_but_not_in_v1():
    v2 = TrialPlan.create("P01", "real", seed=1)
    t1 = v2.trials[0]
    good = rec(v2, t1["trial_id"], "a")
    assert motion_of(good) == t1["motion_condition"] and counts_toward_plan(good, v2)
    wrong = rec(v2, t1["trial_id"], "b")
    wrong["condition"]["motion_condition"] = next(m for m in PLAN_MOTIONS
                                                  if m != t1["motion_condition"])
    unknown = rec(v2, t1["trial_id"], "c")
    unknown["condition"].pop("motion_condition")
    assert not counts_toward_plan(wrong, v2) and not counts_toward_plan(unknown, v2)
    p = compute_progress(v2, {"a": good})
    assert p.by_motion[t1["motion_condition"]] == [1, 50]
    assert sum(n for _, n in p.by_motion.values()) == 200

    v1 = TrialPlan.from_dict(v1_plan_dict())
    old = rec(v1, v1.trials[0]["trial_id"], "d")                    # recorded before motion
    assert motion_of(old) is None and counts_toward_plan(old, v1)
    cand = training_candidate(old)
    assert cand is not None and cand["motion_condition"] is None      # unknown preserved


# -- controller / files -----------------------------------------------------------

def test_moving_trial_metadata_and_quality(sim):
    r = sim
    folder = run_trial(r, manual_spec(r, motion_condition="forearm_rotation",
                                      target_label="low"))
    d = sidecar(folder)
    assert d["schema_version"] == 2
    assert d["condition"]["motion_condition"] == "forearm_rotation"
    assert d["motion"] == motion_info("forearm_rotation")
    hold = d["label"]["participant_instructions"]["HOLD"]
    assert "continue the movement" in hold and "palm-up" in hold
    assert d["label"]["target_label"] == "low"                      # one gain per trial
    assert d["training_candidate"]["motion_condition"] == "forearm_rotation"
    assert [p["planned_s"] for p in d["timing"]["phases"]] == [
        FAST["prepare_s"], FAST["settle_s"], FAST["hold_s"], FAST["rest_s"]]
    # intentional movement is never rejected automatically; no amplitude requirement
    assert d["training_candidate"]["quality_status"] == "clean" and r.c.acceptable
    review(r, "accept")
    assert sidecar(folder)["review"]["status"] == "accepted"


def test_plan_trial_uses_the_planned_motion(sim):
    r = sim
    make_plan(r, seed=4)
    spec = plan_spec(r)
    planned = r.c.planned_trial()["motion_condition"]
    assert spec.motion_condition == planned
    spec.motion_condition = next(m for m in MOTION_ORDER if m != planned)
    assert "Movement does not match the planned trial" in r.c.start_blockers(spec)
    folder = run_trial(r, plan_spec(r))
    assert sidecar(folder)["condition"]["motion_condition"] == planned


def test_new_review_reasons_are_saved(sim):
    r = sim
    for key in ("incorrect_movement", "grasp_change", "effort_not_maintained"):
        assert key in REVIEW_REASONS
    folder = run_trial(r, manual_spec(r, motion_condition="reach_sweep"))
    reasons = ["incorrect_movement", "grasp_change", "effort_not_maintained"]
    review(r, "repeat", reasons)
    assert sidecar(folder)["review"]["reasons"] == reasons
    run_trial(r, manual_spec(r, motion_condition="reach_sweep"))
    with pytest.raises(ValueError):
        r.c.submit_review("reject", ["moved_too_much"])


def test_invalid_motion_blocks_start(sim):
    spec = manual_spec(sim, motion_condition="jumping")
    assert any(b.startswith("Movement:") for b in sim.c.start_blockers(spec))
    assert TrialSpec().motion_condition == STATIONARY


def test_v1_plan_and_old_trials_are_never_rewritten(qapp, tmp_path):
    r = make_rig(qapp, tmp_path)
    try:
        proot = participant_root(r.root, "P01", simulated=True)
        proot.mkdir(parents=True)
        v1 = v1_plan_dict()
        plan_file = proot / f"plan_{v1['plan_id']}.json"
        atomic_write_text(plan_file, dumps(v1))
        # an attempt recorded before motion existed (trial.json schema 1, no motion)
        old = rec(TrialPlan.from_dict(v1), "t005", "old00001")
        old.update(schema_version=1, collection_session_id="S00")
        old_dir = proot / "S00" / "20261014_120000_old_t005_a01_old00001"
        old_dir.mkdir(parents=True)
        atomic_write_text(old_dir / "trial.json", dumps(old))
        digests = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (plan_file, old_dir / "trial.json")}

        r.c.open_participant(r.root, "P01")
        assert drive(r.c, lambda: not r.c.loading and r.c.plan is not None, r.app)
        assert r.c.plan.motions is None
        assert motion_of(r.c.records["old00001"]) is None             # still unknown
        assert r.c.progress.accepted == 1 and "t005" in r.c.progress.completed
        spec = plan_spec(r)
        assert spec.motion_condition == STATIONARY
        folder = run_trial(r, spec)
        d = sidecar(folder)
        assert d["condition"]["motion_condition"] == STATIONARY and d["plan_id"] == v1["plan_id"]
        review(r, "accept")
        assert r.c.progress.accepted == 2
        for p, digest in digests.items():                              # untouched on disk
            assert hashlib.sha256(p.read_bytes()).hexdigest() == digest
        assert scan_participant(proot)["records"]["old00001"]["condition"].get(
            "motion_condition") is None
    finally:
        close_rig(r)


# -- UI ---------------------------------------------------------------------------

def test_movement_selector_and_moving_instructions_in_the_panel(qapp, tmp_path):
    win, src, panel = make_window(qapp, tmp_path)
    try:
        combo = panel.motion_combo
        assert [combo.itemData(i) for i in range(combo.count())] == list(MOTION_ORDER)
        assert [combo.itemText(i) for i in range(combo.count())] == [
            MOTIONS[m]["display"] for m in MOTION_ORDER]
        win.record_tabs.setCurrentIndex(1)
        combo.setCurrentIndex(MOTION_ORDER.index("wrist_flex_ext"))
        assert panel.spec().motion_condition == "wrist_flex_ext"
        assert wait_until(lambda: panel.btn_start.isEnabled(), 5, qapp), panel.ready_label.text()
        assert combo.isEnabled()
        panel.btn_start.click()
        c = win.collection
        assert not combo.isEnabled()                                  # locked during a trial
        assert wait_until(lambda: c.machine.phase == HOLD, 5, qapp)
        qapp.processEvents()
        assert "continue the movement" in panel.instruction_label.text()
        assert "wrist flex/ext" in panel.target_label.text()
        assert wait_until(lambda: c.state == "review", 10, qapp)
        panel.btn_accept.click()
        assert wait_until(lambda: c.state == "idle", 5, qapp)
        panel.mode_combo.setCurrentIndex(0)                           # plan mode: planned motion
        panel._new_plan(confirm=False)
        assert wait_until(lambda: c.plan is not None, 5, qapp)
        qapp.processEvents()
        assert not combo.isEnabled()
        assert combo.currentData() == c.planned_trial()["motion_condition"]
        assert "Movement:" in panel.next_label.text()
    finally:
        src.disconnect_device()
        win.close()
