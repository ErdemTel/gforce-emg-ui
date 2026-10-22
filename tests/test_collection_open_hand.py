"""Open hand condition: instructions, the one-time +35 plan extension (either order
with surface press), validation of extended plans, opt-in selection, one-trial
edits, recording metadata and progress.

Plans and trial.json files live in tmp_path only (real data is never touched).
"""

import collections
import copy
import hashlib
import json
from pathlib import Path

import pytest
from PyQt5 import QtCore, QtWidgets

from conftest import wait_until
from gforce_emg_ui.collection import plan as plan_mod
from gforce_emg_ui.collection import storage as storage_mod
from gforce_emg_ui.collection.controller import (IDLE, MANUAL_OVERRIDE, REVIEW, REVIEW_REASONS,
                                                 RUNNING, TrialSpec)
from gforce_emg_ui.collection.plan import (DEFAULT_PLAN_COUNTS, EMPTY_HAND, EXTENDABLE,
                                           EXTENDED, INCOMPATIBLE, OPEN_HAND_EXTENSION,
                                           PLAN_BACKUP_DIR, SURFACE_PRESS_EXTENSION, TASKS,
                                           TrialPlan, compute_progress, counts_toward_plan,
                                           eligible_trials, extend_plan_file, extend_with_task,
                                           extension_problems, extension_status, max_target_run,
                                           merge_objects, pick_trial, plan_problems,
                                           resolve_request, scan_participant)
from gforce_emg_ui.collection.protocol import (ABORTED, GRASP, HOLD, MOTION_ORDER, MOTIONS,
                                               OPEN_HAND, OPEN_HAND_NAME, OPEN_HAND_POSTURE,
                                               PHASES, PREPARE, REST, SETTLE, STATIONARY,
                                               SURFACE_PHASE_TEXT, SURFACE_PRESS,
                                               SURFACE_TARGET_INSTRUCTIONS, TARGET_INSTRUCTIONS,
                                               TARGET_ORDER, TARGETS, operator_note,
                                               participant_instructions, phase_instruction,
                                               task_instructions)
from gforce_emg_ui.collection.storage import atomic_write_text, dumps
from gforce_emg_ui.recorder import HEADERS
from gforce_emg_ui.ui.collection_panel import MOTION_SHORT
from test_collection_plan import rec
from test_collection_selection import plan_window, shown, user_pick, want
from test_collection_workflow import (close_rig, drive, header, make_plan, make_rig, manual_spec,
                                      plan_spec, review, rows, run_trial, sidecar, start,
                                      start_panel_trial)

# Outputs of the code before Open hand existed (14d5fa5), plan_id / created fixed
BASE_SHA = {3: "7f3cbb5e47199752367474fdf746a2505ca78f53f06978f2f4ee499369d4a9ca",
            7: "4f9df478c7d211bd88a9d0a8e763a8d316550fa9990ba699e0a55f5f3bea515d",
            11: "408204d7f22bcf6d14b5b943b1e7b2aee71917d5dc364626f6e28dd26df04451"}
SURFACE_SHA = {3: "c99b3819705acd8fb3cc99f9f886f87068b4f469a6face02280b725f0ae12756",
               7: "d4d90016d257a890493a7ea5fc2c3c241548bd6d64751a411f952409acafa849",
               11: "64afced2ef2605dcaba90e0b1afe2932cd2ced367fc8bfbb75c4aea93e0e9346"}
SURFACE_TEXT_SHA = "99f8923b0031ccfec212db14b0bf848ab89c05c67b5829c8c5ec957d86907088"
GRASP_TASK_SHA = "91bd507d8695ad9fc43f2bc008481931b58744ce7deba6286908adf846605934"
FIXED = "2026-01-01T00:00:00"

PREPARE_TEXT = ("Keep your hand comfortably open, without making a fist or holding an object. "
                "Allow a natural slight finger curve; do not force your fingers straight or "
                "spread them widely. Choose a comfortable arm position and keep it consistent. "
                "Do not press your fingertips or palm into a surface.")
LEVEL_TEXT = {
    "low": "Keep your hand comfortably open and relax your forearm and hand as much as the "
           "required posture allows. Do not intentionally tense, squeeze, press, or spread "
           "your fingers.",
    "medium": "While keeping your hand open, establish a comfortable, repeatable moderate "
              "forearm contraction. Do not close your hand into a fist or press against "
              "anything.",
    "high": "While keeping your hand open, establish a firm, comfortable, repeatable forearm "
            "contraction. Keep the same open-hand posture. Do not use a maximal effort, force "
            "your joints straight, or press against anything.",
}
REST_TEXT = "Release the intentional contraction and rest your hand and forearm."
ABORT_TEXT = "Trial stopped. Stop any movement, release the contraction, and rest."
# Default allocation over the four movements (low / medium / high)
ALLOCATION = {STATIONARY: (4, 3, 2), "wrist_flex_ext": (4, 2, 3),
              "forearm_rotation": (4, 2, 3), "reach_sweep": (3, 3, 2)}


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def standard(seed=7, source="real"):
    plan = TrialPlan.create("P01", source, seed=seed)
    plan.plan_id, plan.created = "plan-fixed", FIXED
    return plan


def opened(plan, **kw):
    return extend_with_task(plan, OPEN_HAND, added=FIXED, **kw)


def pressed(plan, **kw):
    return extend_with_task(plan, SURFACE_PRESS, added=FIXED, **kw)


def of(plan, cond):
    return [t for t in plan.trials if t["object_id"] == cond]


def task_rec(plan, trial_id, attempt, **kw):
    """Synthetic accepted record; task-condition records carry their trial type."""
    r = rec(plan, trial_id, attempt, **kw)
    if plan.trial(trial_id)["object_id"] in TASKS:
        r["condition"]["trial_type"] = plan.trial(trial_id)["object_id"]
    return r


def allocation(trials):
    n = collections.Counter((t["motion_condition"], t["target_label"]) for t in trials)
    return {m: tuple(n[(m, t)] for t in TARGET_ORDER) for m in MOTION_ORDER}


def extend(r, cond, total):
    """Extend the rig's selected plan through the controller (real I/O path)."""
    assert r.c.extend_plan(cond), r.c.error
    assert drive(r.c, lambda: not r.c.busy, r.app)
    assert not r.c.error, r.c.error
    assert r.c.plan.total == total


def open_seed(pred=lambda t: True):
    """A plan seed whose first random pick with only Open hand ticked (after both
    extensions) is an open-hand trial matching pred."""
    for seed in range(1, 300):
        plan = opened(pressed(standard(seed=seed)))
        i = pick_trial(plan, compute_progress(plan, {}), [OPEN_HAND], key="0")
        if plan.trials[i]["object_id"] == OPEN_HAND and pred(plan.trials[i]):
            return seed
    pytest.fail("no seed found")


@pytest.fixture
def sim(qapp, tmp_path):
    r = make_rig(qapp, tmp_path)
    yield r
    close_rig(r)


# -- instructions ---------------------------------------------------------------

@pytest.mark.parametrize("target", TARGET_ORDER)
@pytest.mark.parametrize("motion", MOTION_ORDER)
def test_open_hand_instructions_for_every_level_and_movement(target, motion):
    text = participant_instructions(target, motion, OPEN_HAND)
    move = MOTIONS[motion]["movement"]
    assert text[PREPARE].startswith(PREPARE_TEXT)
    assert LEVEL_TEXT[target] in text[SETTLE]
    if target == "low":                         # relaxation, never "establish" a contraction
        assert "establish" not in text[SETTLE].replace(LEVEL_TEXT[target], "")
        assert text[HOLD].startswith("Stay relaxed")
    else:
        assert f"establish the {TARGETS[target]['display']} forearm contraction" in text[SETTLE]
        assert text[HOLD].startswith(f"Maintain the {TARGETS[target]['display']} contraction")
    if move:                                     # movement in SETTLE and HOLD, stop at REST
        assert move in text[SETTLE] and move in text[HOLD]
        assert "Movement this trial" in text[PREPARE]
        assert text[REST] == f"Stop the movement. {REST_TEXT}"
        assert "do not" in text[HOLD]
    else:
        assert "without compressing the armband" in text[PREPARE]
        assert "steady" in text[HOLD] and text[REST] == REST_TEXT
    assert text[ABORTED] == ABORT_TEXT
    for phase in PHASES + (ABORTED,):            # one source for display and metadata
        assert phase_instruction(phase, target, motion, OPEN_HAND) == text[phase]
    info = task_instructions(target, motion, OPEN_HAND)
    assert info["target_instruction"] == LEVEL_TEXT[target]
    assert info["participant_instructions"] == text
    assert info["operator_note"] == operator_note(target, OPEN_HAND)
    assert info["task_instruction_version"] == "open-hand/1"
    joined = " ".join(text.values())
    for leak in (list(TARGET_INSTRUCTIONS.values()) + list(SURFACE_TARGET_INSTRUCTIONS.values())
                 + list(SURFACE_PHASE_TEXT.values())
                 + ["grasp", "Grasp", "hand/object", "Set down", "desk", "pressing effort",
                    "fingertip contact", "as hard as possible"]):
        assert leak not in joined, leak


def test_moving_low_is_distinguished_from_stationary_relaxation():
    still = participant_instructions("low", STATIONARY, OPEN_HAND)
    for m in MOTION_ORDER[1:]:
        moving = participant_instructions("low", m, OPEN_HAND)
        assert moving[SETTLE] != still[SETTLE] and moving[HOLD] != still[HOLD]
        assert "no hand or grip tension" in moving[SETTLE]
        assert "add hand or grip tension" in moving[HOLD]
    assert "Gradually relax" in still[SETTLE]
    assert "Moving LOW" in operator_note("low", OPEN_HAND)


def test_operator_notes():
    low, med, high = (operator_note(t, OPEN_HAND) for t in TARGET_ORDER)
    assert "not zero EMG" in low and "perfectly flat" in low
    assert "fist" in med and "fist" in high
    assert "50%" in med and "not a maximal strength test" in high
    for note in (low, med, high):
        assert "does not verify co-contraction or physical stiffness" in note
    assert OPEN_HAND_NAME == "Open hand (no object)" and "relax" not in OPEN_HAND_NAME.lower()


def test_existing_task_texts_are_unchanged():
    grasp = {f"{t}/{m}": task_instructions(t, m) for t in TARGETS for m in MOTION_ORDER}
    assert h(json.dumps(grasp, sort_keys=True)) == GRASP_TASK_SHA
    surface = {t: task_instructions(t, STATIONARY, SURFACE_PRESS) for t in TARGETS}
    assert h(json.dumps(surface, sort_keys=True)) == SURFACE_TEXT_SHA


def test_spec_allows_open_hand_movements_and_enforces_its_posture(tmp_path):
    base = dict(participant_id="P01", session_id="S01", object_id=OPEN_HAND,
                posture_id=OPEN_HAND_POSTURE, target_label="high", dataset_role="practice",
                output_root=tmp_path)
    for m in MOTION_ORDER:
        spec = TrialSpec(**base, motion_condition=m)
        assert spec.problems() == [] and spec.trial_type == OPEN_HAND
    wrong = TrialSpec(**{**base, "posture_id": "fist"})
    assert "Open hand posture is open_hand" in wrong.problems()
    assert "Surface Press is stationary only" in TrialSpec(   # surface press is unchanged
        **{**base, "object_id": SURFACE_PRESS, "posture_id": "index_finger_press"},
        motion_condition="reach_sweep").problems()
    assert TrialSpec(**{**base, "object_id": EMPTY_HAND, "posture_id": "fist"}).trial_type == GRASP
    # an old objects.json can never turn the condition into an object; names are kept
    objs = dict(merge_objects([[OPEN_HAND, "Hand"], ["obj01", "Mug"], ["obj06", "Ball"]]))
    assert OPEN_HAND not in objs and objs["obj01"] == "Mug" and objs["obj06"] == "Ball"
    assert "incorrect_posture" in REVIEW_REASONS


# -- plan extension (pure) ---------------------------------------------------------

def test_standard_plan_is_extended_from_200_to_235():
    plan = standard()
    before = dumps(plan.to_dict())
    assert extension_status(plan, OPEN_HAND) == (EXTENDABLE, "")
    ext = opened(plan)
    assert dumps(plan.to_dict()) == before                # the original object is unchanged
    assert (ext.total, ext.plan_id, ext.seed, ext.created, ext.participant_id) == (
        235, plan.plan_id, plan.seed, plan.created, plan.participant_id)
    assert [dumps(t) for t in ext.trials[:200]] == [dumps(t) for t in plan.trials]
    new = ext.trials[200:]
    assert collections.Counter(t["target_label"] for t in new) == {"low": 15, "medium": 10,
                                                                   "high": 10}
    assert all((t["object_id"], t["posture_id"], t["trial_type"]) ==
               (OPEN_HAND, OPEN_HAND_POSTURE, OPEN_HAND) for t in new)
    assert allocation(new) == ALLOCATION                   # every movement x target cell
    assert [t["trial_id"] for t in new] == [f"t{i}" for i in range(201, 236)]
    assert [t["index"] for t in new] == list(range(201, 236))
    assert ext.counts == {**plan.counts, OPEN_HAND: (15, 10, 10)}
    assert extension_problems(plan, ext) == [] and plan_problems(ext) == []
    d = ext.to_dict()
    assert d["schema_version"] == 4 and d["total"] == 235
    assert d["counts_low_medium_high"][OPEN_HAND] == [15, 10, 10]
    assert d["motion_counts"][OPEN_HAND] == {
        t: {m: ALLOCATION[m][i] for m in MOTION_ORDER} for i, t in enumerate(TARGET_ORDER)}
    assert {k: v for k, v in d["motion_counts"].items() if k != OPEN_HAND} == \
        plan.to_dict()["motion_counts"]                    # old assignments not redistributed
    e = d["extensions"][0]
    assert (e["id"], e["condition"], e["trial_type"], e["posture_id"]) == (
        OPEN_HAND_EXTENSION, OPEN_HAND, OPEN_HAND, OPEN_HAND_POSTURE)
    assert e["task_instruction_version"] == "open-hand/1"
    assert e["counts_low_medium_high"] == [15, 10, 10] and isinstance(e["seed"], int)
    assert e["motion_conditions"] == list(MOTION_ORDER)
    assert e["motion_counts"] == d["motion_counts"][OPEN_HAND]
    assert (e["total_before"], e["total_after"], e["added"]) == (200, 235, FIXED)
    assert e["added_trial_ids"] == [t["trial_id"] for t in new] and e["generator"]
    assert TrialPlan.from_dict(json.loads(dumps(d))).to_dict() == json.loads(dumps(d))


def test_motion_counts_are_counted_from_the_trials():
    for plan in (standard(), pressed(standard()), opened(pressed(standard()))):
        d = plan.to_dict()
        n = collections.Counter((t["object_id"], t["target_label"], t["motion_condition"])
                                for t in plan.trials)
        assert d["motion_counts"] == {o: {t: {m: n[(o, t, m)] for m in MOTION_ORDER}
                                          for t in TARGET_ORDER} for o in plan.counts}
        assert sum(v for o in d["motion_counts"].values() for tt in o.values()
                   for v in tt.values()) == plan.total


@pytest.mark.parametrize("seed", sorted(BASE_SHA))
def test_new_and_surface_press_plans_are_byte_identical_to_before(seed):
    plan = standard(seed=seed)
    assert h(dumps(plan.to_dict())) == BASE_SHA[seed]
    assert h(dumps(pressed(plan).to_dict())) == SURFACE_SHA[seed]
    assert plan.schema_version == 2 and pressed(plan).schema_version == 3


def test_both_extension_orders_reach_265():
    plan = standard()
    a = opened(pressed(plan))                             # surface press first
    b = pressed(opened(plan))                             # open hand first
    for ext, first, second in ((a, SURFACE_PRESS, OPEN_HAND), (b, OPEN_HAND, SURFACE_PRESS)):
        assert ext.total == 265 and ext.schema_version == 4 and plan_problems(ext) == []
        assert [e["id"] for e in ext.extensions] == [TASKS[first].extension,
                                                     TASKS[second].extension]
        assert [dumps(t) for t in ext.trials[:200]] == [dumps(t) for t in plan.trials]
        n = TASKS[first].total
        assert {t["object_id"] for t in ext.trials[200:200 + n]} == {first}
        assert {t["object_id"] for t in ext.trials[200 + n:]} == {second}
        assert ext.counts == {**plan.counts, SURFACE_PRESS: (10, 10, 10),
                              OPEN_HAND: (15, 10, 10)}
        assert len({t["trial_id"] for t in ext.trials}) == len({t["index"] for t in ext.trials}) \
            == 265
        assert [t["trial_id"] for t in ext.trials[200:]] == [f"t{i}" for i in range(201, 266)]
        assert collections.Counter(t["target_label"] for t in of(ext, SURFACE_PRESS)) == \
            {"low": 10, "medium": 10, "high": 10}
        assert {t["motion_condition"] for t in of(ext, SURFACE_PRESS)} == {STATIONARY}
        assert allocation(of(ext, OPEN_HAND)) == ALLOCATION
        for cond in (SURFACE_PRESS, OPEN_HAND):           # each extension is idempotent
            status, why = extension_status(ext, cond)
            assert status == EXTENDED and "already has" in why and "265" in why
            with pytest.raises(ValueError, match="already has"):
                extend_with_task(ext, cond)
        d = json.loads(dumps(ext.to_dict()))
        assert TrialPlan.from_dict(d).to_dict() == d
    # the earlier extension record and trials are kept exactly by the second extension
    s = pressed(plan)
    assert [dumps(e) for e in a.extensions[:1]] == [dumps(e) for e in s.extensions]
    assert [dumps(t) for t in a.trials[:230]] == [dumps(t) for t in s.trials]
    o = opened(plan)
    assert [dumps(t) for t in b.trials[:235]] == [dumps(t) for t in o.trials]
    # same assignments whatever the order (seeded per extension); only the IDs differ
    strip = (lambda ts: [(t["target_label"], t["motion_condition"]) for t in ts])
    assert strip(of(a, OPEN_HAND)) == strip(of(b, OPEN_HAND))
    assert strip(of(a, SURFACE_PRESS)) == strip(of(b, SURFACE_PRESS))
    assert extension_problems(pressed(plan), a) == [] and extension_problems(o, b) == []


def test_open_hand_order_is_seeded_and_spread_out():
    plan = standard(seed=3)
    a, b = opened(plan), opened(plan)
    assert a.trials == b.trials                            # reproducible from the plan seed
    seed = a.extensions[0]["seed"]
    assert opened(plan, seed=seed).trials == a.trials      # saved seed
    assert opened(plan, seed=seed + 1).trials != a.trials
    for s in range(30):
        new = of(opened(plan, seed=s), OPEN_HAND)
        assert max_target_run((t["object_id"], t["target_label"]) for t in new) <= 2
        assert allocation(new) == ALLOCATION


def test_used_trial_ids_are_skipped():
    plan = TrialPlan.from_dict(json.loads(dumps(pressed(standard()).to_dict())))
    plan.trials[4] = {**plan.trials[4], "trial_id": "t231"}      # t231 is taken
    assert extension_status(plan, OPEN_HAND)[0] == EXTENDABLE
    ext = opened(plan)
    assert [t["trial_id"] for t in of(ext, OPEN_HAND)] == [f"t{i}" for i in range(232, 267)]
    assert [t["index"] for t in of(ext, OPEN_HAND)] == list(range(231, 266))
    assert extension_problems(plan, ext) == []


def test_legacy_plans_load_unchanged():
    v2, v3 = standard(), pressed(standard())
    v1 = standard()
    v1.motions = None
    v1.trials = [{k: v for k, v in t.items() if k != "motion_condition"} for t in v1.trials]
    for plan, version in ((v1, 1), (v2, 2), (v3, 3)):
        d = json.loads(dumps(plan.to_dict()))
        loaded = TrialPlan.from_dict(d)
        assert d["schema_version"] == version and loaded.schema_version == version
        assert dumps(loaded.to_dict()) == dumps(plan.to_dict())     # never rewritten
    status, why = extension_status(v1, OPEN_HAND)
    assert status == INCOMPATIBLE and "v1" in why


def broken(kind):
    """A plan with one specific defect (from a valid 265- or 235-trial plan)."""
    plan = TrialPlan.from_dict(json.loads(dumps(opened(pressed(standard())).to_dict())))
    oh = plan.extensions[1]
    if kind == "unknown":
        plan.extensions.append({"id": "wrist_band/v1"})
    elif kind == "duplicate_record":
        plan.extensions.append(copy.deepcopy(plan.extensions[0]))
    elif kind == "partial":
        del plan.trials[-5:]
    elif kind == "metadata":
        oh["posture_id"] = "fist"
    elif kind == "counts":
        plan.counts[OPEN_HAND] = (15, 10, 9)
    elif kind == "duplicate_ids":
        plan.trials[-1]["trial_id"] = plan.trials[-2]["trial_id"]
        oh["added_trial_ids"] = [t["trial_id"] for t in plan.trials[230:]]
    elif kind == "allocation":
        t = next(t for t in plan.trials[230:] if t["motion_condition"] == STATIONARY)
        t["motion_condition"] = "reach_sweep"
    elif kind == "unrecorded":
        plan = standard()
        extra = of(opened(plan), OPEN_HAND)[:5]
        plan.trials += copy.deepcopy(extra)
    elif kind == "base":
        t = plan.trials[0]
        t["target_label"] = next(x for x in TARGET_ORDER if x != t["target_label"])
    return plan


@pytest.mark.parametrize("kind, why", [
    ("unknown", "unknown extension"), ("duplicate_record", "duplicate extension records"),
    ("partial", "partial addition"), ("metadata", "conflicting extension metadata"),
    ("counts", "condition counts do not match"), ("duplicate_ids", "duplicate trial IDs"),
    ("allocation", "not the planned allocation"), ("unrecorded", "5 trials are not covered"),
    ("base", "first 200 trials")])
def test_invalid_or_partial_plans_are_refused_not_repaired(kind, why):
    plan = broken(kind)
    before = dumps(plan.to_dict()) if kind != "metadata" else None
    for cond in (OPEN_HAND, SURFACE_PRESS):
        status, text = extension_status(plan, cond)
        assert status == INCOMPATIBLE and why in text and "Nothing was changed" in text, text
        with pytest.raises(ValueError):
            extend_with_task(plan, cond)
    if before is not None:
        assert dumps(plan.to_dict()) == before


@pytest.mark.parametrize("kind, why", [
    ("v4_without_open_hand", "without open-hand"), ("v3_with_open_hand", "schema_version 4"),
    ("v2_with_open_hand", "schema_version 4"), ("posture", "wrong trial type, posture"),
    ("moving_press", "must be stationary"), ("unknown_extension", "unknown plan extension"),
    ("future", "unsupported plan schema_version 5")])
def test_inconsistent_plan_files_are_not_loaded(tmp_path, kind, why):
    d = json.loads(dumps(opened(pressed(standard())).to_dict()))
    if kind == "v4_without_open_hand":
        d = json.loads(dumps(pressed(standard()).to_dict()))
        d["schema_version"] = 4
    elif kind == "v3_with_open_hand":
        d["schema_version"] = 3
    elif kind == "v2_with_open_hand":
        d["schema_version"], d["extensions"] = 2, []
    elif kind == "posture":
        next(t for t in d["trials"] if t["object_id"] == OPEN_HAND)["posture_id"] = "fist"
    elif kind == "moving_press":
        next(t for t in d["trials"] if t["object_id"] == SURFACE_PRESS)["motion_condition"] = \
            "reach_sweep"
    elif kind == "unknown_extension":
        d["extensions"].append({"id": "x/v9"})
    else:
        d["schema_version"] = 5
    with pytest.raises(ValueError, match=why):
        TrialPlan.from_dict(d)
    proot = tmp_path / "P01"                               # listed as unreadable, never guessed
    proot.mkdir()
    (proot / "plan_plan-fixed.json").write_text(dumps(d))
    scan = scan_participant(proot)
    assert scan["plans"] == [] and len(scan["unreadable"]) == 1


# -- plan extension (file) ---------------------------------------------------------

def write(tmp_path, plan):
    path = tmp_path / "P01" / plan.filename
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, dumps(plan.to_dict()))
    return path


def test_plan_file_230_to_265_with_its_own_verified_backup(tmp_path):
    plan = standard()
    path = write(tmp_path, plan)
    raw200 = path.read_bytes()
    p230 = extend_plan_file(path, plan)                    # surface press (default)
    raw230 = path.read_bytes()
    surface_backup = path.parent / PLAN_BACKUP_DIR / f"{path.stem}.before-surface-press.json"
    p265 = extend_plan_file(path, p230, condition=OPEN_HAND)
    backup = path.parent / PLAN_BACKUP_DIR / f"{path.stem}.before-open-hand.json"
    assert backup.read_bytes() == raw230 and surface_backup.read_bytes() == raw200
    assert json.loads(path.read_text()) == json.loads(dumps(p265.to_dict()))
    e = p265.extensions[-1]
    assert e["id"] == OPEN_HAND_EXTENSION and (e["total_before"], e["total_after"]) == (230, 265)
    assert e["original_sha256"] == hashlib.sha256(raw230).hexdigest()
    assert e["backup"] == f"{PLAN_BACKUP_DIR}/{backup.name}"
    assert p265.extensions[0] == p230.extensions[0]        # surface record unchanged
    scan = scan_participant(path.parent)                    # backups are never plans or trials
    assert [p.plan_id for p in scan["plans"]] == [plan.plan_id] and not scan["unreadable"]
    assert scan["records"] == {} and not list(path.parent.rglob("trial.json"))
    loaded = scan["plans"][0]
    assert loaded.total == 265 and loaded.trials[:230] == p230.trials
    assert not list(path.parent.glob("**/*.tmp"))
    text = path.read_text()
    for cond in (OPEN_HAND, SURFACE_PRESS):                # applying either again: refused
        with pytest.raises(ValueError, match="already has"):
            extend_plan_file(path, loaded, condition=cond)
    with pytest.raises(ValueError, match="differs"):
        extend_plan_file(path, p230, condition=OPEN_HAND)  # stale in-memory plan
    assert path.read_text() == text


def test_existing_backup_with_other_content_is_kept(tmp_path):
    plan = standard()
    path = write(tmp_path, plan)
    other = path.parent / PLAN_BACKUP_DIR / f"{path.stem}.before-open-hand.json"
    other.parent.mkdir()
    other.write_text("something else\n")
    ext = extend_plan_file(path, plan, condition=OPEN_HAND)
    assert other.read_text() == "something else\n"
    name = ext.extensions[0]["backup"]
    assert name.startswith(f"{PLAN_BACKUP_DIR}/{path.stem}.before-open-hand.") and \
        name != f"{PLAN_BACKUP_DIR}/{other.name}"
    assert (path.parent / ext.extensions[0]["backup"]).read_text() == dumps(plan.to_dict())


@pytest.mark.parametrize("fail", ["write", "replace", "validate", "changed_on_disk"])
def test_failed_open_hand_extension_leaves_the_plan_usable(tmp_path, monkeypatch, fail):
    plan = pressed(standard())
    path = write(tmp_path, plan)
    raw = path.read_bytes()
    if fail == "write":
        real = plan_mod.atomic_write_text

        def broken_write(p, text):
            if Path(p) == path:
                raise OSError("No space left on device")
            return real(p, text)
        monkeypatch.setattr(plan_mod, "atomic_write_text", broken_write)
    elif fail == "replace":
        real_replace = storage_mod.os.replace

        def broken_replace(src, dst):
            if Path(dst) == path:
                raise OSError("rename failed")
            return real_replace(src, dst)
        monkeypatch.setattr(storage_mod.os, "replace", broken_replace)
    elif fail == "validate":
        monkeypatch.setattr(plan_mod, "extension_problems", lambda a, b: ["synthetic problem"])
    else:
        d = json.loads(raw)
        d["trials"][0]["target_label"] = "low" if d["trials"][0]["target_label"] != "low" \
            else "high"
        raw = dumps(d).encode()
        path.write_bytes(raw)
    with pytest.raises((OSError, ValueError)):
        extend_plan_file(path, plan, condition=OPEN_HAND)
    assert path.read_bytes() == raw                         # original untouched
    assert TrialPlan.from_dict(json.loads(raw)).total == 230
    assert not list(path.parent.glob("**/.*.tmp"))
    if fail in ("validate", "changed_on_disk"):
        assert not (path.parent / PLAN_BACKUP_DIR).exists()  # refused before any write


# -- progress and selection (no Qt) --------------------------------------------------

def test_progress_counts_accepted_open_hand_trials_once():
    plan = opened(pressed(standard(source="simulation")), seed=5)
    lows = [t for t in of(plan, OPEN_HAND) if t["target_label"] == "low"]
    records = {r["attempt_id"]: r for r in [
        task_rec(plan, lows[0]["trial_id"], "a", t=1),
        task_rec(plan, lows[0]["trial_id"], "b", t=2),                      # duplicate accept
        task_rec(plan, lows[1]["trial_id"], "c", review="rejected", t=3),
        task_rec(plan, lows[2]["trial_id"], "d", protocol="aborted", t=4),
        task_rec(plan, lows[3]["trial_id"], "e", review="pending", status="finalized", t=5),
        task_rec(plan, lows[4]["trial_id"], "f", quality="flagged", t=6),
        task_rec(plan, lows[5]["trial_id"], "g", role="practice", t=7),     # not development
        task_rec(plan, of(plan, SURFACE_PRESS)[0]["trial_id"], "h", t=8),
        task_rec(plan, plan.trials[0]["trial_id"], "i", t=9),               # a grasp trial
    ]}
    p = compute_progress(plan, records)
    assert p.by_cell[(OPEN_HAND, "low")] == [1, 15]
    assert p.by_cell[(OPEN_HAND, "medium")] == p.by_cell[(OPEN_HAND, "high")] == [0, 10]
    assert sum(p.by_cell[(SURFACE_PRESS, t)][0] for t in TARGET_ORDER) == 1
    assert (p.total_planned, p.accepted, p.remaining) == (265, 3, 262)
    assert sum(n for _, n in p.by_motion.values()) == 200      # movement split = grasp only
    bare = rec(plan, lows[6]["trial_id"], "x")                  # no open-hand trial type
    assert not counts_toward_plan(bare, plan)
    assert counts_toward_plan(task_rec(plan, lows[6]["trial_id"], "y"), plan)


def test_open_hand_is_opt_in_and_empty_hand_keeps_its_eligibility():
    plan = opened(pressed(standard()), seed=5)
    p = compute_progress(plan, {})
    objs = [o for o in DEFAULT_PLAN_COUNTS if o != EMPTY_HAND]
    assert len(eligible_trials(plan, p, objs)) == 200
    assert len(eligible_trials(plan, p, objs + [OPEN_HAND])) == 235
    assert len(eligible_trials(plan, p, objs + [OPEN_HAND, SURFACE_PRESS])) == 265
    assert len(eligible_trials(plan, p, [])) == 40                   # empty hand only
    assert all(plan.trials[pick_trial(plan, p, objs, key=str(k))]["object_id"] not in TASKS
               for k in range(200))
    picks = collections.Counter(plan.trials[pick_trial(plan, p, [OPEN_HAND], key=str(k))]
                                ["object_id"] for k in range(300))
    assert set(picks) == {OPEN_HAND, EMPTY_HAND}
    # every movement x target cell can be requested; surface press stays stationary only
    for m in MOTION_ORDER:
        for t in TARGET_ORDER:
            assert resolve_request(plan, p, want(OPEN_HAND, t, m))[0] is not None
    assert "stationary only" in resolve_request(plan, p, want(SURFACE_PRESS, "low",
                                                              "reach_sweep"))[2]
    plain = standard()
    _, _, why = resolve_request(plain, compute_progress(plain, {}),
                                want(OPEN_HAND, "low", STATIONARY))
    assert "no Open hand trials" in why


# -- controller ------------------------------------------------------------------

def test_controller_adds_both_extensions_and_keeps_progress(qapp, tmp_path):
    r = make_rig(qapp, tmp_path)
    try:
        make_plan(r)
        c = r.c
        plan_id, original = c.plan.plan_id, [dumps(t) for t in c.plan.trials]
        kept = run_trial(r, plan_spec(r))
        review(r, "accept")
        tid = c.spec.trial_id
        files = {p: sha(p) for p in kept.iterdir()}
        extend(r, SURFACE_PRESS, 230)
        surface_record = dumps(c.plan.extensions[0])
        assert c.extend_plan(OPEN_HAND)
        assert c.extending and "Plan update in progress" in c.start_blockers(plan_spec(r))
        assert c.create_plan(5) is None                              # no conflicting writes
        assert not c.extend_plan(OPEN_HAND) and not c.extend_plan(SURFACE_PRESS)
        assert not c.open_participant(r.root, "P02")                 # no context switch
        c.select_plan(None)
        assert c.plan.plan_id == plan_id and c.participant_id == "P01"
        assert drive(c, lambda: not c.busy, r.app) and not c.error, c.error
        assert not c.extending
        assert c.plan.total == 265 and c.plan.plan_id == plan_id
        assert [dumps(t) for t in c.plan.trials[:200]] == original
        assert dumps(c.plan.extensions[0]) == surface_record
        assert set(c.progress.completed) == {tid} and c.progress.accepted == 1
        assert "230 → 265" in c.message and "35 Open hand" in c.message
        assert PLAN_BACKUP_DIR in c.message
        assert [p.plan_id for p in c.plans] == [plan_id]
        assert {p: sha(p) for p in kept.iterdir()} == files          # recordings untouched
        path = c.participant_root() / c.plan.filename
        text = path.read_text()
        assert not c.extend_plan(OPEN_HAND) and "already has" in c.error     # idempotent
        assert path.read_text() == text and c.plan.total == 265
        assert OPEN_HAND not in c.available()                        # opt-in
    finally:
        close_rig(r)
    r = make_rig(qapp, tmp_path)                                     # restart: same from disk
    try:
        assert r.c.plan.plan_id == plan_id and r.c.plan.total == 265
        assert r.c.progress.accepted == 1 and tid in r.c.progress.completed
        assert r.c.extension_status(OPEN_HAND)[0] == EXTENDED
        assert r.c.extension_status(SURFACE_PRESS)[0] == EXTENDED
    finally:
        close_rig(r)


def test_opening_a_participant_never_extends_its_plan(qapp, tmp_path):
    r = make_rig(qapp, tmp_path)
    try:
        make_plan(r)
        path = r.c.participant_root() / r.c.plan.filename
        raw = path.read_bytes()
    finally:
        close_rig(r)
    r = make_rig(qapp, tmp_path)
    try:
        assert r.c.plan.total == 200 and r.c.extension_status(OPEN_HAND)[0] == EXTENDABLE
        assert path.read_bytes() == raw and not (path.parent / PLAN_BACKUP_DIR).exists()
    finally:
        close_rig(r)


def test_failed_open_hand_extension_in_the_controller_keeps_the_plan(sim, monkeypatch):
    r = sim
    make_plan(r)
    c = r.c
    path = c.participant_root() / c.plan.filename
    raw = path.read_bytes()
    monkeypatch.setattr(plan_mod, "atomic_write_text",
                        lambda p, t: (_ for _ in ()).throw(OSError("disk full")))
    assert c.extend_plan(OPEN_HAND)
    assert drive(c, lambda: not c.busy, r.app)
    assert "NOT extended" in c.error and "disk full" in c.error
    assert c.plan.total == 200 and path.read_bytes() == raw and OPEN_HAND not in c.plan.counts
    assert "Plan update in progress" not in c.start_blockers(plan_spec(r))


def test_one_trial_edits_with_open_hand(sim):
    r = sim
    make_plan(r)
    extend(r, SURFACE_PRESS, 230)
    extend(r, OPEN_HAND, 265)
    c = r.c
    auto = c.proposed_trial()
    assert auto["object_id"] not in TASKS
    if auto["object_id"] != EMPTY_HAND:
        assert c.edit_selection("object_id", EMPTY_HAND)
    before = dict(c.pending.requested)
    assert c.edit_selection("object_id", OPEN_HAND)            # empty hand -> open hand
    assert c.pending.requested == {**before, "object_id": OPEN_HAND}
    t = c.planned_trial()
    assert (t["object_id"], t["target_label"], t["motion_condition"], t["posture_id"]) == (
        OPEN_HAND, before["target_label"], before["motion_condition"], OPEN_HAND_POSTURE)
    sel = c.selection_record()
    assert sel["mode"] == MANUAL_OVERRIDE and sel["enforced_fields"] == []
    assert sel["availability_bypass"] is True                  # open hand was unticked
    assert sel["automatic_proposal"]["trial_id"] == auto["trial_id"]
    if before["motion_condition"] == STATIONARY:               # open hand keeps a movement
        assert c.edit_selection("motion_condition", "forearm_rotation")
    motion = c.pending.requested["motion_condition"]
    assert motion != STATIONARY and c.planned_trial()["motion_condition"] == motion
    tgt = next(x for x in TARGET_ORDER if x != c.pending.requested["target_label"])
    assert c.edit_selection("target_label", tgt)               # target only: condition kept
    t = c.planned_trial()
    assert (t["object_id"], t["target_label"], t["motion_condition"]) == (OPEN_HAND, tgt, motion)
    assert c.edit_selection("object_id", SURFACE_PRESS)        # surface press: stationary
    assert c.pending.requested["motion_condition"] == STATIONARY
    assert c.pending.enforced == {"motion_condition": motion}
    assert c.selection_record()["enforced_fields"] == ["motion_condition"]
    assert c.planned_trial()["object_id"] == SURFACE_PRESS
    assert c.edit_selection("object_id", OPEN_HAND)            # back: the movement returns
    assert c.pending.requested["motion_condition"] == motion and c.pending.enforced == {}
    assert c.planned_trial()["motion_condition"] == motion
    c.set_available_objects([o for o in c.plan.counts if o not in TASKS] + [OPEN_HAND])
    assert c.selection_record()["availability_bypass"] is False
    assert c.clear_edits() and c.planned_trial()["trial_id"] == auto["trial_id"]
    assert c.pending.edited == [] and c.pending.enforced == {}


def test_completed_open_hand_cell_gives_a_clear_blocker(sim):
    r = sim
    make_plan(r)
    extend(r, OPEN_HAND, 235)
    c = r.c
    cell = [t for t in of(c.plan, OPEN_HAND)
            if (t["target_label"], t["motion_condition"]) == ("low", STATIONARY)]
    assert len(cell) == 4
    for k, t in enumerate(cell):                               # as if recorded and accepted
        rr = task_rec(c.plan, t["trial_id"], f"syn{k}", t=1000.0 + k, source="simulation")
        c.records[rr["attempt_id"]] = rr
    c.select_plan(c.plan.plan_id)
    assert c.progress.by_cell[(OPEN_HAND, "low")] == [4, 15]
    for field, value in (("object_id", OPEN_HAND), ("target_label", "low"),
                         ("motion_condition", STATIONARY)):
        assert c.edit_selection(field, value)
    assert c.planned_trial() is None
    assert "No remaining planned trials for open_hand / low / stationary" in c.pending.problem
    assert c.pending.requested == want(OPEN_HAND, "low", STATIONARY)     # nothing substituted
    assert c.edit_selection("motion_condition", "wrist_flex_ext") and c.planned_trial()


def test_open_hand_trial_is_recorded_reviewed_and_counted(sim):
    r = sim
    make_plan(r, seed=open_seed(lambda t: t["motion_condition"] != STATIONARY))
    extend(r, SURFACE_PRESS, 230)
    extend(r, OPEN_HAND, 265)
    c = r.c
    c.set_available_objects([OPEN_HAND])
    auto = c.proposed_trial()
    assert auto["object_id"] == OPEN_HAND and c.pending.mode == "automatic"
    tgt, motion = auto["target_label"], auto["motion_condition"]
    spec = plan_spec(r)
    spec.posture_description = "Comfortably open hand, no object"
    start(r, spec)
    spec.target_label = "high" if tgt != "high" else "low"     # the attempt froze its copy
    assert c.spec.target_label == tgt and c.spec.trial_type == OPEN_HAND
    assert not c.edit_selection("target_label", "medium")      # no edits while running
    assert not c.set_available_objects([])
    folder = c.session.folder
    assert drive(c, lambda: c.state == REVIEW, r.app), (c.state, c.error)
    assert folder.parent.parent.parent.name == "simulated"     # simulation kept apart
    assert f"_{OPEN_HAND}_{tgt}_" in folder.name and f"_{auto['trial_id']}_a01_" in folder.name
    assert {p.name for p in folder.iterdir()} == {"emg.csv", "imu.csv", "contact.csv",
                                                 "markers.csv", "session.json", "trial.json"}
    for name in ("emg", "imu", "contact", "markers"):         # raw layout unchanged
        assert header(folder / f"{name}.csv") == HEADERS[name]
    assert [m["label"] for m in rows(folder / "markers.csv")][:4] == [
        "PREPARE_START", "SETTLE_START", "HOLD_START", "HOLD_END"]
    d = sidecar(folder)
    assert (d["schema_version"], d["protocol_version"]) == (2, "stiffness-phase0/1.3")
    assert (d["data_source"], d["is_simulation"]) == ("simulation", True)
    cond = d["condition"]
    assert (cond["object_id"], cond["trial_type"], cond["posture_id"], cond["motion_condition"]) \
        == (OPEN_HAND, OPEN_HAND, OPEN_HAND_POSTURE, motion)
    assert cond["object_name"] == OPEN_HAND_NAME
    assert cond["posture_description"] == "Comfortably open hand, no object"
    lab = d["label"]
    assert (lab["target_label"], lab["target_class_id"], lab["target_s"]) == (
        tgt, TARGETS[tgt]["class_id"], TARGETS[tgt]["s"])
    assert lab["mapping"] == {"low": {"target_class_id": 0, "target_s": 0.0},
                              "medium": {"target_class_id": 1, "target_s": 0.5},
                              "high": {"target_class_id": 2, "target_s": 1.0}}
    assert lab["label_source"] == "instructed_effort"
    assert {k: lab[k] for k in task_instructions(tgt, motion, OPEN_HAND)} == \
        task_instructions(tgt, motion, OPEN_HAND)
    assert lab["task_instruction_version"] == "open-hand/1"
    assert lab["target_instruction"] == LEVEL_TEXT[tgt]
    assert d["motion"]["motion_condition"] == motion
    assert d["training_candidate"]["phase"] == HOLD
    assert d["training_candidate"]["motion_condition"] == motion
    assert d["selection"]["mode"] == "automatic"
    assert OPEN_HAND in d["selection"]["available_objects"]
    for key in ("force", "stiffness", "onset"):                # nothing measured is claimed
        assert key not in d and key not in lab
    review(r, "repeat", ["incorrect_posture"])                # keeps the planned trial
    assert c.sidecar["review"]["reasons"] == ["incorrect_posture"]
    assert not folder.exists() and c.planned_trial()["trial_id"] == auto["trial_id"]
    folder = run_trial(r, plan_spec(r))
    assert "_a01_" in folder.name
    review(r, "accept")
    assert c.progress.by_cell[(OPEN_HAND, tgt)] == [1, 15 if tgt == "low" else 10]
    assert c.progress.completed[auto["trial_id"]] == sidecar(folder)["attempt_id"]
    assert sum(n for _, n in c.progress.by_motion.values()) == 200
    reader = pytest.importorskip("emg_classifier.trials")
    tr = reader.read_trial(folder)
    assert tr.problems == {}, tr.problems
    assert (tr.info["object_id"], tr.info["posture_id"], tr.info["motion_condition"]) == (
        OPEN_HAND, OPEN_HAND_POSTURE, motion)


def test_manual_open_hand_practice_never_counts(sim):
    r = sim
    make_plan(r)
    extend(r, OPEN_HAND, 235)
    c = r.c
    spec = manual_spec(r, object_id=OPEN_HAND, object_name=OPEN_HAND_NAME,
                       posture_id=OPEN_HAND_POSTURE, target_label="low",
                       motion_condition="reach_sweep")
    folder = run_trial(r, spec)
    d = sidecar(folder)
    assert (d["condition"]["trial_type"], d["dataset_role"], d["plan_id"]) == (
        OPEN_HAND, "practice", None)
    review(r, "accept")
    assert c.progress.accepted == 0 and c.progress.by_cell[(OPEN_HAND, "low")] == [0, 15]


# -- panel (headless Qt) ------------------------------------------------------------

def test_panel_extension_dialogs_allocation_and_progress(qapp, tmp_path, monkeypatch):
    win, src, panel, c = plan_window(qapp, tmp_path)
    asked, answer = [], [QtWidgets.QMessageBox.No]
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        lambda *a: asked.append(a[2]) or answer[0])
    infos = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a: infos.append(a[2]))
    try:
        path = c.participant_root() / c.plan.filename
        raw = path.read_bytes()
        assert panel.btn_extend_open.text() == "Add 35 Open-Hand Trials"
        assert not panel.open_check.isEnabled() and not panel.open_check.isChecked()
        panel.btn_extend_open.click()                            # No: nothing changes
        wait_until(lambda: False, 0.2, qapp)
        assert len(asked) == 1 and path.read_bytes() == raw and c.plan.total == 200
        text = asked[0]
        assert c.plan.plan_id in text and "200 → 235" in text
        assert "15 LOW, 10 MEDIUM, 10 HIGH" in text
        for m, (lo, me, hi) in ALLOCATION.items():               # motion-by-target allocation
            assert f"{MOTION_SHORT[m]}: LOW {lo} · MEDIUM {me} · HIGH {hi}" in text
        assert "stationary: LOW 4 · MEDIUM 3 · HIGH 2" in text
        assert "Stationary relaxed LOW: 4 trials" in text
        answer[0] = QtWidgets.QMessageBox.Yes
        panel.btn_extend_open.click()                            # open hand first
        assert wait_until(lambda: c.plan.total == 235 and not c.busy, 5, qapp), c.error
        wait_until(lambda: False, 0.2, qapp)
        progress = panel.progress_label.text()
        assert "/ 235" in progress and "Surface Press:" not in progress
        assert "Open hand: LOW 0/15 · MEDIUM 0/10 · HIGH 0/10 · total 0/35" in progress
        assert panel.open_check.isEnabled() and not panel.open_check.isChecked()
        assert not panel.surface_check.isEnabled()
        panel.btn_extend_plan.click()                            # then surface press
        assert "235 → 265" in asked[-1]
        assert wait_until(lambda: c.plan.total == 265 and not c.busy, 5, qapp), c.error
        wait_until(lambda: False, 0.2, qapp)
        progress = panel.progress_label.text()
        assert "/ 265" in progress
        assert "Surface Press: LOW 0/10 · MEDIUM 0/10 · HIGH 0/10 · total 0/30" in progress
        assert "Open hand: LOW 0/15 · MEDIUM 0/10 · HIGH 0/10 · total 0/35" in progress
        assert panel.surface_check.isEnabled() and panel.open_check.isEnabled()
        raw = path.read_bytes()
        n = len(asked)
        panel.btn_extend_open.click()                            # again: message only
        assert len(asked) == n and "already has" in infos[-1] and path.read_bytes() == raw
        panel.btn_advanced.setChecked(True)
        wait_until(lambda: False, 0.1, qapp)
        assert panel.minimumSizeHint().width() <= panel.width()
    finally:
        src.disconnect_device()
        win.close()


def test_panel_open_hand_trial_without_relayouts(qapp, tmp_path, monkeypatch):
    win, src, panel, c = plan_window(qapp, tmp_path)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a: QtWidgets.QMessageBox.Yes)
    side = win.centralWidget().widget(0).widget()
    layouts = []

    class Spy(QtCore.QObject):
        def eventFilter(self, obj, ev):
            if obj is side and ev.type() == QtCore.QEvent.LayoutRequest:
                layouts.append(1)
            return False
    spy = Spy()
    try:
        assert panel._extend_plan(OPEN_HAND)
        assert wait_until(lambda: c.plan.total == 235 and not c.busy, 5, qapp)
        panel.open_check.setChecked(True)
        assert OPEN_HAND in c.available()
        panel.open_check.setChecked(False)
        assert OPEN_HAND not in c.available()
        if shown(panel)["motion_condition"] == STATIONARY:      # a moving open-hand trial
            user_pick(qapp, panel.motion_combo, "wrist_flex_ext")
        before = shown(panel)
        user_pick(qapp, panel.object_combo, OPEN_HAND)          # target and movement kept
        wait_until(lambda: False, 0.2, qapp)
        assert shown(panel) == {**before, "object_id": OPEN_HAND}
        assert panel.posture_combo.currentText() == OPEN_HAND_POSTURE
        assert "stationary only" not in panel.sel_note.text()
        assert OPEN_HAND_NAME in panel.next_label.text()
        if shown(panel)["target_label"] != "medium":            # widest "Next:" line
            user_pick(qapp, panel.target_combo, "medium")
        wait_until(lambda: False, 0.1, qapp)
        assert shown(panel)["object_id"] == OPEN_HAND and "MEDIUM" in panel.next_label.text()
        assert panel.minimumSizeHint().width() <= panel.width()
        t = c.planned_trial()
        tgt, motion = t["target_label"], t["motion_condition"]
        assert motion != STATIONARY and t["object_id"] == OPEN_HAND
        assert operator_note(tgt, OPEN_HAND) in panel.instruction_label.text()
        text = participant_instructions(tgt, motion, OPEN_HAND)
        for key, value in dict(prepare_s=0.5, settle_s=0.5, hold_s=2.0).items():
            panel.dur_spins[key].setValue(value)
        qapp.installEventFilter(spy)
        start_panel_trial(qapp, win, panel)
        assert wait_until(lambda: c.machine.phase == PREPARE, 2, qapp)
        assert panel.instruction_label.text() == text[PREPARE]
        assert wait_until(lambda: c.machine.phase == HOLD, 5, qapp)
        wait_until(lambda: False, 0.1, qapp)
        layouts.clear()
        wait_until(lambda: False, 1.0, qapp)
        assert len(layouts) <= 3, len(layouts)                # no per-tick re-layout
        assert panel.instruction_label.text() == text[HOLD]
        assert f"Target: {TARGETS[tgt]['display']}" in panel.target_label.text()
        assert wait_until(lambda: c.state == REVIEW, 30, qapp)
        wait_until(lambda: False, 0.1, qapp)
        assert sidecar(c.session.folder)["label"]["participant_instructions"] == text
        assert panel.btn_accept.isEnabled(), c.accept_blockers
        panel.btn_accept.click()
        assert wait_until(lambda: c.state == IDLE, 5, qapp), (c.state, c.error, c.message)
        wait_until(lambda: False, 0.2, qapp)
        n = 15 if tgt == "low" else 10
        assert f"{TARGETS[tgt]['display']} 1/{n}" in panel.progress_label.text()
        assert "total 1/35" in panel.progress_label.text()
    finally:
        qapp.removeEventFilter(spy)
        if c.state == RUNNING:
            c.abort("user_abort")
        wait_until(lambda: not c.busy, 5, qapp)
        src.disconnect_device()
        win.close()


def test_panel_locks_plan_and_participant_while_the_plan_is_written(qapp, tmp_path):
    win, src, panel, c = plan_window(qapp, tmp_path)
    try:
        assert c.extend_plan(OPEN_HAND)
        panel._show_controls()                                   # before the write is polled
        for w in (panel.participant_edit, panel.plan_combo, panel.mode_combo, panel.btn_out,
                  panel.btn_start, panel.btn_new_plan, *panel.extend_buttons.values()):
            assert not w.isEnabled(), w
        assert wait_until(lambda: c.plan.total == 235 and not c.busy, 5, qapp), c.error
        wait_until(lambda: False, 0.2, qapp)
        assert panel.participant_edit.isEnabled() and panel.plan_combo.isEnabled()
        assert panel.btn_extend_plan.isEnabled() and panel.btn_extend_open.isEnabled()
    finally:
        src.disconnect_device()
        win.close()
