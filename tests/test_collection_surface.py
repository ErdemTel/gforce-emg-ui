"""Surface Press condition: instructions, the one-time 200 -> 230 plan extension,
opt-in random selection, one-trial edits, recording metadata, progress, and the
.gitignore rules that make recorded data trackable.

Plans and trial.json files live in tmp_path only (real data is never touched).
"""

import collections
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from PyQt5 import QtCore, QtWidgets

from conftest import wait_until
from gforce_emg_ui.collection import plan as plan_mod
from gforce_emg_ui.collection import storage as storage_mod
from gforce_emg_ui.collection.controller import (IDLE, MANUAL_OVERRIDE, REVIEW, RUNNING,
                                                 TrialSpec)
from gforce_emg_ui.collection.plan import (DEFAULT_PLAN_COUNTS, EMPTY_HAND, EXTENDABLE,
                                           EXTENDED, INCOMPATIBLE, PLAN_BACKUP_DIR,
                                           SURFACE_PRESS_EXTENSION, TrialPlan, compute_progress,
                                           counts_toward_plan, eligible_trials,
                                           extend_plan_file, extend_with_surface_press,
                                           extension_problems, max_target_run, merge_objects,
                                           pick_trial, resolve_request, scan_participant,
                                           surface_press_status)
from gforce_emg_ui.collection.protocol import (ABORTED, GRASP, HOLD, MOTION_ORDER, OPERATOR_NOTES,
                                               PHASES, PREPARE, REST, SETTLE, STATIONARY,
                                               SURFACE_OPERATOR_NOTES, SURFACE_PRESS,
                                               SURFACE_PRESS_POSTURE,
                                               TARGET_INSTRUCTIONS, TARGET_ORDER, TARGETS,
                                               participant_instructions, phase_instruction,
                                               task_instructions)
from gforce_emg_ui.collection.storage import atomic_write_text, dumps
from gforce_emg_ui.recorder import HEADERS
from test_collection_plan import rec
from test_collection_selection import plan_window, shown, user_pick, want
from test_collection_workflow import (close_rig, drive, header, make_plan, make_rig, plan_spec,
                                      review, rows, run_trial, sidecar, start_panel_trial)

REPO = Path(__file__).resolve().parents[1]
# sha256 of every grasp / empty-hand instruction text before Surface Press existed (89920f9)
GRASP_TEXT_SHA = "3f1db0a76c93ed04f8da0b10aad66677f1abaa8d1695002af430f3f34e4bc06f"
# sha256 of TrialPlan.create("P01", "real", seed=7) trials before this change
PLAN_SEED7_SHA = "86c8f174b3beafd60138c124739f030792f877cc22faee8de8917d9cee414205"

PREPARE_TEXT = ("Place your index fingertip lightly against the marked area of the desk. Keep a "
                "comfortable slight bend in the finger, maintain a consistent wrist position, "
                "and support your forearm without compressing the armband. Avoid leaning your "
                "body weight onto the finger.")
LEVEL_TEXT = {
    "low": "Maintain light fingertip contact with the surface without intentionally adding "
           "downward pressure. Keep your finger and wrist posture steady.",
    "medium": "Apply a comfortable, repeatable moderate press against the surface. Maintain "
              "the same fingertip position, finger posture, and wrist position.",
    "high": "Apply a firm, comfortable, repeatable press against the surface without changing "
            "finger posture. Do not push maximally or lean your body weight onto the finger.",
}
SETTLE_TEXT = "Gradually establish the instructed pressing effort."
HOLD_TEXT = ("Maintain the instructed pressing effort and keep the finger, wrist, and arm "
             "posture as steady as possible.")
REST_TEXT = "Release the pressing effort and allow your hand and forearm to rest."


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def standard(seed=7, source="real"):
    return TrialPlan.create("P01", source, seed=seed)


def added(plan):
    return [t for t in plan.trials if t["object_id"] == SURFACE_PRESS]


def sp_rec(plan, trial_id, attempt, **kw):
    """Synthetic accepted record; surface-press records carry their trial type."""
    r = rec(plan, trial_id, attempt, **kw)
    if plan.trial(trial_id)["object_id"] == SURFACE_PRESS:
        r["condition"]["trial_type"] = SURFACE_PRESS
    return r


def extend(r):
    """Extend the rig's selected plan through the controller (real I/O path)."""
    assert r.c.extend_plan(), r.c.error
    assert drive(r.c, lambda: not r.c.busy, r.app)
    assert not r.c.error, r.c.error
    assert r.c.plan.total == 230


def surface_seed():
    """A plan seed whose first random pick with only Surface Press ticked is a press."""
    for seed in range(1, 100):
        plan = extend_with_surface_press(standard(seed=seed))
        i = pick_trial(plan, compute_progress(plan, {}), [SURFACE_PRESS], key="0")
        if plan.trials[i]["object_id"] == SURFACE_PRESS:
            return seed
    pytest.fail("no seed found")


def surface_proposal(c):
    """Tick only Surface Press: the automatic proposal is then a press (surface_seed)."""
    c.set_available_objects([SURFACE_PRESS])
    assert c.proposed_trial()["object_id"] == SURFACE_PRESS
    return c.proposed_trial()


@pytest.fixture
def sim(qapp, tmp_path):
    r = make_rig(qapp, tmp_path)
    yield r
    close_rig(r)


# -- instructions ---------------------------------------------------------------

@pytest.mark.parametrize("target", TARGET_ORDER)
def test_surface_press_instructions_for_every_level(target):
    text = participant_instructions(target, STATIONARY, SURFACE_PRESS)
    assert text[PREPARE] == PREPARE_TEXT
    assert text[SETTLE] == f"{SETTLE_TEXT} {LEVEL_TEXT[target]}"
    assert text[HOLD] == HOLD_TEXT and text[REST] == REST_TEXT
    assert text[ABORTED].startswith("Trial stopped. Release the pressing effort")
    for phase in PHASES:                                  # one source for display and metadata
        assert phase_instruction(phase, target, STATIONARY, SURFACE_PRESS) == text[phase]
    info = task_instructions(target, STATIONARY, SURFACE_PRESS)
    assert info["target_instruction"] == LEVEL_TEXT[target]
    assert info["participant_instructions"] == text
    assert info["task_instruction_version"] == "surface-press/1"
    joined = " ".join(text.values()).lower()
    assert "grasp" not in joined and "object" not in joined and "fist" not in joined
    assert "as hard as possible" not in joined and "maximal effort" not in joined


def test_grasp_and_empty_hand_instructions_are_unchanged():
    d = {f"{t}/{m}": participant_instructions(t, m) for t in TARGETS for m in MOTION_ORDER}
    d["_target"], d["_operator"] = TARGET_INSTRUCTIONS, OPERATOR_NOTES
    assert hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest() == GRASP_TEXT_SHA
    for t in TARGETS:
        for m in MOTION_ORDER:
            info = task_instructions(t, m)
            assert info["participant_instructions"] == participant_instructions(t, m, GRASP)
            assert (info["target_instruction"], info["operator_note"]) == (
                TARGET_INSTRUCTIONS[t], OPERATOR_NOTES[t])
            assert info["task_instruction_version"] == "grasp/1"


def test_spec_enforces_stationary_and_the_finger_press_posture(tmp_path):
    base = dict(participant_id="P01", session_id="S01", object_id=SURFACE_PRESS,
                posture_id=SURFACE_PRESS_POSTURE, target_label="high", dataset_role="practice",
                output_root=tmp_path)
    assert TrialSpec(**base).problems() == [] and TrialSpec(**base).trial_type == SURFACE_PRESS
    assert "Surface Press is stationary only" in TrialSpec(
        **base, motion_condition="reach_sweep").problems()
    assert any("posture" in p for p in TrialSpec(**{**base, "posture_id": "grasp"}).problems())
    assert TrialSpec(**{**base, "object_id": "obj01", "posture_id": "grasp",
                        "motion_condition": "reach_sweep"}).trial_type == GRASP
    # a saved object catalog can never turn the condition into an object
    assert SURFACE_PRESS not in dict(merge_objects([[SURFACE_PRESS, "Desk"]]))


# -- plan extension (pure) ---------------------------------------------------------

def test_standard_plan_is_extended_from_200_to_230():
    plan = standard()
    before = dumps(plan.to_dict())
    assert surface_press_status(plan) == (EXTENDABLE, "")
    ext = extend_with_surface_press(plan)
    assert dumps(plan.to_dict()) == before                # the original object is unchanged
    assert (ext.total, ext.plan_id, ext.seed, ext.created) == (230, plan.plan_id, plan.seed,
                                                               plan.created)
    assert [dumps(t) for t in ext.trials[:200]] == [dumps(t) for t in plan.trials]
    new = ext.trials[200:]
    assert collections.Counter(t["target_label"] for t in new) == {"low": 10, "medium": 10,
                                                                   "high": 10}
    assert all((t["object_id"], t["posture_id"], t["motion_condition"], t["trial_type"]) ==
               (SURFACE_PRESS, SURFACE_PRESS_POSTURE, STATIONARY, SURFACE_PRESS) for t in new)
    assert [t["trial_id"] for t in new] == [f"t{i}" for i in range(201, 231)]
    assert [t["index"] for t in new] == list(range(201, 231))
    assert len({t["trial_id"] for t in ext.trials}) == len({t["index"] for t in ext.trials}) == 230
    assert ext.counts == {**plan.counts, SURFACE_PRESS: (10, 10, 10)}
    assert extension_problems(plan, ext) == []
    d = ext.to_dict()
    assert d["schema_version"] == 3 and d["total"] == 230 and d["plan_id"] == plan.plan_id
    assert d["counts_low_medium_high"][SURFACE_PRESS] == [10, 10, 10]
    assert d["motion_counts"][SURFACE_PRESS] == {t: {m: 10 if m == STATIONARY else 0
                                                     for m in MOTION_ORDER} for t in TARGET_ORDER}
    orig = plan.to_dict()
    assert {k: v for k, v in d["motion_counts"].items() if k != SURFACE_PRESS} == \
        orig["motion_counts"]
    e = d["extensions"][0]
    assert e["id"] == SURFACE_PRESS_EXTENSION and (e["total_before"], e["total_after"]) == (200, 230)
    assert e["added_trial_ids"] == [t["trial_id"] for t in new] and isinstance(e["seed"], int)
    assert TrialPlan.from_dict(json.loads(dumps(d))).to_dict() == json.loads(dumps(d))


def test_extension_order_is_seeded_and_spread_out():
    plan = standard(seed=3)
    a, b = extend_with_surface_press(plan), extend_with_surface_press(plan)
    assert a.trials == b.trials                            # reproducible from the plan seed
    seed = a.extensions[0]["seed"]
    assert extend_with_surface_press(plan, seed=seed).trials == a.trials     # saved seed
    assert extend_with_surface_press(plan, seed=seed + 1).trials != a.trials
    for s in range(30):
        ext = extend_with_surface_press(plan, seed=s)
        assert max_target_run((t["object_id"], t["target_label"]) for t in added(ext)) <= 2


def test_extension_twice_is_refused():
    ext = extend_with_surface_press(standard())
    status, why = surface_press_status(ext)
    assert status == EXTENDED and "already has" in why and "230" in why
    with pytest.raises(ValueError, match="already has"):
        extend_with_surface_press(ext)


def test_non_standard_plans_are_refused_with_a_reason():
    plan = standard()
    v1 = TrialPlan.create("P01", "real", seed=7)
    v1.motions = None
    small = TrialPlan.create("P01", "real", seed=7, counts={**DEFAULT_PLAN_COUNTS,
                                                            "obj05": (1, 1, 1)})
    moves = TrialPlan.create("P01", "real", seed=7, motions=(STATIONARY,))
    dup = TrialPlan.from_dict(json.loads(dumps(plan.to_dict())))
    dup.trials[1] = {**dup.trials[1], "trial_id": dup.trials[0]["trial_id"]}
    for bad, why in ((v1, "v1"), (small, "counts"), (moves, "movements"), (dup, "duplicate")):
        status, text = surface_press_status(bad)
        assert status == INCOMPATIBLE and why in text and "Nothing was changed" in text
        with pytest.raises(ValueError):
            extend_with_surface_press(bad)


def test_used_trial_ids_are_skipped():
    plan = TrialPlan.from_dict(json.loads(dumps(standard().to_dict())))
    plan.trials[4] = {**plan.trials[4], "trial_id": "t201"}       # t201 is taken
    assert surface_press_status(plan)[0] == EXTENDABLE
    ext = extend_with_surface_press(plan)
    assert [t["trial_id"] for t in added(ext)] == [f"t{i}" for i in range(202, 232)]
    assert [t["index"] for t in added(ext)] == list(range(201, 231))
    assert extension_problems(plan, ext) == []


def test_new_plans_are_still_the_standard_200_trial_plan():
    plan = standard()
    assert hashlib.sha256(dumps(plan.to_dict()["trials"]).encode()).hexdigest() == PLAN_SEED7_SHA
    d = plan.to_dict()
    assert plan.total == 200 and d["schema_version"] == 2 and "extensions" not in d
    assert SURFACE_PRESS not in plan.counts


# -- plan extension (file) ---------------------------------------------------------

def write(tmp_path, plan):
    path = tmp_path / "P01" / plan.filename
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, dumps(plan.to_dict()))
    return path


def test_plan_file_is_extended_in_place_with_a_verified_backup(tmp_path):
    plan = standard()
    path = write(tmp_path, plan)
    raw = path.read_bytes()
    ext = extend_plan_file(path, plan)
    backup = path.parent / PLAN_BACKUP_DIR / f"{path.stem}.before-surface-press.json"
    assert backup.read_bytes() == raw                       # recoverable original
    assert json.loads(path.read_text()) == json.loads(dumps(ext.to_dict()))
    e = ext.extensions[0]
    assert e["original_sha256"] == hashlib.sha256(raw).hexdigest()
    assert e["backup"] == f"{PLAN_BACKUP_DIR}/{backup.name}"
    scan = scan_participant(path.parent)                     # backup is never loaded as a plan
    assert [p.plan_id for p in scan["plans"]] == [plan.plan_id] and not scan["unreadable"]
    loaded = scan["plans"][0]
    assert loaded.total == 230 and loaded.trials[:200] == plan.trials
    assert not list(path.parent.glob("**/*.tmp"))
    # applying it again changes nothing
    with pytest.raises(ValueError):
        extend_plan_file(path, loaded)
    with pytest.raises(ValueError):
        extend_plan_file(path, plan)                         # stale in-memory plan
    assert path.read_text() == dumps(ext.to_dict())


@pytest.mark.parametrize("fail", ["write", "replace", "validate", "changed_on_disk"])
def test_failed_extension_leaves_the_original_plan_usable(tmp_path, monkeypatch, fail):
    plan = standard()
    path = write(tmp_path, plan)
    raw = path.read_bytes()
    if fail == "write":                                       # e.g. disk full mid-write
        real = plan_mod.atomic_write_text

        def broken(p, text):
            if Path(p) == path:
                raise OSError("No space left on device")
            return real(p, text)
        monkeypatch.setattr(plan_mod, "atomic_write_text", broken)
    elif fail == "replace":
        real_replace = storage_mod.os.replace

        def broken(src, dst):
            if Path(dst) == path:
                raise OSError("rename failed")
            return real_replace(src, dst)
        monkeypatch.setattr(storage_mod.os, "replace", broken)
    elif fail == "validate":
        monkeypatch.setattr(plan_mod, "extension_problems", lambda a, b: ["synthetic problem"])
    else:
        d = json.loads(raw)
        d["trials"][0]["target_label"] = "low" if d["trials"][0]["target_label"] != "low" \
            else "high"
        raw = dumps(d).encode()
        path.write_bytes(raw)
    with pytest.raises((OSError, ValueError)):
        extend_plan_file(path, plan)
    assert path.read_bytes() == raw                           # original untouched
    assert TrialPlan.from_dict(json.loads(raw)).total == 200
    assert not list(path.parent.glob("**/.*.tmp"))
    if fail in ("validate", "changed_on_disk"):
        assert not (path.parent / PLAN_BACKUP_DIR).exists()   # refused before any write


# -- progress and selection (no Qt) --------------------------------------------------

def test_progress_counts_accepted_surface_press_trials_once():
    plan = extend_with_surface_press(standard(source="simulation"), seed=5)
    sp = added(plan)
    lows = [t for t in sp if t["target_label"] == "low"]
    records = {r["attempt_id"]: r for r in [
        sp_rec(plan, lows[0]["trial_id"], "a", t=1),
        sp_rec(plan, lows[0]["trial_id"], "b", t=2),                       # duplicate accept
        sp_rec(plan, lows[1]["trial_id"], "c", review="rejected", t=3),
        sp_rec(plan, lows[2]["trial_id"], "d", protocol="aborted", t=4),
        sp_rec(plan, lows[3]["trial_id"], "e", review="pending", status="finalized", t=5),
        sp_rec(plan, lows[4]["trial_id"], "f", quality="flagged", t=6),
        sp_rec(plan, plan.trials[0]["trial_id"], "g", t=7),                # a grasp trial
    ]}
    p = compute_progress(plan, records)
    assert p.by_cell[(SURFACE_PRESS, "low")] == [1, 10]
    assert p.by_cell[(SURFACE_PRESS, "medium")] == p.by_cell[(SURFACE_PRESS, "high")] == [0, 10]
    assert (p.total_planned, p.accepted, p.remaining) == (230, 2, 228)
    assert sum(n for _, n in p.by_motion.values()) == 200       # movement split = grasp only
    # a record of the right trial but without the surface-press trial type never counts
    bare = rec(plan, lows[5]["trial_id"], "x")
    assert not counts_toward_plan(bare, plan)
    assert counts_toward_plan(sp_rec(plan, lows[5]["trial_id"], "y"), plan)


def test_surface_press_is_opt_in_for_the_random_selection():
    plan = extend_with_surface_press(standard(), seed=5)
    p = compute_progress(plan, {})
    objs = [o for o in DEFAULT_PLAN_COUNTS if o != EMPTY_HAND]
    off = eligible_trials(plan, p, objs)
    on = eligible_trials(plan, p, objs + [SURFACE_PRESS])
    assert len(off) == 200 and len(on) == 230
    assert all(plan.trials[i]["object_id"] != SURFACE_PRESS
               for k in range(200) for i in [pick_trial(plan, p, objs, key=str(k))])
    picks = collections.Counter(plan.trials[pick_trial(plan, p, [SURFACE_PRESS], key=str(k))]
                                ["object_id"] for k in range(300))
    assert set(picks) == {SURFACE_PRESS, EMPTY_HAND}          # empty hand always eligible
    # remaining counts drive the targets: all low presses done -> only medium / high
    lows = [t for t in added(plan) if t["target_label"] == "low"]
    done = {f"a{k}": sp_rec(plan, t["trial_id"], f"a{k}", t=k) for k, t in enumerate(lows)}
    p = compute_progress(plan, done)
    targets = {plan.trials[i]["target_label"] for i in eligible_trials(plan, p, [SURFACE_PRESS])
               if plan.trials[i]["object_id"] == SURFACE_PRESS}
    assert targets == {"medium", "high"}
    # the run rule still applies
    for k in range(100):
        i = pick_trial(plan, p, [SURFACE_PRESS], ["high", "high"], key=str(k))
        assert plan.trials[i]["target_label"] != "high"
    # unavailable combinations: clear reasons
    _, _, why = resolve_request(plan, p, want(SURFACE_PRESS, "low", STATIONARY))
    assert "No remaining planned trials for surface_press / low / stationary" in why
    _, _, why = resolve_request(plan, p, want(SURFACE_PRESS, "high", "reach_sweep"))
    assert "stationary only" in why
    _, _, why = resolve_request(standard(), compute_progress(standard(), {}),
                                want(SURFACE_PRESS, "high", STATIONARY))
    assert "no Surface Press trials" in why


# -- controller ------------------------------------------------------------------

def test_controller_extension_keeps_plan_id_progress_and_recordings(qapp, tmp_path):
    r = make_rig(qapp, tmp_path)
    try:
        make_plan(r)
        c = r.c
        plan_id, original = c.plan.plan_id, [dumps(t) for t in c.plan.trials]
        kept = run_trial(r, plan_spec(r))
        review(r, "accept")
        tid = c.spec.trial_id
        files = {p: sha(p) for p in kept.iterdir()}
        path = c.participant_root() / c.plan.filename
        assert c.extend_plan()
        assert "Plan update in progress" in c.start_blockers(plan_spec(r))    # no recording
        assert c.create_plan(5) is None and not c.extend_plan()               # no second write
        assert drive(c, lambda: not c.busy, r.app) and not c.error, c.error
        assert c.plan.total == 230 and c.plan.plan_id == plan_id
        assert [dumps(t) for t in c.plan.trials[:200]] == original
        assert set(c.progress.completed) == {tid} and c.progress.accepted == 1
        assert "200 → 230" in c.message and PLAN_BACKUP_DIR in c.message
        assert [p.plan_id for p in c.plans] == [plan_id]
        assert {p: sha(p) for p in kept.iterdir()} == files      # recordings untouched
        text = path.read_text()
        assert not c.extend_plan() and "already has" in c.error  # idempotent
        assert path.read_text() == text and c.plan.total == 230
    finally:
        close_rig(r)
    r = make_rig(qapp, tmp_path)                                  # restart: same state from disk
    try:
        assert r.c.plan.plan_id == plan_id and r.c.plan.total == 230
        assert r.c.progress.accepted == 1 and tid in r.c.progress.completed
        assert r.c.extension_status()[0] == EXTENDED
    finally:
        close_rig(r)


def test_failed_extension_in_the_controller_keeps_the_plan(sim, monkeypatch):
    r = sim
    make_plan(r)
    c = r.c
    path = c.participant_root() / c.plan.filename
    raw = path.read_bytes()
    monkeypatch.setattr(plan_mod, "atomic_write_text",
                        lambda p, t: (_ for _ in ()).throw(OSError("disk full")))
    assert c.extend_plan()
    assert drive(c, lambda: not c.busy, r.app)
    assert "NOT extended" in c.error and "disk full" in c.error
    assert c.plan.total == 200 and path.read_bytes() == raw
    assert "Plan update in progress" not in c.start_blockers(plan_spec(r))


def test_unticked_surface_press_stays_pending_and_ticking_includes_it(sim):
    r = sim
    make_plan(r)
    extend(r)
    c = r.c
    objs = [o for o in c.plan.counts if o not in (EMPTY_HAND, SURFACE_PRESS)]
    assert SURFACE_PRESS not in c.available() and c.selection["eligible"] == 200
    c.set_available_objects(objs)
    assert c.selection["eligible"] == 200 and c.proposed_trial()["object_id"] != SURFACE_PRESS
    c.set_available_objects(objs + [SURFACE_PRESS])
    assert c.selection["eligible"] == 230
    assert SURFACE_PRESS in c.selection["available_objects"]
    c.set_available_objects(objs)
    assert len([t for t in c.plan.trials if t["object_id"] == SURFACE_PRESS
                and t["trial_id"] not in c.progress.completed]) == 30


def test_effort_edit_of_a_surface_press_proposal_keeps_the_condition(sim):
    r = sim
    make_plan(r, seed=surface_seed())
    extend(r)
    c = r.c
    auto = surface_proposal(c)
    tgt = next(t for t in TARGET_ORDER if t != auto["target_label"])
    assert c.edit_selection("target_label", tgt)
    t = c.planned_trial()
    assert (t["object_id"], t["target_label"], t["motion_condition"]) == (SURFACE_PRESS, tgt,
                                                                          STATIONARY)
    assert c.pending.edited == ["target_label"] and c.pending.enforced == {}
    assert c.selection_record()["automatic_proposal"]["trial_id"] == auto["trial_id"]
    assert c.clear_edits() and c.planned_trial()["trial_id"] == auto["trial_id"]


def test_object_edit_to_surface_press_keeps_effort_and_forces_stationary(sim):
    r = sim
    make_plan(r)
    extend(r)
    c = r.c
    c.set_available_objects([o for o in c.plan.counts if o not in (EMPTY_HAND, SURFACE_PRESS)])
    auto = c.proposed_trial()
    if auto["motion_condition"] == STATIONARY:            # make the movement non-stationary
        assert c.edit_selection("motion_condition", "reach_sweep")
    before = dict(c.pending.requested)
    assert c.edit_selection("object_id", SURFACE_PRESS)
    assert c.pending.requested == {**before, "object_id": SURFACE_PRESS,
                                   "motion_condition": STATIONARY}
    t = c.planned_trial()
    assert (t["object_id"], t["target_label"]) == (SURFACE_PRESS, before["target_label"])
    rec_ = c.selection_record()
    assert rec_["mode"] == MANUAL_OVERRIDE and rec_["enforced_fields"] == ["motion_condition"]
    assert rec_["availability_bypass"] is True               # surface press was unticked
    # an explicit non-stationary movement is a clear blocker, never substituted
    assert c.edit_selection("motion_condition", "wrist_flex_ext")
    assert c.planned_trial() is None and "stationary only" in c.pending.problem
    assert c.start_blockers(plan_spec_like(r))
    assert c.edit_selection("motion_condition", STATIONARY) and c.planned_trial() is not None
    # leaving surface press restores the movement it replaced (if not edited since)
    assert c.edit_selection("object_id", before["object_id"])
    assert c.pending.requested["motion_condition"] == STATIONARY      # explicit edit: kept
    assert c.edit_selection("motion_condition", "forearm_rotation")
    assert c.edit_selection("object_id", SURFACE_PRESS)
    assert c.pending.requested["motion_condition"] == STATIONARY
    assert c.pending.enforced == {"motion_condition": "forearm_rotation"}
    assert c.edit_selection("object_id", before["object_id"])
    assert c.pending.requested["motion_condition"] == "forearm_rotation"
    assert c.pending.enforced == {} and c.planned_trial() is not None
    assert c.clear_edits() and c.planned_trial()["trial_id"] == auto["trial_id"]
    assert c.pending.enforced == {}


def plan_spec_like(r):
    """Spec as the panel builds it when no planned trial matches."""
    req = r.c.pending.requested
    return TrialSpec(participant_id="P01", session_id="S01", object_id=req["object_id"],
                     posture_id=SURFACE_PRESS_POSTURE, target_label=req["target_label"],
                     motion_condition=req["motion_condition"], dataset_role="development",
                     output_root=r.root)


def test_completed_surface_press_target_gives_a_clear_blocker(qapp, tmp_path):
    r = make_rig(qapp, tmp_path)
    try:
        make_plan(r)
        extend(r)
        c = r.c
        lows = [t for t in c.plan.trials if t["object_id"] == SURFACE_PRESS
                and t["target_label"] == "low"]
        for k, t in enumerate(lows):                       # as if recorded and accepted
            rr = sp_rec(c.plan, t["trial_id"], f"syn{k}", t=1000.0 + k)
            c.records[rr["attempt_id"]] = rr
        c.select_plan(c.plan.plan_id)
        assert c.progress.by_cell[(SURFACE_PRESS, "low")] == [10, 10]
        c.set_available_objects([SURFACE_PRESS])
        assert c.edit_selection("object_id", SURFACE_PRESS)
        assert c.edit_selection("target_label", "low")
        assert c.planned_trial() is None
        assert "No remaining planned trials for surface_press / low / stationary" in \
            c.pending.problem
        assert c.edit_selection("target_label", "high") and c.planned_trial() is not None
    finally:
        close_rig(r)


def test_surface_press_trial_is_recorded_reviewed_and_counted(sim):
    r = sim
    make_plan(r, seed=surface_seed())
    extend(r)
    c = r.c
    auto = surface_proposal(c)
    tgt = auto["target_label"]
    folder = run_trial(r, plan_spec(r))
    assert c.spec.trial_type == SURFACE_PRESS
    assert f"_{SURFACE_PRESS}_{tgt}_" in folder.name and f"_{auto['trial_id']}_a01_" in folder.name
    assert {p.name for p in folder.iterdir()} == {"emg.csv", "imu.csv", "contact.csv",
                                                 "markers.csv", "session.json", "trial.json"}
    for name in ("emg", "imu", "contact", "markers"):        # raw layout unchanged
        assert header(folder / f"{name}.csv") == HEADERS[name]
    assert [m["label"] for m in rows(folder / "markers.csv")][:4] == [
        "PREPARE_START", "SETTLE_START", "HOLD_START", "HOLD_END"]
    d = sidecar(folder)
    assert (d["schema_version"], d["protocol_version"]) == (2, "stiffness-phase0/1.3")
    cond = d["condition"]
    assert (cond["object_id"], cond["trial_type"], cond["posture_id"], cond["motion_condition"]) \
        == (SURFACE_PRESS, SURFACE_PRESS, SURFACE_PRESS_POSTURE, STATIONARY)
    assert cond["object_name"] == "Surface Press (Desk)"
    lab = d["label"]
    assert (lab["target_label"], lab["target_class_id"], lab["target_s"]) == (
        tgt, TARGETS[tgt]["class_id"], TARGETS[tgt]["s"])
    assert lab["label_source"] == "instructed_effort"
    assert lab["task_instruction_version"] == "surface-press/1"
    assert lab["participant_instructions"] == participant_instructions(tgt, STATIONARY,
                                                                       SURFACE_PRESS)
    assert lab["target_instruction"] == LEVEL_TEXT[tgt]
    assert d["training_candidate"]["phase"] == HOLD
    assert d["selection"]["mode"] == "automatic"
    assert SURFACE_PRESS in d["selection"]["available_objects"]
    # Repeat keeps the trial; Reject deletes; only the accepted attempt counts
    review(r, "repeat")
    assert not folder.exists() and c.planned_trial()["trial_id"] == auto["trial_id"]
    folder = run_trial(r, plan_spec(r))
    assert "_a01_" in folder.name                           # deleted attempts are not on disk
    review(r, "accept")
    assert c.progress.by_cell[(SURFACE_PRESS, tgt)] == [1, 10]
    assert c.progress.completed[auto["trial_id"]] == sidecar(folder)["attempt_id"]
    assert sum(n for _, n in c.progress.by_motion.values()) == 200
    reader = pytest.importorskip("emg_classifier.trials")
    tr = reader.read_trial(folder)
    assert tr.problems == {}, tr.problems
    assert (tr.info["object_id"], tr.info["posture_id"]) == (SURFACE_PRESS, SURFACE_PRESS_POSTURE)


def test_grasp_trial_metadata_only_gains_additive_fields(sim):
    r = sim
    make_plan(r)
    extend(r)
    folder = run_trial(r, plan_spec(r))
    t = r.c.spec.planned_trial
    assert t["object_id"] != SURFACE_PRESS
    d = sidecar(folder)
    assert d["condition"]["trial_type"] == GRASP
    assert d["label"]["task_instruction_version"] == "grasp/1"
    assert d["label"]["participant_instructions"] == participant_instructions(
        t["target_label"], t["motion_condition"])
    review(r, "reject")


# -- panel (headless Qt) ------------------------------------------------------------

def test_panel_extension_confirmation_progress_and_repeat_message(qapp, tmp_path, monkeypatch):
    win, src, panel, c = plan_window(qapp, tmp_path)
    asked = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        lambda *a: asked.append(a[2]) or QtWidgets.QMessageBox.Yes)
    infos = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a: infos.append(a[2]))
    try:
        assert "200" not in panel.mode_combo.itemText(0)
        assert not panel.surface_check.isEnabled() and not panel.surface_check.isChecked()
        path = c.participant_root() / c.plan.filename
        panel.btn_extend_plan.click()
        assert len(asked) == 1 and c.plan.plan_id in asked[0] and "200 → 230" in asked[0]
        assert wait_until(lambda: c.plan.total == 230 and not c.busy, 5, qapp), c.error
        wait_until(lambda: False, 0.2, qapp)
        text = panel.progress_label.text()
        assert "/ 230" in text
        assert "Surface Press: LOW 0/10 · MEDIUM 0/10 · HIGH 0/10 · total 0/30" in text
        assert panel.surface_check.isEnabled() and not panel.surface_check.isChecked()
        assert panel.minimumSizeHint().width() <= panel.width()
        raw = path.read_bytes()
        panel.btn_extend_plan.click()                           # second time: message only
        assert len(asked) == 1 and len(infos) == 1 and "already has" in infos[0]
        assert path.read_bytes() == raw
    finally:
        src.disconnect_device()
        win.close()


def test_panel_surface_press_trial_shows_its_instructions_without_relayouts(qapp, tmp_path,
                                                                            monkeypatch):
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
        assert panel._extend_plan()
        assert wait_until(lambda: c.plan.total == 230 and not c.busy, 5, qapp)
        panel.surface_check.setChecked(True)
        assert SURFACE_PRESS in c.available()
        panel.surface_check.setChecked(False)
        assert SURFACE_PRESS not in c.available()
        if shown(panel)["motion_condition"] == STATIONARY:
            user_pick(qapp, panel.motion_combo, "reach_sweep")
        user_pick(qapp, panel.object_combo, SURFACE_PRESS)      # effort kept, stationary forced
        wait_until(lambda: False, 0.2, qapp)
        assert shown(panel)["object_id"] == SURFACE_PRESS
        assert shown(panel)["motion_condition"] == STATIONARY
        assert panel.posture_combo.currentText() == SURFACE_PRESS_POSTURE
        assert "Surface Press is stationary only" in panel.sel_note.text()
        assert "Surface Press (Desk)" in panel.next_label.text()
        if shown(panel)["target_label"] != "medium":            # widest "Next:" line
            user_pick(qapp, panel.target_combo, "medium")
        wait_until(lambda: False, 0.1, qapp)
        assert shown(panel)["object_id"] == SURFACE_PRESS and "MEDIUM" in panel.next_label.text()
        assert panel.minimumSizeHint().width() <= panel.width()  # narrow panel still fits
        tgt = c.planned_trial()["target_label"]
        assert SURFACE_OPERATOR_NOTES[tgt] in panel.instruction_label.text()
        for key, value in dict(prepare_s=0.5, settle_s=0.5, hold_s=2.0).items():
            panel.dur_spins[key].setValue(value)
        qapp.installEventFilter(spy)
        start_panel_trial(qapp, win, panel)
        assert wait_until(lambda: c.machine.phase == PREPARE, 2, qapp)
        assert panel.instruction_label.text() == PREPARE_TEXT
        assert wait_until(lambda: c.machine.phase == HOLD, 5, qapp)
        wait_until(lambda: False, 0.1, qapp)
        layouts.clear()
        wait_until(lambda: False, 1.0, qapp)
        assert len(layouts) <= 3, len(layouts)                # no per-tick re-layout
        assert panel.instruction_label.text() == HOLD_TEXT
        assert f"Target: {TARGETS[tgt]['display']}" in panel.target_label.text()
        assert wait_until(lambda: c.state == REVIEW, 30, qapp)
        wait_until(lambda: False, 0.1, qapp)
        assert panel.btn_accept.isEnabled(), c.accept_blockers
        panel.btn_accept.click()
        assert wait_until(lambda: c.state == IDLE, 5, qapp), (c.state, c.error, c.message)
        wait_until(lambda: False, 0.2, qapp)
        assert f"{TARGETS[tgt]['display']} 1/10" in panel.progress_label.text()
        assert "total 1/30" in panel.progress_label.text()
    finally:
        qapp.removeEventFilter(spy)
        if c.state == RUNNING:
            c.abort("user_abort")
        wait_until(lambda: not c.busy, 5, qapp)
        src.disconnect_device()
        win.close()


# -- recorded data is trackable by git -----------------------------------------------

def check_ignore(paths):
    out = subprocess.run(["git", "check-ignore", "--no-index", *paths], cwd=REPO,
                         capture_output=True, text=True)
    if out.returncode not in (0, 1):
        pytest.skip(f"git check-ignore unavailable: {out.stderr.strip()}")
    return set(out.stdout.splitlines())


def test_research_data_is_trackable_and_secrets_stay_ignored():
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    trial = "data/stiffness_collection/P01/S01/20261015_134056_obj05_medium_s0p5_t001_a01_f889bbb8"
    data = [f"{trial}/{n}" for n in ("emg.csv", "imu.csv", "contact.csv", "markers.csv",
                                     "session.json", "trial.json")] + [
        "data/stiffness_collection/P01/plan_plan-1a2b3c4d.json",
        "data/stiffness_collection/P01/plan_backups/plan_plan-1a2b3c4d.before-surface-press.json",
        "data/stiffness_collection/objects.json", "data/20261013_122545_test1/emg.csv",
        "data/audits/20261013_122545_test1.json", "data/x.npy", "data/x.npz", "data/x.bin",
        "data/x.mat", "data/x.h5", "data/x.hdf5", "data/x.parquet", "data/x.feather",
        "data/x.edf", "data/x.bdf", "data/x.tsv", "recordings/a.csv", "datasets/a.csv"]
    private = [".env", ".env.local", "id.pem", "id.key", "gh.token", ".venv/bin/python",
               "src/.venv/pyvenv.cfg", "pkg/__pycache__/a.pyc", "live_logs/x/predictions.csv",
               "runs/lda/model.joblib", "models/m.pt", "app.log", "data/.trial.json.ab12.tmp",
               "CLAUDE.md", "memory.md"]
    ignored = check_ignore(data + private)
    assert not ignored & set(data), sorted(ignored & set(data))
    assert ignored >= set(private), sorted(set(private) - ignored)
    assert not list(REPO.glob("data/**/.gitignore"))          # no nested rule hides data
