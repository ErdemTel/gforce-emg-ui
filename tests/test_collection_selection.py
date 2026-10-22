"""Editable next trial in Plan mode: random proposal by default, one-trial edits of
object / target / movement resolved to an existing incomplete planned trial.

Synthetic plans and trial.json files live in tmp_path only; simulated trials
check software behaviour, not hardware timing.
"""

import copy
import hashlib
import json
import random
import time

import pytest
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtTest import QTest

from conftest import wait_until
from gforce_emg_ui.collection.controller import (AUTOMATIC, AVAILABILITY_BYPASS_TEXT, IDLE,
                                                 MANUAL_OVERRIDE, REVIEW, RUNNING,
                                                 SAVING_REVIEW, CollectionController, TrialSpec)
from gforce_emg_ui.collection.plan import (EMPTY_HAND, MANUAL_SELECTION_RULE, SELECTION_RULE,
                                           TrialPlan, compute_progress, eligible_trials,
                                           participant_root, pick_trial, resolve_request,
                                           run_banned_target, trial_values)
from gforce_emg_ui.collection.protocol import (MOTION_ORDER, MOTIONS, STATIONARY, TARGET_ORDER, TARGETS,
                                               ProtocolSettings, motion_info,
                                               participant_instructions, target_info)
from gforce_emg_ui.collection.storage import atomic_write_text, dumps, s_tag
from test_collection_motion import v1_plan_dict
from test_collection_plan import rec
from test_collection_workflow import (FAST, close_rig, drive, make_plan, make_rig, make_window,
                                      manual_spec, plan_spec, review, run_trial, sidecar, sim,
                                      start)

F = ("object_id", "target_label", "motion_condition")


def want(obj, tgt, motion):
    return dict(zip(F, (obj, tgt, motion)))


def vals(c, t):
    return trial_values(c.plan, t)


def remaining(c, combo):
    return [i for i, t in enumerate(c.plan.trials)
            if t["trial_id"] not in c.progress.completed and vals(c, t) == combo]


def another(options, *exclude):
    return next(o for o in options if o not in exclude)


def objects(c):
    return [o for o in c.plan.counts if o != EMPTY_HAND]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec_for(r, t):
    """A planned spec for trial t (e.g. the displaced proposal)."""
    c = r.c
    return TrialSpec(participant_id="P01", session_id="S01", object_id=t["object_id"],
                     object_name=c.object_name(t["object_id"]), posture_id=t["posture_id"],
                     target_label=t["target_label"], motion_condition=c.plan.motion_of(t),
                     dataset_role="development", output_root=r.root, planned_trial=t,
                     plan_id=c.plan.plan_id)


def attempt_folders(r):
    return sorted(p for p in r.c.participant_root().glob("S01/*") if p.is_dir())


def write_plan(root, plan):
    proot = participant_root(root, plan.participant_id, simulated=True)
    proot.mkdir(parents=True, exist_ok=True)
    path = proot / plan.filename
    atomic_write_text(path, dumps(plan.to_dict()))
    return path


def write_accepted(root, plan, trials):
    """Synthetic accepted trial.json files (as if recorded in an earlier session)."""
    proot = participant_root(root, plan.participant_id, simulated=True)
    for k, t in enumerate(trials):
        folder = proot / "S00" / f"syn_{t['trial_id']}"
        folder.mkdir(parents=True)
        atomic_write_text(folder / "trial.json",
                          dumps(rec(plan, t["trial_id"], f"syn{t['trial_id']}", t=1000.0 + k)))


def reopen(r):
    r.c.open_participant(r.root, "P01")
    assert drive(r.c, lambda: not r.c.loading and r.c.plan is not None, r.app), r.c.error


def object_proposal(c):
    """Make the automatic proposal an object (not empty hand), deterministically."""
    objs = objects(c)
    for drop in [None] + objs:
        c.set_available_objects(set(objs) - {drop})
        if c.proposed_trial()["object_id"] != EMPTY_HAND:
            return
    pytest.fail("no object proposal for this seed")


def old_pick(plan, progress, available, recent_targets=(), key="", exclude=None, max_run=2):
    """pick_trial as it was before one-trial edits (ec27ed0), verbatim."""
    idx = eligible_trials(plan, progress, available)
    if exclude is not None and len(idx) > 1:
        idx = [i for i in idx if i != exclude]
    if not idx:
        return None
    tail = list(recent_targets)[-max_run:]
    banned = tail[0] if len(tail) == max_run and len(set(tail)) == 1 else None
    pool = [i for i in idx if plan.trials[i]["target_label"] != banned] or idx
    return random.Random(f"{plan.seed}:{key}").choice(pool)


def user_pick(qapp, combo, value):
    """Choose an item the way an operator does (popup + Enter): Qt emits activated."""
    QtWidgets.QApplication.setActiveWindow(combo.window())
    combo.setFocus()
    combo.showPopup()
    qapp.processEvents()
    view = combo.view()
    view.setCurrentIndex(view.model().index(combo.findData(value), 0))
    QTest.keyClick(view, QtCore.Qt.Key_Return)
    qapp.processEvents()
    assert not view.isVisible() and combo.currentData() == value


# -- plan level (no Qt) ---------------------------------------------------------------

def test_random_proposal_is_unchanged():
    plan = TrialPlan.create("P01", "real", seed=5)
    done = {f"a{k}": rec(plan, plan.trials[k]["trial_id"], f"a{k}") for k in range(0, 60, 3)}
    p = compute_progress(plan, done)
    for k in range(40):
        for avail in (set(), {"obj02"}, {"obj01", "obj04"}, set(plan.counts)):
            for recent in ((), ("high", "high"), ("low", "medium")):
                args = (plan, p, avail, recent, str(k), k % 7 if k % 2 else None)
                assert pick_trial(*args) == old_pick(*args)
    assert run_banned_target(["low", "low"]) == "low"
    assert run_banned_target(["low", "high"]) is None and run_banned_target(["low"]) is None


def test_resolver_matches_exactly_and_never_changes_the_plan():
    plan = TrialPlan.create("P01", "real", seed=9)
    before = copy.deepcopy(plan.trials)
    p = compute_progress(plan, {})
    req = want("obj02", "medium", "reach_sweep")
    match = [i for i, t in enumerate(plan.trials) if trial_values(plan, t) == req]
    assert len(match) >= 2
    assert resolve_request(plan, p, req) == (match[0], "lowest_index", "")
    assert resolve_request(plan, p, req, [("proposal", -1), ("kept", match[1])]) == \
        (match[1], "kept", "")
    done = {f"a{i}": rec(plan, plan.trials[i]["trial_id"], f"a{i}") for i in match[:-1]}
    assert resolve_request(plan, compute_progress(plan, done), req)[0] == match[-1]
    done["z"] = rec(plan, plan.trials[match[-1]]["trial_id"], "z")
    assert resolve_request(plan, compute_progress(plan, done), req) == (
        None, "", "No remaining planned trials for obj02 / medium / reach/sweep. "
                  "Change a setting or reset edits.")
    assert resolve_request(plan, p, {"object_id": "obj02"})[2] == "Choose target and movement."
    assert "obj06 has no trials in this plan" in \
        resolve_request(plan, p, want("obj06", "low", STATIONARY))[2]
    v1 = TrialPlan.from_dict(v1_plan_dict())
    p1 = compute_progress(v1, {})
    assert "Plan v1 has no movement" in resolve_request(v1, p1, want("obj01", "low", "reach_sweep"))[2]
    assert resolve_request(v1, p1, want("obj01", "low", STATIONARY))[0] is not None
    assert plan.trials == before


# -- controller ---------------------------------------------------------------------

def test_without_edits_the_pending_trial_is_the_random_proposal(sim):
    r = sim
    c = r.c
    make_plan(r)
    assert c.pending.mode == AUTOMATIC and c.pending.edited == []
    assert c.planned_trial() is c.proposed_trial()
    assert c.plan_cursor == pick_trial(c.plan, c.progress, c.available(), [], "0")
    assert c.selection_record() == {"mode": AUTOMATIC, **c.selection}
    version, cursor = c.selection_version, c.plan_cursor
    drive(c, lambda: False, r.app, timeout=0.3)               # idle ticks never re-roll
    assert (c.selection_version, c.plan_cursor) == (version, cursor)
    assert c.clear_edits() and c.plan_cursor == cursor         # nothing to reset: no re-roll
    c.set_available_objects({"obj02"})                         # checkboxes still govern it
    assert c.pending.mode == AUTOMATIC
    assert c.planned_trial()["object_id"] in {"obj02", EMPTY_HAND}


def test_each_edit_changes_only_its_own_field(sim):
    r = sim
    c = r.c
    make_plan(r)
    trials_before = copy.deepcopy(c.plan.trials)
    auto = c.proposed_trial()
    a = vals(c, auto)
    cursor, key = c.plan_cursor, c.selection["key"]
    obj = another(objects(c), a["object_id"])
    assert c.edit_selection("object_id", obj)
    exp = {**a, "object_id": obj}
    assert c.pending.requested == exp and vals(c, c.planned_trial()) == exp
    assert c.planned_trial()["trial_id"] != auto["trial_id"]
    assert c.pending.edited == ["object_id"] and c.differing_fields() == ["object_id"]
    assert c.planned_trial()["posture_id"] == "grasp"
    tgt = another(TARGET_ORDER, a["target_label"])
    assert c.edit_selection("target_label", tgt)
    exp["target_label"] = tgt
    assert c.pending.requested == exp and vals(c, c.planned_trial()) == exp
    mot = another(MOTION_ORDER, a["motion_condition"])
    assert c.edit_selection("motion_condition", mot)
    exp["motion_condition"] = mot
    assert c.pending.requested == exp and vals(c, c.planned_trial()) == exp
    assert c.pending.edited == list(F) and c.pending.mode == MANUAL_OVERRIDE
    # browsing never re-rolls, consumes or changes the plan
    assert (c.plan_cursor, c.selection["key"]) == (cursor, key)
    assert c.progress.attempted == 0 and c.progress.completed == {}
    assert c.plan.trials == trials_before
    assert not c.edit_selection("target_label", "max") and not c.edit_selection("posture_id", "x")
    assert c.pending.requested == exp


@pytest.mark.parametrize("name", ["target_label", "motion_condition"])
def test_target_only_or_movement_only_edit(sim, name):
    r = sim
    c = r.c
    make_plan(r)
    a = vals(c, c.proposed_trial())
    value = another(TARGET_ORDER if name == "target_label" else MOTION_ORDER, a[name])
    assert c.edit_selection(name, value)
    exp = {**a, name: value}
    assert c.pending.requested == exp and vals(c, c.planned_trial()) == exp
    assert c.differing_fields() == [name]
    assert c.edit_selection(name, a[name])                     # back to the proposal's value
    assert c.planned_trial() is c.proposed_trial()             # the proposal is reused
    assert c.pending.mode == MANUAL_OVERRIDE and c.differing_fields() == []


def test_edited_trial_is_recorded_and_counted_under_the_resolved_identity(sim):
    r = sim
    c = r.c
    make_plan(r)
    plan_file = next(c.participant_root().glob("plan_*.json"))
    digest = sha(plan_file)
    auto = dict(c.proposed_trial())
    a = vals(c, auto)
    obj = another(objects(c), a["object_id"])
    assert c.edit_selection("object_id", obj)
    t = dict(c.planned_trial())
    spec = plan_spec(r)
    assert (spec.trial_id, spec.object_id, spec.posture_id) == (t["trial_id"], obj, "grasp")
    folder = run_trial(r, spec)
    tgt, mot = a["target_label"], a["motion_condition"]
    assert f"_{obj}_{tgt}_{s_tag(TARGETS[tgt]['s'])}_{t['trial_id']}_a01_" in folder.name
    d = sidecar(folder)
    assert (d["trial_id"], d["planned_trial_index"], d["plan_id"]) == \
        (t["trial_id"], t["index"], c.plan.plan_id)
    assert d["condition"]["object_id"] == obj and d["condition"]["posture_id"] == "grasp"
    assert d["condition"]["motion_condition"] == mot and d["motion"] == motion_info(mot)
    assert d["training_candidate"]["motion_condition"] == mot
    assert {k: d["label"][k] for k in target_info(tgt)} == target_info(tgt)
    assert d["label"]["participant_instructions"] == participant_instructions(tgt, mot)
    session = json.loads((folder / "session.json").read_text())
    assert session["stiffness_collection"]["trial_id"] == t["trial_id"]
    sel = d["selection"]
    assert sel["mode"] == MANUAL_OVERRIDE and sel["rule"] == MANUAL_SELECTION_RULE
    assert sel["edited_fields"] == ["object_id"] and sel["differing_fields"] == ["object_id"]
    assert sel["requested"] == {**a, "object_id": obj}
    assert sel["resolved"] == {"trial_id": t["trial_id"], "planned_trial_index": t["index"],
                               "how": "lowest_index"}
    ap = sel["automatic_proposal"]
    assert ap["trial_id"] == auto["trial_id"] and {k: ap[k] for k in F} == a
    assert ap["selection"]["rule"] == SELECTION_RULE
    assert ap["selection"]["available_objects"] == objects(c)
    assert (sel["availability_bypass"], sel["ordering_preference_bypass"], sel["repeat"]) == \
        (False, False, False)

    review(r, "accept")
    p = c.progress
    assert p.completed == {t["trial_id"]: d["attempt_id"]}     # the displaced proposal stays
    assert auto["trial_id"] not in p.completed
    assert p.by_cell[(obj, tgt)][0] == 1 and sum(v[0] for v in p.by_cell.values()) == 1
    assert p.by_motion[mot][0] == 1 and sum(v[0] for v in p.by_motion.values()) == 1
    assert c.pending.mode == AUTOMATIC and c.pending.edited == []      # one-shot
    assert c.selection["recent_targets"] == [tgt]              # history = what was recorded
    assert sha(plan_file) == digest
    # restart: progress and the next pick come from trial.json files only
    c2 = CollectionController(r.hub, r.rec, r.src, settings=ProtocolSettings(**FAST))
    c2.open_participant(r.root, "P01")
    assert drive(c2, lambda: not c2.loading and c2.plan is not None, r.app)
    assert c2.progress.completed == p.completed
    assert (c2.progress.by_cell, c2.progress.by_motion) == (p.by_cell, p.by_motion)
    assert c2.pending.mode == AUTOMATIC
    assert c2.planned_trial()["trial_id"] == c.planned_trial()["trial_id"]
    c2.io.shutdown()


def test_missing_combination_stays_visible_blocks_start_and_records_nothing(sim):
    r = sim
    c = r.c
    make_plan(r)
    a = vals(c, c.proposed_trial())
    x = another(objects(c), a["object_id"])
    combo = want(x, a["target_label"], a["motion_condition"])
    for _ in range(len(remaining(c, combo))):                  # use up x / target / movement
        for f in F:
            assert c.edit_selection(f, combo[f])
        assert vals(c, c.planned_trial()) == combo
        run_trial(r, plan_spec(r))
        review(r, "accept")
    assert not remaining(c, combo) and c.pending.mode == AUTOMATIC
    proposal = c.proposed_trial()
    for f in F:
        c.edit_selection(f, combo[f])
    p = c.pending
    assert p.requested == combo and c.planned_trial() is None
    assert p.problem == (f"No remaining planned trials for {x} / {a['target_label']} / "
                         f"{MOTIONS[a['motion_condition']]['display'].lower()}. "
                         "Change a setting or reset edits.")
    folders = attempt_folders(r)
    old = spec_for(r, proposal)                                # never the old fallback
    assert not c.start_trial(old) and "No remaining planned trials" in c.error
    blocked = TrialSpec(participant_id="P01", session_id="S01", object_id=x,
                        posture_id="grasp", target_label=combo["target_label"],
                        motion_condition=combo["motion_condition"],
                        dataset_role="development", output_root=r.root)
    assert not c.start_trial(blocked) and "come from the plan" in c.error
    assert attempt_folders(r) == folders and c.pending.requested == combo
    # a second edit reaches a valid combination
    tgt = next(t for t in TARGET_ORDER if remaining(c, {**combo, "target_label": t}))
    assert c.edit_selection("target_label", tgt)
    assert vals(c, c.planned_trial()) == {**combo, "target_label": tgt} and not c.pending.problem


def test_object_override_is_one_trial_only_and_logged(sim):
    r = sim
    c = r.c
    make_plan(r)
    c.set_available_objects({"obj01"})
    a = vals(c, c.proposed_trial())
    assert a["object_id"] in {"obj01", EMPTY_HAND}
    cursor = c.plan_cursor
    assert c.edit_selection("object_id", "obj04")
    p = c.pending
    assert p.availability_bypass and p.warnings == [AVAILABILITY_BYPASS_TEXT]
    assert c.planned_trial()["object_id"] == "obj04"
    assert c.available() == {"obj01"} and c.plan_cursor == cursor     # nothing ticked
    folder = run_trial(r, plan_spec(r))
    sel = sidecar(folder)["selection"]
    assert sel["availability_bypass"] is True and sel["available_objects"] == ["obj01"]
    assert sel["automatic_proposal"]["selection"]["available_objects"] == ["obj01"]
    review(r, "accept")
    assert c.pending.mode == AUTOMATIC and c.available() == {"obj01"}
    assert c.proposed_trial()["object_id"] in {"obj01", EMPTY_HAND}
    assert c.selection["available_objects"] == ["obj01"]


def test_unticking_an_unedited_object_invalidates_the_draft_without_replacing_it(sim):
    r = sim
    c = r.c
    make_plan(r)
    object_proposal(c)
    a = vals(c, c.proposed_trial())
    tgt = another(TARGET_ORDER, a["target_label"])
    assert c.edit_selection("target_label", tgt)
    c.set_available_objects(set(objects(c)) - {a["object_id"]})
    p = c.pending
    assert p.requested == {**a, "target_label": tgt} and c.planned_trial() is None
    assert p.problem.startswith(f"{a['object_id']} is not ticked as available")
    assert c.start_blockers(spec_for(r, c.proposed_trial()))     # no silent fallback
    c.set_available_objects(set(objects(c)))
    assert vals(c, c.planned_trial()) == {**a, "target_label": tgt}


def test_reset_restores_the_proposal_and_browsing_never_rerolls(sim):
    r = sim
    c = r.c
    make_plan(r)
    object_proposal(c)
    auto, sel, cursor = c.proposed_trial(), copy.deepcopy(c.selection), c.plan_cursor
    a = vals(c, auto)
    others = [o for o in objects(c) if o != a["object_id"]]
    for f, v in (("object_id", others[0]), ("target_label", another(TARGET_ORDER, a["target_label"])),
                 ("motion_condition", another(MOTION_ORDER, a["motion_condition"])),
                 ("object_id", others[1]), ("object_id", EMPTY_HAND)):
        assert c.edit_selection(f, v)
        assert c.plan_cursor == cursor and c.selection == sel and c.progress.attempted == 0
    assert c.clear_edits()
    assert c.pending.mode == AUTOMATIC and c.planned_trial() is auto
    assert c.selection == sel and c.selection_record() == {"mode": AUTOMATIC, **sel}
    # the original proposal is no longer valid: a normal pick by the existing selector
    assert c.edit_selection("object_id", others[0])
    avail = set(objects(c)) - {a["object_id"]}
    c.set_available_objects(avail)
    assert c.pending.requested["object_id"] == others[0]      # the draft is kept meanwhile
    assert c.clear_edits()
    expected = pick_trial(c.plan, c.progress, avail, [], c.selection["key"])
    assert c.pending.mode == AUTOMATIC and c.plan_cursor == expected
    assert c.planned_trial()["object_id"] in avail | {EMPTY_HAND}


def test_failed_start_keeps_edits_and_busy_states_refuse_changes(sim):
    r = sim
    c = r.c
    make_plan(r)
    a = vals(c, c.proposed_trial())
    assert c.edit_selection("target_label", another(TARGET_ORDER, a["target_label"]))
    req = dict(c.pending.requested)
    manual = r.rec.start("blocker", root=r.root.parent / "manual")   # recorder busy
    assert drive(c, lambda: c.readiness is not None and c.readiness.ready, r.app)
    assert not c.start_trial(plan_spec(r)) and "Recorder busy" in c.error
    assert c.pending.requested == req and c.pending.edited == ["target_label"]
    manual.request_stop()
    assert manual.wait(5)
    start(r, plan_spec(r))
    assert not c.start_trial(plan_spec(r)) and "busy" in c.error      # one attempt only

    def locked():
        assert not c.edit_selection("object_id", "obj05") and not c.clear_edits()
        assert c.pending.requested == req
        assert c.attempt_selection["edited_fields"] == ["target_label"]
    states = {c.state}
    locked()
    while c.state == RUNNING:
        r.app.processEvents()
        c.tick()
        time.sleep(0.005)
    states.add(c.state)                                       # finalizing (or review)
    locked()
    assert drive(c, lambda: c.state == REVIEW, r.app)
    locked()
    assert c.submit_review("accept")
    states |= {REVIEW, c.state}
    locked()
    assert drive(c, lambda: c.state == IDLE, r.app)
    assert {RUNNING, REVIEW, SAVING_REVIEW} <= states
    assert len(attempt_folders(r)) == 1
    assert c.pending.mode == AUTOMATIC


def test_reject_returns_to_automatic_and_repeat_retries_the_edited_trial(sim):
    r = sim
    c = r.c
    make_plan(r)
    a = vals(c, c.proposed_trial())
    c.edit_selection("object_id", another(objects(c), a["object_id"]))
    c.edit_selection("motion_condition", another(MOTION_ORDER, a["motion_condition"]))
    t, req = dict(c.planned_trial()), dict(c.pending.requested)
    first = run_trial(r, plan_spec(r))
    first_id = c.sidecar["attempt_id"]
    review(r, "repeat")
    assert not first.exists()
    assert c.planned_trial()["trial_id"] == t["trial_id"] and c.pending.requested == req
    assert c.pending.edited == ["object_id", "motion_condition"] and c.pending.repeat
    second = run_trial(r, plan_spec(r))
    d2 = sidecar(second)
    assert d2["trial_id"] == t["trial_id"] and d2["attempt_id"] != first_id
    assert d2["attempt_number"] == 1 and d2["condition"]["object_id"] == req["object_id"]
    assert d2["selection"]["mode"] == MANUAL_OVERRIDE and d2["selection"]["repeat"] is True
    assert d2["selection"]["requested"] == req
    review(r, "reject")
    assert not second.exists() and c.progress.completed == {}
    assert c.pending.mode == AUTOMATIC
    assert c.selection["excluded_index"] == t["index"] - 1   # just-rejected exclusion
    assert c.planned_trial()["trial_id"] != t["trial_id"]


def test_aborted_edited_attempt_keeps_its_identity_through_review(sim):
    r = sim
    c = r.c
    make_plan(r)
    a = vals(c, c.proposed_trial())
    c.edit_selection("target_label", another(TARGET_ORDER, a["target_label"]))
    t = dict(c.planned_trial())
    start(r, plan_spec(r))
    folder = c.session.folder
    assert c.abort("user_abort")
    assert drive(c, lambda: c.state == REVIEW, r.app)
    d = sidecar(folder)
    assert d["outcome"]["abort_reason"] == "user_abort" and not c.acceptable
    assert d["trial_id"] == t["trial_id"] and d["selection"]["mode"] == MANUAL_OVERRIDE
    assert c.attempt_selection == d["selection"] and c.pending.resolved == t["index"] - 1
    review(r, "repeat")
    assert c.planned_trial()["trial_id"] == t["trial_id"] and c.pending.repeat


def test_object_and_empty_hand_switch_only_the_dependent_posture(qapp, tmp_path):
    win, src, panel, c = plan_window(qapp, tmp_path)
    try:
        object_proposal(c)
        panel.tick()
        a = vals(c, c.proposed_trial())
        assert panel.posture_combo.currentText() == "grasp"
        user_pick(qapp, panel.object_combo, EMPTY_HAND)
        assert c.pending.requested == {**a, "object_id": EMPTY_HAND}
        assert c.planned_trial()["posture_id"] == "fist" and panel.spec().posture_id == "fist"
        assert panel.posture_combo.currentText() == "fist"
        assert "Posture fist follows empty_hand." in panel.sel_note.text()
        user_pick(qapp, panel.object_combo, another(objects(c), a["object_id"]))
        assert panel.posture_combo.currentText() == "grasp"
        assert (c.pending.requested["target_label"], c.pending.requested["motion_condition"]) \
            == (a["target_label"], a["motion_condition"])
    finally:
        src.disconnect_device()
        win.close()


def test_manual_third_identical_target_is_allowed_with_a_warning(sim):
    r = sim
    c = r.c
    make_plan(r)
    for _ in range(2):
        c.edit_selection("target_label", "high")
        run_trial(r, plan_spec(r))
        review(r, "accept")
    assert c.proposed_trial()["target_label"] != "high"        # automatic rule unchanged
    assert c.edit_selection("target_label", "high")
    p = c.pending
    assert p.ordering_bypass and p.requested["target_label"] == "high"
    assert any("Third high target in a row" in w for w in p.warnings)
    folder = run_trial(r, plan_spec(r))
    sel = sidecar(folder)["selection"]
    assert sel["ordering_preference_bypass"] is True and sel["recent_targets"] == ["high", "high"]
    assert sidecar(folder)["label"]["target_label"] == "high"


def test_participant_and_plan_changes_drop_idle_edits(sim):
    r = sim
    c = r.c
    make_plan(r)
    first_plan = c.plan.plan_id
    a = vals(c, c.proposed_trial())
    c.edit_selection("object_id", another(objects(c), a["object_id"]))
    c.open_participant(r.root, "P02")
    assert c.pending is None and c.planned_trial() is None
    assert drive(c, lambda: not c.loading, r.app) and c.pending is None    # P02: no plan
    reopen(r)
    assert c.pending.mode == AUTOMATIC
    c.edit_selection("target_label", another(TARGET_ORDER, c.pending.requested["target_label"]))
    assert c.create_plan(12) is not None
    assert drive(c, lambda: c.plan.plan_id != first_plan, r.app), c.error
    assert c.pending.mode == AUTOMATIC
    stale = spec_for(r, c.planned_trial())                    # identity from the other plan
    c.edit_selection("target_label", another(TARGET_ORDER, c.pending.requested["target_label"]))
    c.select_plan(first_plan)
    assert c.pending.mode == AUTOMATIC and c.plan.trial(c.planned_trial()["trial_id"]) \
        is c.planned_trial()
    assert "No saved plan selected" in c.start_blockers(stale)


def test_no_proposal_allows_an_explicit_choice_without_borrowing(sim):
    r = sim
    c = r.c
    plan = TrialPlan.create("P01", "simulation", seed=4)
    write_plan(r.root, plan)
    write_accepted(r.root, plan, [t for t in plan.trials if t["object_id"] == EMPTY_HAND])
    reopen(r)
    c.set_available_objects(set())
    assert c.proposed_trial() is None and c.planned_trial() is None
    assert c.pending.requested == dict.fromkeys(F) and c.pending.mode == AUTOMATIC
    assert c.edit_selection("object_id", "obj03")
    assert c.pending.requested == {"object_id": "obj03", "target_label": None,
                                   "motion_condition": None}
    assert c.planned_trial() is None and c.pending.problem == "Choose target and movement."
    c.edit_selection("target_label", "high")
    c.edit_selection("motion_condition", "forearm_rotation")
    t = c.planned_trial()
    assert vals(c, t) == want("obj03", "high", "forearm_rotation") and c.pending.availability_bypass
    sel = c.selection_record()
    assert sel["automatic_proposal"] is None and sel["differing_fields"] == list(F)
    run_trial(r, plan_spec(r))
    review(r, "accept")
    assert t["trial_id"] in c.progress.completed and c.progress.accepted == 41


def test_completed_plan_off_plan_objects_and_manual_mode(sim):
    r = sim
    c = r.c
    plan = TrialPlan.create("P01", "simulation", seed=2)
    write_plan(r.root, plan)
    write_accepted(r.root, plan, plan.trials)
    reopen(r)
    assert c.progress.remaining == 0 and c.planned_trial() is None
    for f, v in zip(F, ("obj01", "low", STATIONARY)):
        c.edit_selection(f, v)
    assert c.pending.problem.startswith("No remaining planned trials for obj01 / low / stationary")
    c.edit_selection("object_id", "obj06")
    assert "obj06 has no trials in this plan" in c.pending.problem
    assert c.planned_trial() is None
    # off-plan objects stay in Manual mode: never a development-plan slot
    folder = run_trial(r, manual_spec(r, object_id="obj06", object_name="Ball"))
    d = sidecar(folder)
    assert d["selection"] is None and d["plan_id"] is None and d["dataset_role"] == "practice"
    review(r, "accept")
    assert c.progress.accepted == 200
    assert any("come from the plan" in b for b in
               c.start_blockers(manual_spec(r, object_id="obj06", dataset_role="development")))


def test_v1_plan_edits_stay_stationary_and_files_are_untouched(sim):
    r = sim
    c = r.c
    v1 = v1_plan_dict()
    path = participant_root(r.root, "P01", simulated=True) / f"plan_{v1['plan_id']}.json"
    path.parent.mkdir(parents=True)
    atomic_write_text(path, dumps(v1))
    digest = sha(path)
    reopen(r)
    a = vals(c, c.proposed_trial())
    assert c.plan.motions is None and a["motion_condition"] == STATIONARY
    obj = another(objects(c), a["object_id"])
    assert c.edit_selection("object_id", obj)
    assert vals(c, c.planned_trial()) == {**a, "object_id": obj}
    c.edit_selection("motion_condition", "reach_sweep")
    assert c.planned_trial() is None and c.pending.problem.startswith("Plan v1 has no movement")
    c.edit_selection("motion_condition", STATIONARY)
    folder = run_trial(r, plan_spec(r))
    d = sidecar(folder)
    assert d["condition"]["motion_condition"] == STATIONARY and d["condition"]["object_id"] == obj
    assert d["selection"]["mode"] == MANUAL_OVERRIDE
    review(r, "accept")
    assert c.progress.accepted == 1 and sha(path) == digest


def test_classifier_reader_accepts_an_edited_trial(sim):
    trials = pytest.importorskip("emg_classifier.trials")
    r = sim
    c = r.c
    make_plan(r)
    a = vals(c, c.proposed_trial())
    obj = another(objects(c), a["object_id"])
    c.edit_selection("object_id", obj)
    t = dict(c.planned_trial())
    folder = run_trial(r, plan_spec(r))
    review(r, "accept")
    tr = trials.read_trial(folder)
    assert tr.problems == {}, tr.problems
    assert tr.reasons(trials.Selection(dataset_role="development",
                                       data_source="simulation")) == []
    assert (tr.info["trial_id"], tr.info["object_id"], tr.info["label"]) == \
        (t["trial_id"], obj, a["target_label"])
    assert tr.info["planned_trial_index"] == t["index"]


# -- panel (headless Qt) ------------------------------------------------------------

def plan_window(qapp, tmp_path, create=True):
    win, src, panel = make_window(qapp, tmp_path)
    win.show()
    win.record_tabs.setCurrentIndex(1)
    panel.mode_combo.setCurrentIndex(0)
    c = win.collection
    if create:
        panel._new_plan(confirm=False)
    assert wait_until(lambda: c.plan is not None and c.pending is not None, 5, qapp)
    wait_until(lambda: False, 0.2, qapp)                      # a few timer ticks
    return win, src, panel, c


def shown(panel):
    return {n: cb.currentData() for n, cb in panel._sel_combos.items()}


def test_panel_edits_one_field_at_a_time_and_refreshes_never_edit(qapp, tmp_path):
    win, src, panel, c = plan_window(qapp, tmp_path)
    try:
        assert c.pending.edited == []                          # display alone never edits
        auto = dict(c.proposed_trial())
        a = vals(c, auto)
        assert shown(panel) == a and all(cb.isEnabled() for cb in panel._sel_combos.values())
        assert "Automatic" in panel.sel_state.text() and auto["trial_id"] in panel.next_label.text()
        assert "random pick of 200 eligible" in panel.next_label.text()
        assert not panel.btn_reset_sel.isEnabled() and panel.sel_note.isHidden()
        assert not panel.role_combo.isEnabled() and panel.role_combo.currentData() == "development"
        assert not panel.posture_combo.isEnabled()

        obj = another(objects(c), a["object_id"])
        user_pick(qapp, panel.object_combo, obj)
        assert c.pending.requested == {**a, "object_id": obj} and shown(panel) == c.pending.requested
        t = c.planned_trial()
        text = panel.next_label.text()
        assert t["trial_id"] in text and "changed: object" in text and "random pick" not in text
        assert "Edited" in panel.sel_state.text() and panel.btn_reset_sel.isEnabled()
        assert f"Automatic proposal: {auto['trial_id']}" in panel.sel_note.text()
        assert (panel.spec().trial_id, panel.spec().object_id) == (t["trial_id"], obj)
        tgt = another(TARGET_ORDER, a["target_label"])
        user_pick(qapp, panel.target_combo, tgt)
        assert c.pending.requested == want(obj, tgt, a["motion_condition"])
        mot = another(MOTION_ORDER, a["motion_condition"])
        user_pick(qapp, panel.motion_combo, mot)
        final = want(obj, tgt, mot)
        assert c.pending.requested == final and c.pending.edited == list(F)
        assert vals(c, c.planned_trial()) == final
        assert panel.minimumSizeHint().width() <= panel.width()       # narrow panel still fits

        # timer ticks, dashboard and Recording tab switches keep the edits
        version, widgets = c.selection_version, len(panel.findChildren(QtWidgets.QWidget))
        for i in range(win.tabs.count()):
            win.tabs.setCurrentIndex(i)
            qapp.processEvents()
        for i in (0, 1, 0, 1):
            win.record_tabs.setCurrentIndex(i)
            qapp.processEvents()
        wait_until(lambda: False, 0.5, qapp)
        assert c.pending.requested == final and c.selection_version == version
        assert shown(panel) == final
        assert len(panel.findChildren(QtWidgets.QWidget)) == widgets
        # programmatic refreshes are never edits; the tick shows the controller's values
        panel._objects_shown = None
        panel.tick()                                          # object list rebuilt
        panel.target_combo.setCurrentIndex(panel.target_combo.findData(a["target_label"]))
        panel.tick()
        assert c.pending.requested == final and c.pending.edited == list(F)
        assert shown(panel) == final

        panel.btn_reset_sel.click()
        assert c.pending.mode == AUTOMATIC and c.planned_trial()["trial_id"] == auto["trial_id"]
        assert shown(panel) == a and "Automatic" in panel.sel_state.text()
        assert panel.sel_note.isHidden() and not panel.btn_reset_sel.isEnabled()
    finally:
        src.disconnect_device()
        win.close()


def test_open_dropdown_is_not_disturbed_by_ticks(qapp, tmp_path):
    win, src, panel, c = plan_window(qapp, tmp_path)
    try:
        combo = panel.target_combo
        QtWidgets.QApplication.setActiveWindow(win)
        combo.setFocus()
        changes = []
        combo.currentIndexChanged.connect(changes.append)
        before = shown(panel)
        combo.showPopup()
        qapp.processEvents()
        assert combo.view().isVisible()
        focus, row = QtWidgets.QApplication.focusWidget(), combo.view().currentIndex().row()
        # the proposal changes underneath (as after an I/O completion) while the list is open
        c.set_available_objects({another(objects(c), before["object_id"])})
        wait_until(lambda: False, 0.6, qapp)                  # ~12 ticks
        assert combo.view().isVisible() and changes == [] and shown(panel) == before
        assert combo.view().currentIndex().row() == row
        assert QtWidgets.QApplication.focusWidget() is focus
        assert c.pending.edited == []
        combo.hidePopup()
        wait_until(lambda: False, 0.1, qapp)
        assert shown(panel) == c.pending.requested            # shown once the list closed
    finally:
        src.disconnect_device()
        win.close()


def test_panel_missing_combination_then_valid_edit_and_start(qapp, tmp_path):
    plan = TrialPlan.create("P01", "simulation", seed=6)
    root = tmp_path / "stiffness_collection"
    write_plan(root, plan)
    write_accepted(root, plan, [t for t in plan.trials if t["object_id"] == "obj03"])
    win, src, panel, c = plan_window(qapp, tmp_path, create=False)
    try:
        assert c.plan.plan_id == plan.plan_id and c.progress.accepted == 32
        a = vals(c, c.proposed_trial())
        assert a["object_id"] != "obj03"
        assert wait_until(lambda: panel.btn_start.isEnabled(), 5, qapp), panel.ready_label.text()
        user_pick(qapp, panel.object_combo, "obj03")
        panel.tick()
        assert c.pending.requested == {**a, "object_id": "obj03"} and c.planned_trial() is None
        assert shown(panel) == c.pending.requested                # requested values stay visible
        assert not panel.btn_start.isEnabled()
        assert "No remaining planned trials for obj03" in panel.ready_label.text()
        assert "No remaining planned trials for obj03" in panel.sel_note.text()
        assert not panel.sel_note.isHidden() and "no match" in panel.next_label.text()
        assert all(cb.isEnabled() for cb in panel._sel_combos.values())
        session_dir = c.participant_root() / "S01"
        assert panel.start_trial() is False and not session_dir.exists()
        y = another(objects(c), a["object_id"], "obj03")
        user_pick(qapp, panel.object_combo, y)
        t = dict(c.planned_trial())
        assert vals(c, t) == {**a, "object_id": y}
        assert wait_until(lambda: panel.btn_start.isEnabled(), 5, qapp), panel.ready_label.text()
        panel.btn_start.click()
        assert c.state == RUNNING
        qapp.processEvents()
        assert not any(cb.isEnabled() for cb in panel._sel_combos.values())
        assert not panel.btn_reset_sel.isEnabled() and "Edited" in panel.sel_state.text()
        assert (c.sidecar["trial_id"], c.sidecar["condition"]["object_id"]) == (t["trial_id"], y)
        assert wait_until(lambda: c.state == REVIEW, 10, qapp)
        assert not any(cb.isEnabled() for cb in panel._sel_combos.values())
        panel.btn_accept.click()
        assert wait_until(lambda: c.state == IDLE, 5, qapp)
        assert t["trial_id"] in c.progress.completed and c.progress.accepted == 33
        assert c.pending.mode == AUTOMATIC and "Automatic" in panel.sel_state.text()
    finally:
        src.disconnect_device()
        win.close()


def test_panel_mode_session_and_output_changes_clear_edits(qapp, tmp_path):
    win, src, panel, c = plan_window(qapp, tmp_path)
    try:
        def edit():
            cur = c.pending.requested["target_label"]
            user_pick(qapp, panel.target_combo, another(TARGET_ORDER, cur))
            assert c.pending.edited == ["target_label"]
        edit()
        panel.mode_combo.setCurrentIndex(1)                   # Manual mode
        assert c.pending.edited == []
        assert panel.spec().planned_trial is None and panel.spec().dataset_role == "practice"
        panel.mode_combo.setCurrentIndex(0)
        panel.tick()
        assert c.pending.mode == AUTOMATIC and shown(panel) == c.pending.requested
        edit()
        panel.session_edit.setText("S02")
        panel.session_edit.editingFinished.emit()
        assert c.pending.edited == []
        edit()
        panel.set_output_root(tmp_path / "elsewhere")
        assert c.pending is None and c.planned_trial() is None
        assert wait_until(lambda: not c.loading, 5, qapp) and c.pending is None
        panel.tick()
        assert not any(cb.isEnabled() for cb in panel._sel_combos.values())
    finally:
        src.disconnect_device()
        win.close()
