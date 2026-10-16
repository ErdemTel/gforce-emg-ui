"""Stiffness-collection workflow on the real Recorder: simulated end-to-end
trials, saved files, aborts / failures, resume, and MainWindow integration.

Simulation verifies software behaviour only (not EMG separability, hardware
timing or biological validity). Durations are shortened to keep tests fast.
"""

import csv
import json
import re
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import make_emg, wait_until
from gforce_emg_ui.collection import controller as controller_mod
from gforce_emg_ui.collection import storage as storage_mod
from gforce_emg_ui.collection.controller import (IDLE, REVIEW, RUNNING, CollectionController,
                                                 TrialSpec)
from gforce_emg_ui.collection.plan import attempt_category, training_candidate
from gforce_emg_ui.collection.protocol import HOLD, ProtocolSettings
from gforce_emg_ui.hub import DataHub
from gforce_emg_ui import recorder as recorder_mod
from gforce_emg_ui.recorder import HEADERS, SAVED, Recorder
from gforce_emg_ui.sources import STREAMING, SimFaults, SimSource

FAST = dict(prepare_s=0.15, settle_s=0.15, hold_s=0.3, rest_s=0.2, readiness_s=0.2)
MARKERS_DONE = ["PREPARE_START", "SETTLE_START", "HOLD_START", "HOLD_END", "REST_START",
                "REST_END", "TRIAL_END"]


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def header(path):
    with open(path) as f:
        return f.readline().strip().split(",")


def drive(c, cond, app, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        c.tick()
        if cond():
            return True
        time.sleep(0.005)
    return cond()


def make_rig(qapp, tmp_path, source=None, **settings):
    hub = DataHub()
    src = source(hub) if source else SimSource(hub, seed=3, contact="good")
    rec = Recorder(hub, tmp_path / "manual")
    c = CollectionController(hub, rec, src, settings=ProtocolSettings(**{**FAST, **settings}))
    root = tmp_path / "stiffness_collection"
    c.open_participant(root, "P01")
    if isinstance(src, SimSource):
        src.connect_device(src.ADDRESS)
    assert drive(c, lambda: not c.loading, qapp)
    return SimpleNamespace(hub=hub, src=src, rec=rec, c=c, root=root, app=qapp)


def close_rig(r):
    if r.c.state == RUNNING:
        r.c.abort("user_abort")
    drive(r.c, lambda: not r.c.busy, r.app)
    if isinstance(r.src, SimSource):
        r.src.disconnect_device()
    r.c.io.shutdown()


@pytest.fixture
def sim(qapp, tmp_path):
    r = make_rig(qapp, tmp_path)
    yield r
    close_rig(r)


def make_plan(r, seed=11):
    assert r.c.create_plan(seed) is not None
    assert drive(r.c, lambda: r.c.plan is not None, r.app), r.c.error


def plan_spec(r, session="S01"):
    c, t = r.c, r.c.planned_trial()
    return TrialSpec(participant_id="P01", session_id=session, object_id=t["object_id"],
                     object_name=c.object_name(t["object_id"]), posture_id=t["posture_id"],
                     target_label=t["target_label"], motion_condition=c.plan.motion_of(t),
                     dataset_role="development", output_root=r.root, planned_trial=t,
                     plan_id=c.plan.plan_id)


def manual_spec(r, **kw):
    base = dict(participant_id="P01", session_id="S01", object_id="obj02",
                object_name="Object 2", posture_id="grasp", target_label="medium",
                dataset_role="practice", output_root=r.root, trial_number=1)
    return TrialSpec(**{**base, **kw})


def start(r, spec):
    assert drive(r.c, lambda: r.c.readiness is not None and r.c.readiness.ready, r.app), \
        r.c.readiness.blockers
    assert r.c.start_trial(spec), r.c.error


def run_trial(r, spec):
    start(r, spec)
    folder = r.c.session.folder
    assert drive(r.c, lambda: r.c.state == REVIEW, r.app), (r.c.state, r.c.error)
    return folder


def review(r, decision, reasons=()):
    assert r.c.submit_review(decision, reasons), r.c.error
    assert drive(r.c, lambda: r.c.state == IDLE, r.app), r.c.error


def sidecar(folder):
    return json.loads((folder / "trial.json").read_text())


# -- end to end ------------------------------------------------------------------

def test_simulated_end_to_end_trial(sim):
    r = sim
    make_plan(r)
    planned = r.c.planned_trial()
    tid = planned["trial_id"]
    assert r.c.selection["eligible"] == 200                    # all objects ticked by default
    folder = run_trial(r, plan_spec(r))
    c = r.c
    assert c.metadata_ok and c.message.startswith("Saved") and not c.error
    assert c.acceptable and c.stop_requests == 1
    # folder: <stamp>_<object>_<target>_s<val>_<trial>_a<NN>_<uid> under simulated/P01/S01
    assert folder.parent == r.root / "simulated" / "P01" / "S01"
    assert not (r.root / "P01").exists()                    # nothing saved as real data
    s = {"low": "s0p0", "medium": "s0p5", "high": "s1p0"}[planned["target_label"]]
    assert re.fullmatch(rf"\d{{8}}_\d{{6}}_{planned['object_id']}_{planned['target_label']}_"
                        rf"{s}_{tid}_a01_[0-9a-f]{{8}}", folder.name)
    assert {p.name for p in folder.iterdir()} == {"emg.csv", "imu.csv", "contact.csv",
                                                 "markers.csv", "session.json", "trial.json"}
    for name in ("emg", "imu", "contact", "markers"):        # raw schemas unchanged
        assert header(folder / f"{name}.csv") == HEADERS[name]
    marks = rows(folder / "markers.csv")
    assert [m["label"] for m in marks] == MARKERS_DONE       # each transition once, in order
    emg = rows(folder / "emg.csv")
    assert len(emg) > 100 and all(row["segment"] == "1" for row in emg)
    session = json.loads((folder / "session.json").read_text())
    assert session["state"] == SAVED
    assert session["stiffness_collection"]["attempt_id"] in folder.name
    for key in ("session_id", "device", "emg_rate", "imu_rate", "onboard_filters", "counts"):
        assert key in session

    d = sidecar(folder)
    assert d["schema"] == "gforce_emg_ui.stiffness_trial" and d["schema_version"] == 2
    assert d["status"] == "finalized" and d["review"]["status"] == "pending"
    assert d["is_simulation"] is True and d["data_source"] == "simulation"
    assert d["plan_id"] == c.plan.plan_id and d["trial_id"] == tid
    assert d["selection"]["rule"].startswith("random-available")
    assert d["selection"]["available_objects"] == ["obj01", "obj02", "obj03", "obj04", "obj05"]
    assert d["attempt_number"] == 1 and d["attempt_id"] in folder.name
    assert d["participant_id"] == "P01" and d["collection_session_id"] == "S01"
    assert d["dataset_role"] == "development"
    lab = d["label"]
    assert lab["target_label"] == planned["target_label"]
    assert lab["label_source"] == "instructed_effort"
    assert (lab["target_class_id"], lab["target_s"]) == {
        "low": (0, 0.0), "medium": (1, 0.5), "high": (2, 1.0)}[planned["target_label"]]
    assert lab["participant_instructions"]["HOLD"].startswith("Maintain")
    assert d["condition"]["posture_id"] == planned["posture_id"]
    assert d["outcome"] == {"protocol": "completed", "abort_reason": None,
                            "abort_reason_text": None, "recorder": "saved",
                            "recorder_error": "", "metadata": "ok"}
    assert d["protocol"]["planned_durations_s"] == {"PREPARE": 0.15, "SETTLE": 0.15,
                                                    "HOLD": 0.3, "REST": 0.2}
    assert d["protocol"]["settings"]["readiness_s"] == 0.2
    assert [p["phase"] for p in d["timing"]["phases"]] == ["PREPARE", "SETTLE", "HOLD", "REST"]
    assert all(p["completed"] for p in d["timing"]["phases"])
    assert all("target" not in k for p in d["timing"]["phases"] for k in p)  # phase != target
    # markers.csv and trial.json share the same host timestamps
    ev = d["timing"]["events"]
    assert [e["name"] for e in ev] == MARKERS_DONE
    for e, m in zip(ev, marks):
        assert f"{e['host_mono']:.6f}" == m["host_mono"]
    cand = d["training_candidate"]
    assert cand["phase"] == "HOLD" and cand["quality_status"] == "clean" and cand["eligible"]
    assert f"{cand['start_host_mono']:.6f}" == marks[2]["host_mono"]       # HOLD_START
    assert f"{cand['end_host_mono']:.6f}" == marks[3]["host_mono"]         # HOLD_END
    assert d["readiness"]["initial_snapshot"]["ready"] is True
    assert set(d["readiness"]["initial_snapshot"]["contact_levels"].values()) == {"good"}
    assert d["quality"]["hold"]["lost_samples"] == 0 and d["quality"]["hold"]["batches"] > 0
    assert d["acquisition"]["emg_units"].startswith("unknown")
    assert d["software"]["git"].keys() == {"revision", "dirty"}

    review(r, "accept")
    d = sidecar(folder)
    assert d["status"] == "reviewed" and d["review"]["status"] == "accepted"
    assert c.progress.accepted == 1 and c.progress.remaining == 199
    nxt = c.planned_trial()["trial_id"]
    assert nxt != tid and nxt not in c.progress.completed


def test_only_hold_is_selected_and_rest_is_never_labelled_low(sim):
    r = sim
    folder = run_trial(r, manual_spec(r, target_label="high"))
    review(r, "accept")
    d = sidecar(folder)
    emg = rows(folder / "emg.csv")
    phases = {p["phase"]: (p["start_host_mono"], p["end_host_mono"])
              for p in d["timing"]["phases"]}

    def phase_of(t):
        return next((ph for ph, (a, b) in phases.items() if a <= t < b), None)
    by_phase = {}
    for row in emg:
        by_phase.setdefault(phase_of(float(row["host_mono"])), []).append(row)
    assert by_phase.get("REST") and by_phase.get(HOLD)
    cand = training_candidate(d)
    assert cand["target_label"] == "high" and cand["target_s"] == 1.0
    selected = [row for row in emg
                if cand["start_host_mono"] <= float(row["host_mono"]) <= cand["end_host_mono"]]
    assert selected and all(phase_of(float(row["host_mono"])) == HOLD for row in selected)
    # REST rows are recorded (raw data untouched) but never part of the candidate,
    # and nothing labels them low / s=0
    assert all(float(row["host_mono"]) > cand["end_host_mono"] for row in by_phase["REST"])
    assert "low" not in json.dumps(d["training_candidate"])


def test_repeat_deletes_attempt_and_planned_trial_counts_once(sim):
    r = sim
    make_plan(r)
    tid = r.c.planned_trial()["trial_id"]
    first = run_trial(r, plan_spec(r))
    first_id = r.c.sidecar["attempt_id"]
    review(r, "repeat", ["hand_movement"])
    assert not first.exists() and first_id not in r.c.records    # only accepted are kept
    assert r.c.planned_trial()["trial_id"] == tid             # same planned trial again
    assert r.c.selection["repeat"] is True
    second = run_trial(r, plan_spec(r))
    assert second != first and second.exists()
    d2 = sidecar(second)
    assert d2["trial_id"] == tid and d2["attempt_id"] != first_id
    assert d2["attempt_number"] == 1 and f"_{tid}_a01_" in second.name
    review(r, "accept")
    assert second.exists() and sidecar(second)["review"]["status"] == "accepted"
    p = r.c.progress
    assert (p.accepted, p.attempted, p.rejected, p.remaining) == (1, 1, 0, 199)
    assert p.completed == {tid: d2["attempt_id"]}
    done = plan_spec(r)
    done.planned_trial = r.c.plan.trial(tid)
    done.object_id, done.target_label = (done.planned_trial["object_id"],
                                         done.planned_trial["target_label"])
    done.motion_condition = done.planned_trial["motion_condition"]
    assert any("already completed" in b for b in r.c.start_blockers(done))


def test_reject_moves_on_and_rejected_trial_stays_pending(sim):
    r = sim
    make_plan(r)
    tid = r.c.planned_trial()["trial_id"]
    folder = run_trial(r, plan_spec(r))
    review(r, "reject", ["wrong_effort"])
    assert not folder.exists()
    assert r.c.planned_trial()["trial_id"] != tid              # moves on, comes back later
    assert r.c.message == "Rejected – deleted"
    p = r.c.progress
    assert (p.accepted, p.attempted, p.rejected, p.remaining) == (0, 0, 0, 200)
    assert tid not in p.completed
    # consecutive rejects keep drawing new trials (the attempt count on disk does not change)
    keys = {r.c.selection["key"]}
    for _ in range(2):
        run_trial(r, plan_spec(r))
        review(r, "reject")
        keys.add(r.c.selection["key"])
    assert keys == {"0/d1", "0/d2", "0/d3"}


def test_flagged_hold_requires_repeat(qapp, tmp_path):
    r = make_rig(qapp, tmp_path, source=lambda hub: SimSource(
        hub, seed=5, contact="good", faults=SimFaults(lost_prob=0.05)))
    try:
        folder = run_trial(r, manual_spec(r))
        d = sidecar(folder)
        cand = d["training_candidate"]
        assert cand["quality_status"] == "flagged"
        assert any(f.startswith("lost_samples") for f in cand["quality_flags"])
        assert not r.c.acceptable and any("HOLD quality" in b for b in r.c.accept_blockers)
        assert r.c.submit_review("accept") is False and "Cannot accept" in r.c.error
        lost = [row for row in rows(folder / "emg.csv") if row["lost"] == "1"]
        assert lost and all(row["ch1"] == "" for row in lost)          # kept, not filled
        review(r, "repeat", ["contact_problem"])
        assert not folder.exists()
    finally:
        close_rig(r)


def test_link_loss_aborts_keeps_partial_data_and_readiness_restarts(qapp, tmp_path):
    r = make_rig(qapp, tmp_path, source=lambda hub: SimSource(
        hub, seed=3, contact="good", reconnect_delay=0.3), hold_s=2.0)
    try:
        start(r, manual_spec(r))
        folder = r.c.session.folder
        assert drive(r.c, lambda: r.c.machine.phase == HOLD, r.app)
        r.src.inject_disconnect()
        assert drive(r.c, lambda: r.c.state == REVIEW, r.app)
        d = sidecar(folder)
        assert d["outcome"]["protocol"] == "aborted"
        assert d["outcome"]["abort_reason"] == "link_lost"
        assert d["outcome"]["recorder"] == "saved"                # partial data kept
        labels = [m["label"] for m in rows(folder / "markers.csv")]
        assert labels[-2:] == ["ABORT", "TRIAL_END"] and "HOLD_START" in labels
        assert "HOLD_END" not in labels and "REST_START" not in labels
        assert len(rows(folder / "emg.csv")) > 0
        assert d["training_candidate"]["eligible"] is False
        assert d["training_candidate"]["quality_status"] == "not_available"
        assert not r.c.acceptable and r.c.submit_review("accept") is False
        assert attempt_category(d) == "invalid"
        assert not r.c.message.startswith("Saved")
        # after the reconnect the next attempt must pass readiness again
        assert drive(r.c, lambda: r.src.state == STREAMING and r.hub.emg_segment == 2, r.app)
        review(r, "reject", ["other"])
        assert not folder.exists()                            # aborted attempts are deleted
        r.c.tick()
        assert not r.c.readiness.ready
        assert drive(r.c, lambda: r.c.readiness.ready, r.app)
        assert r.c.readiness.segment == 2
    finally:
        close_rig(r)


class FakeSource:
    """Data pushed by the test; only the attributes the controller uses."""
    is_simulation = True
    target_emg_rate = 1000

    def __init__(self, hub):
        self.hub = hub
        self.state = STREAMING
        self.paused = threading.Event()
        self.stream_id = 1
        self.imp = None                     # impedance override (Ohm per channel)
        self._stop = threading.Event()
        self._index = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def backend_name(self):
        return "fake"

    def _run(self):
        last_stream = self.stream_id
        while not self._stop.is_set():
            if not self.paused.is_set():
                if self.stream_id != last_stream:
                    last_stream, self._index = self.stream_id, 0
                self.hub.push_emg(make_emg(self._index, stream_id=self.stream_id, imp=self.imp))
                self._index += 16
            time.sleep(0.016)

    def stop(self):
        self._stop.set()


@pytest.mark.parametrize("fault,reason", [("pause", "stale_stream"),
                                          ("new_stream", "segment_reset")])
def test_stale_stream_and_segment_reset_abort(qapp, tmp_path, fault, reason):
    r = make_rig(qapp, tmp_path, source=FakeSource, hold_s=2.0)
    try:
        start(r, manual_spec(r))
        folder = r.c.session.folder
        assert drive(r.c, lambda: r.c.machine.phase == HOLD, r.app)
        if fault == "pause":
            r.src.paused.set()
        else:
            r.src.stream_id = 2
        assert drive(r.c, lambda: r.c.state == REVIEW, r.app)
        d = sidecar(folder)
        assert d["outcome"]["abort_reason"] == reason
        assert d["outcome"]["recorder"] == "saved" and not r.c.acceptable
        if fault == "new_stream":
            assert 2 in {int(row["segment"]) for row in rows(folder / "emg.csv")}
    finally:
        r.src.stop()
        close_rig(r)


def test_orange_contact_trial_is_ready_clean_and_acceptable(qapp, tmp_path):
    r = make_rig(qapp, tmp_path, source=FakeSource)
    try:
        r.src.imp = [250e3, 700e3, 250e3, 250e3, 250e3, 950e3, 250e3, 250e3]   # 2 orange
        folder = run_trial(r, manual_spec(r))
        d = sidecar(folder)
        snap = d["readiness"]["initial_snapshot"]["contact_levels"]
        assert snap["ch2"] == snap["ch6"] == "marginal" and snap["ch1"] == "good"
        hold = d["quality"]["hold"]
        assert hold["contact_marginal"] == ["ch2", "ch6"] and hold["contact_not_ok"] == []
        assert d["training_candidate"]["quality_status"] == "clean" and r.c.acceptable
        assert "MARGINAL" in d["readiness"]["settings"]["contact_rule"]
        review(r, "accept")
        r.src.imp = [250e3] * 7 + [1.5e6]                                    # red -> blocked
        assert drive(r.c, lambda: any("CH8 poor" in b for b in r.c.readiness.blockers), r.app)
        assert not r.c.readiness.ready
    finally:
        r.src.stop()
        close_rig(r)


def test_recorder_error_is_never_reported_as_saved(sim, monkeypatch):
    r = sim

    def boom(self, f, block):
        raise OSError("disk full (injected)")
    monkeypatch.setattr(recorder_mod.RecordingSession, "_write_emg", boom)
    start(r, manual_spec(r))
    folder = r.c.session.folder
    assert drive(r.c, lambda: r.c.state == REVIEW, r.app)
    d = sidecar(folder)
    assert d["outcome"]["recorder"] == "error" and "disk full" in d["outcome"]["recorder_error"]
    assert d["outcome"]["abort_reason"] == "recorder_error"
    assert "recording failed" in r.c.error and not r.c.message.startswith("Saved")
    assert not r.c.acceptable and attempt_category(d) == "invalid"


def test_metadata_write_failure_is_never_reported_as_saved(sim, monkeypatch):
    r = sim
    real = controller_mod.atomic_write_text

    def failing(path, text):
        if path.name == "trial.json":
            raise OSError("metadata disk full (injected)")
        return real(path, text)
    monkeypatch.setattr(controller_mod, "atomic_write_text", failing)
    start(r, manual_spec(r))
    folder = r.c.session.folder
    assert drive(r.c, lambda: r.c.state == REVIEW, r.app)
    assert not r.c.metadata_ok and "trial.json could not be written" in r.c.error
    assert not r.c.message.startswith("Saved") and not r.c.acceptable
    assert r.c.sidecar["outcome"]["abort_reason"] == "metadata_error"
    assert not (folder / "trial.json").exists()
    # raw files kept: everything the recorder received was written (the abort can come
    # before the first batch, so the attempt may legitimately hold zero samples)
    assert header(folder / "emg.csv") == HEADERS["emg"]
    counts = json.loads((folder / "session.json").read_text())["counts"]["emg"]
    assert counts["written"] == counts["received"] == len(rows(folder / "emg.csv"))
    assert r.c.sidecar["attempt_id"] not in r.c.records


def test_final_metadata_failure_leaves_attempt_invalid_until_rejected(sim, monkeypatch):
    r = sim
    real = controller_mod.atomic_write_text

    def failing(path, text):
        if '"status": "finalized"' in text:
            raise OSError("injected")
        return real(path, text)
    monkeypatch.setattr(controller_mod, "atomic_write_text", failing)
    make_plan(r)
    folder = run_trial(r, plan_spec(r))
    assert not r.c.metadata_ok and not r.c.acceptable
    d = sidecar(folder)
    assert d["status"] == "recording" and attempt_category(d) == "invalid"
    assert r.c.progress.invalid == 1
    review(r, "reject", ["other"])
    assert not folder.exists()
    assert r.c.progress.accepted == 0 and r.c.progress.invalid == 0


def test_failed_delete_keeps_a_rejected_attempt_and_moves_on(sim, monkeypatch):
    r = sim

    def failing(path):
        raise OSError("injected")
    monkeypatch.setattr(storage_mod.shutil, "rmtree", failing)
    folder = run_trial(r, manual_spec(r))
    review(r, "reject", ["other"])
    assert "could not delete" in r.c.error and "injected" in r.c.error
    d = sidecar(folder)
    assert d["review"]["status"] == "rejected" and attempt_category(d) == "rejected"
    assert r.c.records[d["attempt_id"]]["review"]["status"] == "rejected"


def test_delete_never_leaves_the_output_root(tmp_path):
    root = tmp_path / "root"
    (root / "P01").mkdir(parents=True)
    with pytest.raises(ValueError):
        storage_mod.discard_attempt(root, tmp_path, "{}")
    with pytest.raises(ValueError):
        storage_mod.discard_attempt(root, root, "{}")
    assert (root / "P01").exists()


def test_crash_leaves_interrupted_attempt_that_never_counts(sim):
    r = sim
    make_plan(r)
    start(r, plan_spec(r))
    attempt = r.c.sidecar["attempt_id"]
    assert drive(r.c, lambda: attempt in r.c.records, r.app)     # initial trial.json persisted
    # a fresh app instance (as after a crash) only sees what is on disk
    c2 = CollectionController(r.hub, Recorder(r.hub, r.root / "x"), r.src)
    c2.open_participant(r.root, "P01")
    assert drive(c2, lambda: not c2.loading and c2.plan is not None, r.app)
    assert c2.records[attempt]["status"] == "recording"
    assert attempt_category(c2.records[attempt]) == "invalid"
    assert (c2.progress.accepted, c2.progress.invalid) == (0, 1)
    labels = [m["label"] for m in rows(r.c.session.folder / "markers.csv")] \
        if (r.c.session.folder / "markers.csv").exists() else []
    assert "TRIAL_END" not in labels
    c2.io.shutdown()


def test_resume_plan_across_sessions_from_disk(sim):
    r = sim
    make_plan(r, seed=21)
    plan_id, order = r.c.plan.plan_id, [dict(t) for t in r.c.plan.trials]
    first = r.c.planned_trial()["trial_id"]
    run_trial(r, plan_spec(r, "S01"))
    review(r, "accept")
    expected_next = r.c.planned_trial()["trial_id"]
    c2 = CollectionController(r.hub, r.rec, r.src, settings=ProtocolSettings(**FAST))
    c2.open_participant(r.root, "P01")
    assert drive(c2, lambda: not c2.loading and c2.plan is not None, r.app)
    assert c2.plan.plan_id == plan_id and c2.plan.trials == order       # not reshuffled
    # same next trial after a restart: the pick depends only on what is on disk
    assert c2.progress.accepted == 1 and c2.planned_trial()["trial_id"] == expected_next
    r2 = SimpleNamespace(**{**vars(r), "c": c2})
    folder = run_trial(r2, plan_spec(r2, "S02"))
    review(r2, "accept")
    assert folder.parent.name == "S02" and sidecar(folder)["collection_session_id"] == "S02"
    assert c2.progress.accepted == 2 and set(c2.progress.completed) == {first, expected_next}
    plans = list((r.root / "simulated" / "P01").glob("plan_*.json"))
    assert len(plans) == 1
    c2.io.shutdown()


def test_practice_trials_never_count_toward_the_plan(sim):
    r = sim
    make_plan(r)
    folder = run_trial(r, manual_spec(r, dataset_role="practice"))
    assert re.match(r"\d{8}_\d{6}_practice_obj02_medium_s0p5_m001_a01_", folder.name)
    review(r, "accept")
    assert r.c.progress.accepted == 0 and r.c.progress.attempted == 0
    assert r.c.role_counts() == {"practice": {"accepted": 1}}
    dev = manual_spec(r, dataset_role="development")
    assert any("come from the plan" in b for b in r.c.start_blockers(dev))


def test_attempt_folders_are_unique_and_never_overwritten(sim):
    r = sim
    folders = []
    for _ in range(2):
        folders.append(run_trial(r, manual_spec(r)))
        review(r, "accept")
    assert folders[0] != folders[1] and all(f.exists() for f in folders)
    ids = {sidecar(f)["attempt_id"] for f in folders}
    assert len(ids) == 2
    assert [sidecar(f)["attempt_number"] for f in folders] == [1, 2]


def test_invalid_ids_block_start_and_no_folder_is_created(sim):
    r = sim
    for bad in (dict(participant_id="../evil"), dict(session_id="a/b"),
                dict(object_id="Obj 1"), dict(target_label="max")):
        spec = manual_spec(r, **bad)
        assert r.c.start_blockers(spec)
        assert r.c.start_trial(spec) is False
    assert not (r.root / "simulated" / "P01").exists()


def test_recorder_extensions_are_backward_compatible(tmp_path):
    hub = DataHub()
    rec = Recorder(hub, tmp_path / "default")
    s = rec.start("x", root=tmp_path / "other")
    assert s.folder.parent == tmp_path / "other" and rec.root == tmp_path / "default"
    hub.push_emg(make_emg(0))
    s.add_marker("explicit", host_time=1234.5, host_mono=99.25)
    s.add_marker("now")
    s.request_stop()
    assert s.wait(5) and s.state == SAVED
    m = rows(s.folder / "markers.csv")
    assert m[0]["host_time"] == "1234.500000" and m[0]["host_mono"] == "99.250000"
    assert m[0]["emg_sample_index"] == "15" and m[1]["label"] == "now"


# -- MainWindow integration ---------------------------------------------------------

def make_window(qapp, tmp_path):
    from gforce_emg_ui.ui.main_window import MainWindow
    hub = DataHub()
    src = SimSource(hub, seed=4, contact="good")
    win = MainWindow(hub, src, tmp_path)
    panel = win.collection_panel
    for key, value in dict(prepare_s=0.2, settle_s=0.2, hold_s=0.3, rest_s=0.2,
                           readiness_s=0.2).items():
        panel.dur_spins[key].setValue(value)
    panel.mode_combo.setCurrentIndex(1)                       # manual selection
    panel.participant_edit.setText("P01")
    panel.session_edit.setText("S01")
    panel._reopen_participant()
    src.connect_device(src.ADDRESS)
    assert wait_until(lambda: not win.collection.loading and hub.emg.total > 0, 5, qapp)
    return win, src, panel


def start_panel_trial(qapp, win, panel):
    win.record_tabs.setCurrentIndex(1)
    assert wait_until(lambda: panel.btn_start.isEnabled(), 5, qapp), panel.ready_label.text()
    panel.btn_start.click()
    assert win.collection.owns_recorder and win.collection.state == RUNNING


def test_manual_recording_blocks_collection_and_keeps_its_format(qapp, tmp_path):
    win, src, panel = make_window(qapp, tmp_path)
    try:
        win._toggle_recording()                               # manual recording
        assert win.recorder.active and not win.collection.owns_recorder
        assert wait_until(lambda: "Recorder busy" in panel.ready_label.text(), 3, qapp)
        assert not panel.btn_start.isEnabled()
        assert win.collection.start_trial(panel.spec()) is False
        win._toggle_recording()
        session = win.recorder.session
        assert wait_until(lambda: session.done, 5, qapp) and session.state == SAVED
        assert {p.name for p in session.folder.iterdir()} == {
            "emg.csv", "imu.csv", "contact.csv", "markers.csv", "session.json"}
        for name in ("emg", "imu", "contact", "markers"):
            assert header(session.folder / f"{name}.csv") == HEADERS[name]
        meta = json.loads((session.folder / "session.json").read_text())
        assert "stiffness_collection" not in meta
        assert session.folder.parent == tmp_path
    finally:
        src.disconnect_device()
        win.close()


def test_collection_trial_owns_recorder_across_tab_switches(qapp, tmp_path):
    win, src, panel = make_window(qapp, tmp_path)
    try:
        start_panel_trial(qapp, win, panel)
        c = win.collection
        session, machine = c.session, c.machine
        assert win.recorder is c.recorder and win.recorder.session is session
        assert win.btn_record.text().startswith("■ Stop (aborts")
        assert not win.btn_marker.isEnabled()
        win._add_marker()                                     # Ctrl+M: ignored during a trial
        assert src.settings_locked
        for i in (0, 1, 0):                                   # switching tabs changes nothing
            win.record_tabs.setCurrentIndex(i)
            qapp.processEvents()
            assert c.session is session and c.machine is machine and c.state == RUNNING
            assert panel.timer.isActive()
            assert not panel.summary_label.isHidden()       # status line shown from either tab
            assert "Stiffness trial" in panel.summary_label.text()
        assert wait_until(lambda: c.state == REVIEW, 10, qapp)
        labels = [m["label"] for m in rows(session.folder / "markers.csv")]
        assert labels == MARKERS_DONE                         # no manual marker slipped in
        assert c.stop_requests == 1
        assert wait_until(lambda: not src.settings_locked, 5, qapp)
        panel.btn_accept.click()
        assert wait_until(lambda: c.state == IDLE, 5, qapp)
        assert panel.trial_spin.value() == 2                  # manual trial number advanced
    finally:
        src.disconnect_device()
        win.close()


@pytest.mark.parametrize("how,reason", [("manual_stop", "manual_stop"),
                                        ("disconnect", "user_disconnect")])
def test_manual_stop_and_disconnect_abort_the_trial(qapp, tmp_path, how, reason):
    win, src, panel = make_window(qapp, tmp_path)
    try:
        start_panel_trial(qapp, win, panel)
        c = win.collection
        folder = c.session.folder
        if how == "manual_stop":
            win._toggle_recording()
        else:
            win._disconnect()
        assert wait_until(lambda: c.state == REVIEW, 5, qapp)
        d = sidecar(folder)
        assert d["outcome"]["abort_reason"] == reason and d["outcome"]["recorder"] == "saved"
        assert c.stop_requests == 1 and not c.acceptable
        assert "ABORTED" in panel.phase_label.text()
    finally:
        src.disconnect_device()
        win.close()


def test_app_close_during_trial_finalizes_then_deletes_the_attempt(qapp, tmp_path):
    win, src, panel = make_window(qapp, tmp_path)
    win.show()
    start_panel_trial(qapp, win, panel)
    c = win.collection
    folder = c.session.folder
    win.close()                                               # ignored until finalized
    assert win.isVisible()
    assert wait_until(lambda: not win.isVisible(), 10, qapp)
    sc = c.sidecar
    assert sc["outcome"]["abort_reason"] == "app_close" and sc["outcome"]["recorder"] == "saved"
    assert sc["review"]["status"] == "rejected" and not folder.exists()
    src.disconnect_device()


def test_app_close_before_review_deletes_the_attempt(qapp, tmp_path):
    win, src, panel = make_window(qapp, tmp_path)
    win.show()
    start_panel_trial(qapp, win, panel)
    c = win.collection
    folder = c.session.folder
    assert wait_until(lambda: c.state == REVIEW, 10, qapp)
    assert c.sidecar["outcome"]["protocol"] == "completed" and folder.exists()
    win.close()                                               # waits for the delete
    assert wait_until(lambda: not win.isVisible(), 10, qapp)
    assert not folder.exists()
    src.disconnect_device()


def _wheel(w, dy=-120):
    from PyQt5 import QtCore, QtGui, QtWidgets
    c = QtCore.QPointF(w.rect().center())
    QtWidgets.QApplication.sendEvent(w, QtGui.QWheelEvent(
        c, QtCore.QPointF(w.mapToGlobal(c.toPoint())), QtCore.QPoint(0, 0), QtCore.QPoint(0, dy),
        QtCore.Qt.NoButton, QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False))


def test_wheel_scrolls_the_panel_instead_of_switching_tabs_or_settings(qapp, tmp_path):
    """Scrolling the side panel over the Recording tab bar used to flip Manual /
    Stiffness Collection (panel height jumping ~870 px: the 'blinking' scroll bar),
    and over a combo / spin box it silently changed target, movement or durations."""
    from PyQt5 import QtCore
    win, src, panel = make_window(qapp, tmp_path)
    try:
        win.resize(1500, 950)
        win.show()
        bar = win.centralWidget().widget(0).verticalScrollBar()
        qapp.processEvents()
        assert bar.maximum() > 0
        _wheel(win.record_tabs)
        assert win.record_tabs.currentIndex() == 0 and bar.value() > 0      # scrolled
        win.record_tabs.setCurrentIndex(1)
        panel.tick()
        panel.btn_advanced.setChecked(True)
        qapp.processEvents()
        for combo in (panel.target_combo, panel.motion_combo, panel.object_combo,
                      panel.role_combo, panel.mode_combo):
            assert combo.isEnabled() and not combo.hasFocus()
            assert combo.focusPolicy() == QtCore.Qt.StrongFocus
            before, scroll = combo.currentIndex(), bar.value()
            _wheel(combo, -120 if scroll < bar.maximum() else 120)
            assert combo.currentIndex() == before and bar.value() != scroll
        hold = panel.dur_spins["hold_s"]
        before = hold.value()
        _wheel(hold, 120)
        assert hold.value() == before and win.collection.settings.hold_s == before
        assert win.record_tabs.currentIndex() == 1
    finally:
        src.disconnect_device()
        win.close()


def test_side_panel_is_not_relaid_out_on_every_tick(qapp, tmp_path):
    """Per-tick text changes (readiness counter, countdowns) re-laid out and repainted
    the whole side panel ~12-20x/s: visible flicker on the real display."""
    from PyQt5 import QtCore
    win, src, panel = make_window(qapp, tmp_path)
    side = win.centralWidget().widget(0).widget()
    layouts = []

    class Spy(QtCore.QObject):
        def eventFilter(self, obj, ev):
            if obj is side and ev.type() == QtCore.QEvent.LayoutRequest:
                layouts.append(1)
            return False
    spy = Spy()
    qapp.installEventFilter(spy)
    try:
        win.show()
        win.record_tabs.setCurrentIndex(1)
        assert wait_until(lambda: panel.btn_start.isEnabled(), 5, qapp)
        layouts.clear()
        wait_until(lambda: False, 1.0, qapp)                  # ~20 ticks, ready + idle
        assert len(layouts) <= 3, len(layouts)               # was ~12/s
        for key, value in dict(prepare_s=0.5, settle_s=0.5, hold_s=1.5).items():
            panel.dur_spins[key].setValue(value)
        panel.btn_start.click()
        c = win.collection
        assert wait_until(lambda: c.machine.phase == HOLD, 5, qapp)
        texts = set()
        layouts.clear()
        wait_until(lambda: texts.add(panel.phase_label.text()) or False, 1.0, qapp)
        assert len(texts) >= 5                               # countdown still updates ...
        assert len(layouts) <= 3, len(layouts)               # ... without re-layouts
        assert "HOLD" in panel.phase_label.text() and "HOLD" in panel.summary_label.text()
    finally:
        qapp.removeEventFilter(spy)
        if win.collection.state == RUNNING:
            win.collection.abort("user_abort")
        wait_until(lambda: not win.collection.busy, 5, qapp)
        src.disconnect_device()
        win.close()


def test_contact_chips_restyle_only_on_change(qapp):
    from PyQt5 import QtCore
    from gforce_emg_ui.ui.main_window import ContactStrip
    strip = ContactStrip()
    strip.show()                            # unshown widgets get no style events at all
    imp = np.full(8, 250e3)
    strip.update_contact(imp)
    qapp.processEvents()
    restyles = []

    class Spy(QtCore.QObject):
        def eventFilter(self, obj, ev):
            if obj in strip.chips and ev.type() == QtCore.QEvent.StyleChange:
                restyles.append(obj)
            return False
    spy = Spy()
    qapp.installEventFilter(spy)
    try:
        for _ in range(5):
            strip.update_contact(imp)
            qapp.processEvents()
        assert restyles == []
        imp[2] = 2e6
        strip.update_contact(imp)
        assert restyles == [strip.chips[2]] and "2000 kΩ" in strip.chips[2].text()
    finally:
        qapp.removeEventFilter(spy)
        strip.close()


def test_next_trial_is_random_among_available_objects_plus_empty_hand(sim):
    r = sim
    make_plan(r, seed=8)
    c = r.c
    assert c.set_available_objects({"obj02"})
    seen = set()
    for _ in range(3):
        t = c.planned_trial()
        assert t["object_id"] in {"obj02", "empty_hand"}
        assert c.selection["eligible"] == 32 + 40 - c.progress.accepted
        seen.add(t["trial_id"])
        folder = run_trial(r, plan_spec(r))
        d = sidecar(folder)
        assert d["selection"]["available_objects"] == ["obj02"]
        assert d["selection"]["empty_hand_always_eligible"] is True
        review(r, "accept")
    assert len(seen) == 3 and c.progress.accepted == 3
    # re-ticking the same objects gives the same pick: toggling cannot re-roll it
    c.set_available_objects({"obj01"})
    a = c.planned_trial()["trial_id"]
    c.set_available_objects({"obj01", "obj05"})
    c.set_available_objects({"obj01"})
    assert c.planned_trial()["trial_id"] == a
    # a planned trial whose object is not ticked cannot start
    other = next(t for t in c.plan.trials if t["object_id"] == "obj04"
                 and t["trial_id"] not in c.progress.completed)
    spec = plan_spec(r)
    spec.planned_trial, spec.object_id = other, "obj04"
    spec.target_label, spec.motion_condition = other["target_label"], other["motion_condition"]
    assert "obj04 is not marked available" in c.start_blockers(spec)
    start(r, plan_spec(r))
    assert c.set_available_objects({"obj03"}) is False              # locked during a trial


def test_available_object_checkboxes_in_the_panel(qapp, tmp_path):
    win, src, panel = make_window(qapp, tmp_path)
    try:
        c = win.collection
        assert list(panel.avail_checks) == ["obj01", "obj02", "obj03", "obj04", "obj05"]
        assert all(cb.isChecked() for cb in panel.avail_checks.values())
        assert panel.avail_row.isHidden()                       # manual mode: not shown
        panel.mode_combo.setCurrentIndex(0)
        assert not panel.avail_row.isHidden()
        panel._new_plan(confirm=False)
        assert wait_until(lambda: c.plan is not None, 5, qapp)
        for cb in panel.avail_checks.values():
            cb.setChecked(False)                                # nothing available today
        panel.tick()
        assert c.planned_trial()["object_id"] == "empty_hand"    # empty hand still offered
        assert "random pick of 40 eligible" in panel.next_label.text()
        panel.avail_checks["obj03"].setChecked(True)
        panel.tick()
        assert c.planned_trial()["object_id"] in {"obj03", "empty_hand"}
        assert c.selection["eligible"] == 72
    finally:
        src.disconnect_device()
        win.close()
