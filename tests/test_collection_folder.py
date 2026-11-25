"""Folder mode: choose a folder and stiffness 0 / 1, then the usual protocol,
recorder, trial.json and review. Attempts are saved directly in the folder."""

import json

import pytest

from conftest import wait_until
from gforce_emg_ui.collection.controller import (FOLDER_OBJECT, IDLE, REVIEW, RUNNING,
                                                 TrialSpec, folder_ids)
from gforce_emg_ui.hub import DataHub
from gforce_emg_ui.sources import SimSource
from test_collection_workflow import (FAST, close_rig, drive, make_rig, review, run_trial,
                                      sidecar)


class BandLike(SimSource):
    """Simulator presenting itself as a real band (real-data paths)."""
    is_simulation = False


def folder_spec(folder, target="high", n=1):
    pid, sid = folder_ids(folder)
    return TrialSpec(participant_id=pid, session_id=sid, object_id=FOLDER_OBJECT,
                     object_name="Unspecified", posture_id=FOLDER_OBJECT, target_label=target,
                     dataset_role="development", output_root=folder, trial_number=n,
                     folder=folder)


def open_folder(r, folder):
    assert r.c.open_folder(folder)
    assert drive(r.c, lambda: not r.c.loading, r.app)


@pytest.fixture
def band(qapp, tmp_path):
    r = make_rig(qapp, tmp_path, source=lambda hub: BandLike(hub, seed=3, contact="good"))
    yield r
    close_rig(r)


@pytest.fixture
def sim(qapp, tmp_path):
    r = make_rig(qapp, tmp_path)
    yield r
    close_rig(r)


def test_folder_ids(tmp_path):
    assert folder_ids(tmp_path / "Erdem" / "S3") == ("Erdem", "S3")
    assert folder_ids(tmp_path / "my data" / "day 1!") == ("my_data", "day_1")
    assert folder_ids(tmp_path / "simulated" / "x")[0] == "unknown"       # reserved
    assert len(folder_ids(tmp_path / ("x" * 40))[1]) == 32


def test_development_without_plan_only_in_folder_mode(tmp_path):
    spec = folder_spec(tmp_path / "P" / "S")
    assert spec.problems() == []
    spec.folder = None
    assert any("come from the plan" in p for p in spec.problems())


def test_trial_saved_directly_in_folder(band, tmp_path):
    folder = tmp_path / "new" / "Erdem" / "S3"
    open_folder(band, folder)
    assert band.c.plans == [] and band.c.plan is None
    out = run_trial(band, folder_spec(folder, "low"))
    assert out.parent == folder
    review(band, "accept")
    d = sidecar(out)
    assert (d["participant_id"], d["collection_session_id"]) == ("Erdem", "S3")
    assert (d["trial_id"], d["attempt_number"], d["plan_id"]) == ("m001", 1, None)
    assert d["dataset_role"] == "development" and d["is_simulation"] is False
    assert d["condition"]["object_id"] == FOLDER_OBJECT
    assert d["condition"]["motion_condition"] == "stationary"
    assert (d["label"]["target_label"], d["label"]["target_s"]) == ("low", 0.0)
    assert d["outcome"] == {**d["outcome"], "protocol": "completed", "recorder": "saved"}
    assert d["review"]["status"] == "accepted"
    assert sorted(p.name for p in folder.iterdir()) == [out.name]   # no plan / session dirs
    # reopening the folder finds the accepted attempt
    open_folder(band, folder)
    assert list(band.c.records) == [d["attempt_id"]]


def test_reject_deletes_only_the_attempt(band, tmp_path):
    folder = tmp_path / "f"
    open_folder(band, folder)
    out = run_trial(band, folder_spec(folder))
    review(band, "reject")
    assert not out.exists() and folder.is_dir()
    assert band.c.records == {}


def test_simulation_goes_to_simulated_subfolder(sim, tmp_path):
    folder = tmp_path / "f"
    open_folder(sim, folder)
    out = run_trial(sim, folder_spec(folder))
    review(sim, "accept")
    assert out.parent == folder / "simulated"
    assert sidecar(out)["is_simulation"] is True


def test_folder_ignores_plans_and_session_folders(band, tmp_path):
    folder = tmp_path / "P01"
    open_folder(band, tmp_path / "other")
    run_trial(band, folder_spec(tmp_path / "other"))
    review(band, "accept")
    (folder / "S01").mkdir(parents=True)
    src = next((tmp_path / "other").iterdir())
    src.rename(folder / "S01" / src.name)                  # participant-mode layout
    (folder / "plan_x.json").write_text("{}")
    open_folder(band, folder)
    assert band.c.plans == [] and band.c.records == {} and band.c.unreadable == []
    assert band.c.create_plan(1) is None


def test_start_blocked_when_folder_not_loaded(band, tmp_path):
    open_folder(band, tmp_path / "a")
    assert "Participant data not loaded" in band.c.start_blockers(folder_spec(tmp_path / "b"))


# -- panel ------------------------------------------------------------------------------

def make_window(qapp, tmp_path):
    from gforce_emg_ui.ui.main_window import MainWindow
    hub = DataHub()
    src = BandLike(hub, seed=4, contact="good")
    win = MainWindow(hub, src, tmp_path / "data")
    panel = win.collection_panel
    for key, value in FAST.items():
        panel.dur_spins[key].setValue(value)
    win.record_tabs.setCurrentIndex(1)
    src.connect_device(src.ADDRESS)
    return win, src, panel


def test_panel_folder_mode_is_default_and_simple(qapp, tmp_path):
    win, src, panel = make_window(qapp, tmp_path)
    try:
        assert panel.mode == "folder"
        assert panel.simple_form.isVisibleTo(panel) and not panel.full_form.isVisibleTo(panel)
        assert not panel.seed_edit.isVisibleTo(panel)
        assert "Protocol settings" in panel.btn_advanced.text()
        assert panel.stiff_group.checkedId() == -1           # no default label
        assert wait_until(lambda: win.hub.emg.total > 0, 5, qapp)
        b = panel.start_blockers()
        assert b[:2] == ["Choose a folder", "Choose stiffness 0 or 1"]
        assert not any("Participant" in x or "Target" in x or "plan" in x for x in b)
        assert not panel.btn_start.isEnabled()
        panel.set_folder(tmp_path / "Erdem" / "S9")
        assert wait_until(lambda: panel.start_blockers()[:1] == ["Choose stiffness 0 or 1"],
                          5, qapp), panel.start_blockers()
        panel.stiff_group.button(0).setChecked(True)
        assert wait_until(lambda: panel.btn_start.isEnabled(), 5, qapp), panel.ready_label.text()
    finally:
        src.disconnect_device()
        win.close()


def test_panel_records_stiffness_trials(qapp, tmp_path):
    win, src, panel = make_window(qapp, tmp_path)
    folder = tmp_path / "Erdem" / "S9"
    c = panel.controller
    try:
        panel.set_folder(folder)
        for n, value in enumerate((1, 0), start=1):
            panel.stiff_group.button(value).setChecked(True)
            assert wait_until(lambda: panel.btn_start.isEnabled(), 5, qapp), \
                panel.ready_label.text()
            assert panel.spec().trial_id == f"m{n:03d}"
            panel.btn_start.click()
            assert c.state == RUNNING
            assert not panel.btn_folder.isEnabled()
            assert not any(b.isEnabled() for b in panel.stiff_group.buttons())
            assert wait_until(lambda: c.state == REVIEW, 10, qapp)
            panel.btn_accept.click()
            assert wait_until(lambda: c.state == IDLE, 5, qapp)
        assert wait_until(lambda: "stiffness 0: 1 · stiffness 1: 1"
                          in panel.progress_label.text(), 2, qapp), panel.progress_label.text()
        got = sorted((json.loads(p.read_text())["trial_id"],
                      json.loads(p.read_text())["label"]["target_s"])
                     for p in folder.glob("*/trial.json"))
        assert got == [("m001", 1.0), ("m002", 0.0)]
    finally:
        src.disconnect_device()
        win.close()
