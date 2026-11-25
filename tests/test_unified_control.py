"""One-window operator controls, with a fake coordinator and fake/deferred EMG."""
import copy
import sys
import uuid

import pytest

from conftest import wait_until
from gforce_emg_ui.deferred_source import DeferredSdkSource
from gforce_emg_ui.hub import DataHub
from gforce_emg_ui.live.robot_control_model import ComplianceGuiModel
from gforce_emg_ui.live.robot_out import RobotOutputSettings
from gforce_emg_ui.recorder import SAVED
from gforce_emg_ui.sources import FAILED, SimSource, Source
from gforce_emg_ui.ui.main_window import MainWindow


class FakeClient:
    def __init__(self):
        self.model = ComplianceGuiModel("right", clock=lambda: 0.)
        self.calls = []
        self.pending = None
        self.busy = self.closing = self.close_complete = self.close_confirmed = False
        self.closed = False
        self.model.observe({
            "schema_version": 1, "instance_id": str(uuid.uuid4()), "side": "right",
            "serial": "SIM-RIGHT", "run_state": "stopped", "mode": "manual",
            "pending_mode": None, "dry_run": True, "hold_pose": False, "owner_id": None,
            "selected": {"kp_ratio": .6, "kd_ratio": .8},
            "actual": {"kp_ratio": .6, "kd_ratio": .8, "verified": True,
                       "kp": [12.] * 22, "kd": [.4] * 22, "readback_age_sec": 0.},
            "driver": {"state": "normal", "reason": "", "feedback_fresh": True},
            "emg": {"fresh": False, "candidate_level": None, "requested_level": None},
            "reason": "",
        })

    def submit(self, op, **fields):
        if self.closing:
            return False
        self.calls.append((op, fields))
        state = copy.deepcopy(self.model.state)
        state["owner_id"] = self.model.client_id
        if op == "set_manual":
            state["selected"] = fields
        elif op == "set_mode":
            state["mode"] = fields["mode"]
            if fields["mode"] == "manual":
                state["selected"] = {"kp_ratio": 1., "kd_ratio": 1.}
        elif op == "start":
            state["run_state"] = "running"
        elif op == "pause":
            state["run_state"] = "stopped"
        elif op == "reset":
            state["mode"] = "manual"
            state["selected"] = {"kp_ratio": 1., "kd_ratio": 1.}
        self.model.observe(state)
        return True

    def tick(self):
        pass

    def begin_close(self):
        self.closing = True

    def finish_close(self):
        self.close_complete = self.close_confirmed = True

    def close(self):
        self.closed = True


def close_window(qapp, win, client):
    client.finish_close()
    win.close()
    qapp.processEvents()


def test_normal_cli_uses_one_window_and_keeps_explicit_compatibility_paths():
    from gforce_emg_ui.__main__ import parse_args
    assert parse_args([]).control_target == ("127.0.0.1", 6731)
    assert parse_args([]).hand == "right"
    assert parse_args(["--hand", "left"]).hand == "left"
    assert parse_args(["--robot-udp", "127.0.0.1:7001"]).control_target == ("127.0.0.1", 7001)
    args = parse_args(["--robot-control", "127.0.0.1:7002", "--hand", "right"])
    assert args.control_target == ("127.0.0.1", 7002) and args.hand == "right"
    for flag in ("--emg-producer", "--legacy-robot-output"):
        assert parse_args([flag]).control_target is None
    with pytest.raises(SystemExit):
        parse_args(["--emg-producer", "--legacy-robot-output"])
    with pytest.raises(SystemExit):
        parse_args(["--robot-control", "127.0.0.1:6731", "--robot-udp", "127.0.0.1:6732"])


def test_default_source_does_not_import_or_initialize_missing_sensor_sdk(monkeypatch):
    from gforce_emg_ui.__main__ import make_source, parse_args
    monkeypatch.setitem(sys.modules, "sensor", None)
    source = make_source(parse_args([]), DataHub())
    try:
        assert isinstance(source, DeferredSdkSource)
        assert source._inner is None
        source.start_scan()
        assert source.state == FAILED and source._inner is None
    finally:
        source.shutdown()


def test_deferred_source_forwards_controls_and_signals_only_after_scan(qapp):
    calls = []

    class FakeBand(Source):
        def start_scan(self):
            calls.append("scan")
            self.status.emit("fake scan")
        def stop_scan(self):
            calls.append("stop_scan")
        def shutdown(self):
            calls.append("shutdown")

    def factory(hub, rate, package):
        calls.append(("create", rate, package))
        return FakeBand(hub, rate)

    source = DeferredSdkSource(DataHub(), factory=factory)
    seen = []
    source.status.connect(seen.append)
    assert source.backend_name() == "not initialized" and not calls
    source.set_emg_rate(500)
    source.lock_settings(True)
    source.start_scan()
    assert calls == [("create", 500, 16), "scan"]
    assert source._inner.settings_locked
    assert seen == ["fake scan"]
    source.shutdown()
    assert calls[-1] == "shutdown"


def test_manual_tab_works_without_band_sdk_or_classifier(qapp, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "sensor", None)
    monkeypatch.setitem(sys.modules, "emg_classifier", None)
    hub = DataHub()
    source = DeferredSdkSource(hub)
    client = FakeClient()
    win = MainWindow(hub, source, tmp_path/"data",
                     robot_control_target=("127.0.0.1", 6731), robot_client=client)
    win.show()
    panel = win.robot_control
    try:
        assert win.tabs.currentWidget() is panel
        assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == [
            "Robot control", "EMG – Equal scale", "EMG – Detail", "IMU", "Live Classifier"]
        assert win.live.output_box.isHidden() and win.live.source_box.isHidden()
        assert win.live.gain_panel is None
        assert not client.calls and source._inner is None
        assert win.live.model is None and not win.live.engine.running
        assert not win.live.robot.armed
        panel.kp_edit.setText(".2")
        panel.kd_edit.setText(".5")
        panel._draft_edited()
        assert not panel.start_button.isEnabled()
        panel.apply_button.click()
        assert client.calls[-1] == ("set_manual", {"kp_ratio": .2, "kd_ratio": .5})
        assert "0.6" in panel.labels["verified"].text()
        assert "0.2" in panel.labels["requested"].text()
        panel.start_button.click()
        assert client.calls[-1][0] == "start"
        assert panel.labels["phase"].text() == "RUNNING"
        before = len(client.calls)
        source.start_scan()  # Missing SDK is isolated from robot control.
        win.live.stop("operator stopped classifier")
        source.disconnected.emit()
        assert source.state == FAILED
        assert len(client.calls) == before
        assert client.model.state["run_state"] == "running"
        assert not win.live.robot.armed
        panel.pause_button.click()
        assert client.calls[-1][0] == "pause"
    finally:
        close_window(qapp, win, client)


def test_emg_stream_requires_operator_intent_ready_classifier_and_confirmed_mode(
        qapp, tmp_path):
    from types import SimpleNamespace
    hub = DataHub()
    source = DeferredSdkSource(hub)
    client = FakeClient()
    win = MainWindow(hub, source, tmp_path/"data",
                     robot_control_target=("127.0.0.1", 6731), robot_client=client)
    win.plot_timer.stop()
    win.status_timer.stop()
    win.robot_control.timer.stop()
    original = win.live
    ready = [False]
    class Sender:
        armed = False
        starts = 0
        def configure(self, cfg): pass
        def arm(self):
            self.armed = True
            self.starts += 1
    sender = Sender()
    def stop(reason): sender.armed = False
    win.live = SimpleNamespace(
        robot=sender, disarm_robot=stop, robot_blockers=lambda: [] if ready[0] else ["start classifier"],
        robot_draft=RobotOutputSettings, _robot_changed=lambda: None)
    try:
        client.submit("set_mode", mode="emg")
        win._sync_prediction_stream()
        assert not sender.armed  # Observing EMG mode does not start output.
        win._robot_mode_requested("emg")
        assert not sender.armed
        ready[0] = True
        win._sync_prediction_stream()
        assert sender.armed and sender.starts == 1
        win._sync_prediction_stream()
        assert sender.starts == 1
        ready[0] = False
        win._sync_prediction_stream()
        assert not sender.armed
        ready[0] = True
        win._sync_prediction_stream()
        assert sender.armed
        win._robot_pause_requested()
        win._sync_prediction_stream()
        assert not sender.armed
        win._robot_start_requested("emg")
        assert sender.armed
        client.submit("set_mode", mode="manual")
        win._robot_mode_requested("manual")
        assert not sender.armed
        assert client.model.state["run_state"] == "stopped"
    finally:
        win.live = original
        close_window(qapp, win, client)


def test_close_waits_for_pause_and_recording_finalization(qapp, tmp_path):
    hub = DataHub()
    source = SimSource(hub)
    client = FakeClient()
    win = MainWindow(hub, source, tmp_path/"data",
                     robot_control_target=("127.0.0.1", 6731), robot_client=client)
    win.show()
    try:
        source.connect_device(source.ADDRESS)
        assert wait_until(lambda: hub.emg.total > 0, 5, qapp)
        win._toggle_recording()
        assert wait_until(lambda: win.recorder.session.counts["emg"]["received"] > 0, 5, qapp)
        session = win.recorder.session
        win.close()
        assert client.closing and not client.closed
        assert win.isVisible()
        assert wait_until(lambda: not win.recorder.busy, 5, qapp)
        assert session.state == SAVED
        assert win.isVisible() and not client.closed
        client.finish_close()
        assert wait_until(lambda: not win.isVisible(), 3, qapp)
        assert client.closed and not win.live.robot.armed
    finally:
        close_window(qapp, win, client)


@pytest.mark.parametrize("outcome", ["pending_manual", "manual", "lost_ack_confirmed", "rejected", "unknown", "timeout"])
def test_manual_handover_keeps_stream_until_backend_acknowledgment(
        qapp, tmp_path, outcome):
    from types import SimpleNamespace
    hub = DataHub()
    source = DeferredSdkSource(hub)
    client = FakeClient()
    win = MainWindow(hub, source, tmp_path/"data",
                     robot_control_target=("127.0.0.1", 6731), robot_client=client)
    win.plot_timer.stop()
    win.status_timer.stop()
    win.robot_control.timer.stop()
    original = win.live
    now = [0.]
    win._control_clock = lambda: now[0]
    class Sender:
        armed = True
        def configure(self, cfg): pass
        def arm(self): self.armed = True
    sender = Sender()
    stops = []
    def stop(reason):
        if sender.armed:
            stops.append(reason)
        sender.armed = False
    win.live = SimpleNamespace(
        robot=sender, disarm_robot=stop, robot_blockers=lambda: [],
        robot_draft=RobotOutputSettings, _robot_changed=lambda: None)
    try:
        state = copy.deepcopy(client.model.state)
        state.update(mode="emg", run_state="running", owner_id=client.model.client_id)
        client.model.observe(state)
        request_id = str(uuid.uuid4())
        client.pending = (object(), {"request_id": request_id, "op": "set_mode"}, 0.)
        client.last_ack = None
        win._emg_requested = True
        win._robot_mode_requested("manual")
        assert sender.armed and not stops
        assert "keeping EMG stream" in win.robot_control.prediction_label.text()
        now[0] = .5
        win._sync_prediction_stream()
        assert sender.armed and not stops
        if outcome == "pending_manual":
            state["pending_mode"] = "manual"
            client.model.observe(state)
        elif outcome == "manual":
            state["mode"] = "manual"
            client.model.observe(state)
        elif outcome == "lost_ack_confirmed":
            state["pending_mode"] = "manual"
            client.model.observe(state)
            client.last_ack = {"request_id": request_id, "accepted": False,
                               "outcome": "unknown", "message": "reply lost"}
        elif outcome in ("rejected", "unknown"):
            client.last_ack = {"request_id": request_id, "accepted": False,
                               "outcome": outcome, "message": "mode change not confirmed"}
        else:
            now[0] = 3.6
        win._sync_prediction_stream()
        assert not sender.armed and len(stops) == 1
        assert not win._emg_requested and win._manual_handover is None
        if outcome in ("rejected", "unknown", "timeout"):
            assert "unconfirmed" in win.robot_control.prediction_label.text()
            assert client.calls[-1][0] == "pause"
            assert client.model.state["run_state"] == "stopped"
        else:
            assert not client.calls
            assert client.model.state["run_state"] == "running"
        win._sync_prediction_stream()
        assert not sender.armed
        win._on_robot_armed(True, "127.0.0.1:6731")
        assert win.robot_indicator.text().startswith("EMG PREDICTION STREAM ON")
        assert "ARMED" not in win.robot_indicator.text()
    finally:
        win.live = original
        close_window(qapp, win, client)


def test_emg_mode_labels_manual_fields_as_inactive_and_ack_as_historical(qapp, tmp_path):
    hub = DataHub()
    source = DeferredSdkSource(hub)
    client = FakeClient()
    win = MainWindow(hub, source, tmp_path/"data",
                     robot_control_target=("127.0.0.1", 6731), robot_client=client)
    panel = win.robot_control
    try:
        panel._set_draft(1., 1.)
        state = copy.deepcopy(client.model.state)
        state["mode"] = "emg"
        state["emg"].update(fresh=True, candidate_level=.5, requested_level=.5)
        state["table"] = {"levels": [0, .5, 1], "kp_ratios": [.4, .6, 1.],
                          "kd_ratios": [.65, .8, 1.]}
        client.model.observe(state)
        client.model.notice = "Request accepted: Mode handover pending; see controller state below"
        panel.render()
        assert (panel.kp_edit.text(), panel.kd_edit.text()) == ("1", "1")
        assert "inactive in EMG mode" in panel.draft_label.text()
        assert "Fields match" not in panel.draft_label.text()
        assert "0.6" in panel.labels["requested"].text()
        assert panel.notice_label.text().startswith("Last request acknowledgment:")
        assert not panel.kp_edit.isEnabled() and not panel.kd_edit.isEnabled()
    finally:
        close_window(qapp, win, client)
