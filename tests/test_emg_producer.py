"""Producer entrypoint: fake EMG and loopback UDP only, never a robot or real armband."""
import sys

import pytest

from conftest import wait_until
from gforce_emg_ui.live import robot_out
from gforce_emg_ui.recorder import SAVED
from test_live_robot_out import Receiver, accepted_low
from test_live_ui import dialogs, make, shut


def test_producer_cli_is_explicit_and_prediction_target_only_prefills():
    from gforce_emg_ui.__main__ import parse_args
    assert not parse_args([]).emg_producer
    args = parse_args(["--emg-producer", "--prediction-udp", "127.0.0.1:6731"])
    assert args.emg_producer and args.robot_target == ("127.0.0.1", 6731)
    assert parse_args(["--emg-producer", "--robot-udp", "127.0.0.1:7000"]).robot_target == (
        "127.0.0.1", 7000)


def test_producer_keeps_recording_without_model_and_excludes_robot_editors(
        qapp, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "emg_classifier", None)
    win, src, hub = make(qapp, tmp_path, emg_producer=True)
    live = win.live
    try:
        assert live.model is None and not live.engine.running and not live.robot.armed
        assert "Producer" in win.windowTitle()
        assert live.source_box.isHidden() and not live.source_mode.isEnabled()
        assert live.gain_panel is None
        assert live.manual_controls.isHidden() and live.direct_controls.isHidden()
        assert all(not box.isHidden() for box in live.classifier_boxes)
        assert live.btn_start.text() == "Start classifier"
        assert live.btn_stop.text() == "Stop classifier"
        assert live.robot_check.text() == "Send predictions"
        assert "Sharpa Control" in live.producer_note.text()
        assert not live.robot_check.isEnabled()
        for index in (1, 2):
            live.source_mode.setCurrentIndex(index)
            assert live.source_mode.currentIndex() == 0
            assert live.robot.mode == robot_out.EMG
            assert not live.robot.armed
        win._toggle_recording()
        assert wait_until(lambda: win.recorder.session.counts["emg"]["received"] > 0, 5, qapp)
        win._toggle_recording()
        session = win.recorder.session
        assert session.wait(10) and session.state == SAVED
        assert session.counts["emg"]["written"] == session.counts["emg"]["received"]
    finally:
        shut(qapp, win, src)


def test_producer_status_and_confirmation_do_not_claim_robot_is_armed(
        qapp, tmp_path, dialogs, monkeypatch):
    win, src, hub = make(qapp, tmp_path, emg_producer=True)
    live = win.live
    try:
        st = live.robot.stats()
        st.update(armed=True, sent=4, robot={
            "armed": False, "dry_run": False, "latched": True, "phase": "blocked",
            "kp_ratio": None, "kd_ratio": None})
        monkeypatch.setattr(live.robot, "stats", lambda: st)
        live._show_robot()
        assert "Prediction stream ON" in live.robot_status.text()
        assert "Sharpa Control" in live.robot_status.text()
        assert "gain writes ON" not in live.robot_status.text()
        assert "applied" not in live.robot_status.text()
        win._on_robot_armed(True, "127.0.0.1:6731")
        assert win.robot_indicator.text() == "PREDICTION STREAM ON → 127.0.0.1:6731"
        assert "ARMED" not in win.robot_indicator.text()
        assert live._confirm_robot(live.robot.settings)
        assert "does not start robot control" in dialogs.texts[-1]
        assert "does not confirm" in dialogs.texts[-1]
        st.update(armed=False, robot=None)
        live._show_robot()
        assert "Prediction stream OFF" in live.robot_status.text()
        assert "does not confirm receipt" in live.robot_status.text()
        win._on_robot_armed(False, "127.0.0.1:6731")
        assert win.robot_indicator.isHidden()
    finally:
        shut(qapp, win, src)


def test_producer_sender_is_emg_only_and_stops_with_same_protocol():
    rig = accepted_low()
    rx = Receiver()
    sender = robot_out.RobotOutput(
        rig.eng, robot_out.RobotOutputSettings(rx.host, rx.port),
        simulated=True, emg_only=True, clock=rig.clock)
    try:
        for mode in (robot_out.MANUAL, robot_out.GAINS):
            with pytest.raises(ValueError, match="predictions only"):
                sender.set_mode(mode)
        with pytest.raises(RuntimeError, match="Manual Kp/Kd"):
            sender.select_gains(.6, .8)
        assert not sender.armed
        sid = sender.arm()
        assert rx.collect(lambda messages: len(messages) >= 2 and messages[-1]["s"] == 0.)
        assert all(message["proto"] == "emg-stiffness/1" and
                   message["source"] == "gforce-emg-ui" and
                   message["session"] == sid and "kp_ratio" not in message
                   for message in rx.msgs)
        rig.clock.t += 1.
        assert rx.collect(lambda messages: messages[-1]["valid"] is False)
        assert rx.msgs[-1]["s"] is None
        sender.disarm("prediction stream stopped")
        assert rx.collect(lambda messages: messages[-1]["kind"] == "stop")
        assert rx.msgs[-1]["proto"] == "emg-stiffness/1" and not rx.msgs[-1]["valid"]
    finally:
        sender.close()
        rig.eng.shutdown()
        rx.close()
