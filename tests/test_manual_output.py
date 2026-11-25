"""Manual controls with no band/model; UDP only to ephemeral loopback receivers."""

import pytest

from conftest import wait_until
from gforce_emg_ui.hub import DataHub
from gforce_emg_ui.live import robot_out
from gforce_emg_ui.sources import SimSource
from test_live_robot_out import Receiver
from test_live_ui import dialogs, make_robot, shut, start_and_arm


def test_manual_sender_debounce_no_classifier_and_session_switch():
    class NoClassifier:
        def snapshot(self):
            raise AssertionError("manual mode must never read the classifier")

    rx = Receiver()
    now = [10.0]
    out = robot_out.RobotOutput(NoClassifier(), robot_out.RobotOutputSettings(rx.host, rx.port),
                                simulated=True, clock=lambda: now[0])
    try:
        with pytest.raises(RuntimeError, match="Manual"):
            out.select_manual(0)
        out.set_mode(robot_out.MANUAL)
        with pytest.raises(RuntimeError, match="select a manual"):
            out.arm()
        for value in (True, None, 0.1, float('nan')):
            with pytest.raises(ValueError):
                out.select_manual(value)
        out.select_manual(0.5)
        session = out.arm()
        assert rx.collect(lambda m: len(m) >= 2)
        assert all(m['s'] is None and not m['valid'] for m in rx.msgs)
        now[0] += 0.101
        assert rx.collect(lambda m: m[-1]['s'] == 0.5)
        with pytest.raises(RuntimeError, match="disarm"):
            out.set_mode(robot_out.EMG)
        out.select_manual(1)
        assert rx.collect(lambda m: m[-1]['s'] is None)
        now[0] += 0.101
        assert rx.collect(lambda m: m[-1]['s'] == 1)
        out.select_manual(0)
        now[0] += 0.101
        assert rx.collect(lambda m: m[-1]['s'] == 0)
        out.disarm('source changed')
        out.set_mode(robot_out.EMG)
        assert rx.collect(lambda m: m[-1]['kind'] == 'stop')
        assert all(m['source'] == 'send_stiffness' and m['session'] == session
                   and m['simulated'] is False and m['run_id'] is None
                   and m['stab_ms'] == dict(up=100, down=100, startup=100) for m in rx.msgs)
        assert rx.msgs[-1]['s'] is None
        assert out.stats()['errors'] == 0
        out.set_mode(robot_out.MANUAL)
        assert out.manual_level is None
    finally:
        out.close()
        rx.close()


def test_manual_gui_without_band_model_or_emg_dependency(qapp, tmp_path, dialogs, monkeypatch):
    import sys
    from PyQt5 import QtWidgets
    from gforce_emg_ui.ui.main_window import MainWindow
    monkeypatch.setitem(sys.modules, 'emg_classifier', None)
    rx = Receiver()
    hub = DataHub()
    src = SimSource(hub)  # never connect, never produce samples
    win = MainWindow(hub, src, tmp_path / 'data', robot_target=(rx.host, rx.port))
    win.show()
    win.tabs.setCurrentWidget(win.live)
    live = win.live
    try:
        assert live.model is None and not live.robot.armed
        live.source_mode.setCurrentIndex(1)
        assert not live.engine.running and not live.start()
        assert not live.robot_check.isEnabled()
        assert not rx.collect(lambda m: m, timeout=0.1)
        live.manual_buttons[0.5].click()
        assert live.robot_check.isEnabled()
        dialogs.answers.append(QtWidgets.QMessageBox.No)
        live.robot_check.click()
        assert not live.robot.armed
        live.robot_check.click()
        assert live.robot.armed and 'MANUAL mode' in dialogs.texts[-1]
        assert 'dry_run=false' in dialogs.texts[-1]
        assert rx.collect(lambda m: m[-1]['s'] == 0.5)
        assert wait_until(lambda: live.s_label.text() == 's = 0.5', 2, qapp)
        assert 'classifier OFF' in live.mode_label.text()
        assert live.plot.isHidden() and live.bars.isHidden()
        assert not live.gain_panel.apply_button.isEnabled()
        session = live.robot.session
        # No EMG dependency, including queued disconnects and guards on other tabs.
        src.disconnected.emit()
        live._on_state()
        win.tabs.setCurrentIndex(0)
        live.robot_tick()
        assert live.robot.armed and live.robot.session == session
        for level in (0, 1):
            live.manual_buttons[level].click()
            assert rx.collect(lambda m: m[-1]['s'] == level)
            assert live.robot.session == session
        win.tabs.setCurrentWidget(live)
        live.robot_check.click()
        assert rx.collect(lambda m: m[-1]['kind'] == 'stop')
        live.robot_check.click()
        assert live.robot.session != session
        assert rx.collect(lambda m: m[-1]['kind'] != 'stop' and m[-1]['s'] == 1)
        win.close()
        assert rx.collect(lambda m: m[-1]['kind'] == 'stop')
        reopened = MainWindow(hub, src, tmp_path / 'data')
        try:
            assert not reopened.live.manual_mode and not reopened.live.robot.armed
            assert reopened.live.robot.manual_level is None
        finally:
            reopened.close()
    finally:
        shut(qapp, win, src)
        rx.close()


def test_switching_sources_stops_classifier_and_requires_explicit_restart(qapp, tmp_path, dialogs):
    win, src, hub, rx = make_robot(qapp, tmp_path)
    live = win.live
    try:
        start_and_arm(qapp, live, rx)
        emg_session = live.robot.session
        live.source_mode.setCurrentIndex(1)
        assert not live.engine.running and not live.robot.armed
        assert live.robot.manual_level is None
        assert rx.collect(lambda m: m[-1]['kind'] == 'stop')
        assert rx.msgs[-1]['session'] == emg_session
        assert rx.msgs[-1]['source'] == 'gforce-emg-ui'
        live.manual_buttons[0].click()
        live.robot_check.click()
        manual_session = live.robot.session
        assert manual_session != emg_session
        assert rx.collect(lambda m: m[-1]['s'] == 0 and m[-1]['session'] == manual_session)
        live.source_mode.setCurrentIndex(0)
        assert rx.collect(lambda m: m[-1]['kind'] == 'stop')
        assert rx.msgs[-1]['source'] == 'send_stiffness'
        assert not live.engine.running and not live.robot.armed
        assert not live.robot_check.isEnabled()
        assert live.btn_start.isEnabled()
        assert not live.plot.isHidden() and not live.bars.isHidden()
        live.source_mode.setCurrentIndex(1)
        assert live.robot.manual_level is None
        assert not any(button.isChecked() for button in live.manual_buttons.values())
    finally:
        shut(qapp, win, src)
        rx.close()
