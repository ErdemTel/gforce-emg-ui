"""Manual Kp/Kd without a classifier; draft edits never send until Apply gains."""
import json

import pytest

from conftest import wait_until
from gforce_emg_ui.hub import DataHub
from gforce_emg_ui.live import robot_out
from gforce_emg_ui.sources import SimSource
import test_live_robot_out as transport
from test_live_ui import dialogs, shut


@pytest.fixture
def rx(monkeypatch):
    legacy = transport.check
    def check(raw):
        data = json.loads(raw)
        if data['proto'] != 'emg-gains/1':
            return legacy(raw)
        assert data['source'] == 'send_stiffness' and data['s'] is None
        assert data['simulated'] is False
        assert data['valid'] == (data['kp_ratio'] is not None)
        assert (data['kp_ratio'] is None) == (data['kd_ratio'] is None)
        if data['valid']:
            assert 0 <= data['kp_ratio'] <= 1 and 0 <= data['kd_ratio'] <= 1
        if data['kind'] == 'stop':
            assert not data['valid']
        assert data['accepted_pred_id'] is None and data['run_id'] is None
        # Check the unchanged envelope against the original independent contract checker.
        envelope = {**data, 'proto': 'emg-stiffness/1', 's': .5 if data['valid'] else None}
        legacy(json.dumps(envelope).encode())
    monkeypatch.setattr(transport, 'check', check)
    receiver = transport.Receiver()
    yield receiver
    receiver.close()


@pytest.mark.parametrize('pair', [(True, .8), (None, .8), (-.0001, .8), (.6, 1.0001),
                                  (float('nan'), .8), (.8, float('inf')), (.2, -.0001)])
def test_pair_rejected(pair):
    with pytest.raises(ValueError):
        robot_out.validate_pair(*pair)


def test_direct_sender_debounces_pair_and_stops_same_protocol(rx):
    now = [10.]
    out = robot_out.RobotOutput(None, robot_out.RobotOutputSettings(rx.host, rx.port),
                                simulated=True, clock=lambda: now[0])
    try:
        out.set_mode(robot_out.GAINS)
        with pytest.raises(RuntimeError, match='pair'):
            out.arm()
        out.select_gains(.6, .8)
        session = out.arm()
        assert rx.collect(lambda m: len(m) >= 2)
        assert all(not msg['valid'] for msg in rx.msgs)
        now[0] += .101
        assert rx.collect(lambda m: m[-1]['kp_ratio'] == .6)
        out.select_gains(.55, .75)
        assert rx.collect(lambda m: not m[-1]['valid'])
        now[0] += .101
        assert rx.collect(lambda m: m[-1]['kp_ratio'] == .55)
        assert rx.msgs[-1]['kd_ratio'] == .75 and rx.msgs[-1]['kind'] == 'change'
        out.disarm('switch')
        out.set_mode(robot_out.MANUAL)
        assert out.manual_gains is None
        assert rx.collect(lambda m: m[-1]['kind'] == 'stop')
        assert all(m['session'] == session and m['proto'] == 'emg-gains/1' for m in rx.msgs)
        assert out.stats()['errors'] == 0
    finally:
        out.close()


def test_direct_gui_edit_while_sending_apply_explicit(qapp, tmp_path, dialogs, rx):
    from gforce_emg_ui.ui.main_window import MainWindow
    hub = DataHub()
    src = SimSource(hub)
    win = MainWindow(hub, src, tmp_path / 'data', robot_target=(rx.host, rx.port))
    win.show()
    win.tabs.setCurrentWidget(win.live)
    live = win.live
    try:
        assert live.source_mode.count() == 3
        live.source_mode.setCurrentIndex(2)
        assert live.direct_mode and not live.engine.running and not live.start()
        assert live.manual_controls.isHidden() and not live.direct_controls.isHidden()
        assert not live.robot_check.isEnabled()
        live.direct_apply.click()
        assert live.robot.manual_gains == (.6, .8)
        live.robot_check.click()
        assert 'MANUAL Kp/Kd' in dialogs.texts[-1]
        assert rx.collect(lambda m: m[-1]['kp_ratio'] == .6)
        session = live.robot.session
        assert live.direct_kp.isEnabled() and live.direct_kd.isEnabled()
        assert not live.gain_panel.apply_button.isEnabled()
        live.direct_kp.setValue(.55)
        live.direct_kd.setValue(.75)
        assert live.robot.manual_gains == (.6, .8)
        assert rx.collect(lambda m: len(m) >= 6)
        assert all(m['kp_ratio'] in (None, .6) for m in rx.msgs)
        live.direct_apply.click()
        assert rx.collect(lambda m: m[-1]['kp_ratio'] == .55)
        assert rx.msgs[-1]['kd_ratio'] == .75 and live.robot.session == session
        assert wait_until(lambda: 'Kp × 0.55' in live.s_label.text(), 2, qapp)
        # Invalid draft cannot alter the committed stream.
        live.direct_kd.lineEdit().clear()
        live._direct_edited()
        assert not live.direct_apply.isEnabled()
        assert live.robot.manual_gains == (.55, .75)
        src.disconnected.emit()
        live.robot_tick()
        assert live.robot.armed
        live.source_mode.setCurrentIndex(1)
        assert rx.collect(lambda m: m[-1]['kind'] == 'stop')
        assert rx.msgs[-1]['proto'] == 'emg-gains/1'
        assert not live.robot.armed and live.robot.manual_level is None
        live.source_mode.setCurrentIndex(2)
        assert live.robot.manual_gains is None and not live.robot_check.isEnabled()
    finally:
        shut(qapp, win, src)


@pytest.mark.parametrize('kp_text,kd_text', [
    ('0.6', '0.75'), ('0', '0'), ('0', '1'), ('1', '0'), ('0.1', '0.05'), ('1', '1'),
])
def test_type_pair_then_one_apply_while_send_stays_on(
        qapp, tmp_path, dialogs, rx, kp_text, kd_text):
    from PyQt5 import QtCore, QtTest
    from gforce_emg_ui.ui.main_window import MainWindow
    hub = DataHub()
    src = SimSource(hub)
    win = MainWindow(hub, src, tmp_path / 'data', robot_target=(rx.host, rx.port))
    win.show()
    win.tabs.setCurrentWidget(win.live)
    live = win.live
    try:
        live.source_mode.setCurrentIndex(2)
        live.direct_apply.click()
        live.robot_check.click()
        assert rx.collect(lambda m: m[-1]['kp_ratio'] == .6)
        session = live.robot.session
        for widget, text in ((live.direct_kp, kp_text), (live.direct_kd, kd_text)):
            widget.setFocus()
            QtTest.QTest.keyClick(widget, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
            QtTest.QTest.keyClicks(widget, text)
        # The button must enable before tabbing out of the last edited field.
        assert live.direct_apply.isEnabled()
        assert live.robot_check.isChecked() and live.robot.manual_gains == (.6, .8)
        before = len(rx.msgs)
        assert rx.collect(lambda m: len(m) >= before + 3)
        assert all(not m['valid'] or (m['kp_ratio'], m['kd_ratio']) == (.6, .8)
                   for m in rx.msgs)
        QtTest.QTest.mouseClick(live.direct_apply, QtCore.Qt.LeftButton)
        pair = float(kp_text), float(kd_text)
        assert rx.collect(lambda m: (m[-1]['kp_ratio'], m[-1]['kd_ratio']) == pair)
        assert live.robot.manual_gains == pair
        assert live.robot_check.isChecked() and live.robot.session == session
        assert not live.gain_panel.apply_button.isEnabled()
    finally:
        shut(qapp, win, src)


def test_optional_readback_pair_and_legacy_state():
    legacy = transport.GOLDEN['state_example']
    assert robot_out.parse_state(json.dumps(legacy).encode()) is not None
    msg = {**legacy, 'kp_ratio': .55, 'kd_ratio': .75, 'applied_s': None}
    assert robot_out.parse_state(json.dumps(msg).encode())['kp_ratio'] == .55
    for value in (True, '0.5', float('nan'), -1):
        assert robot_out.parse_state(json.dumps({**msg, 'kp_ratio': value}).encode()) is None
