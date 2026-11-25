"""The actual single Qt window against an isolated fake robot backend.

Run explicitly with RUN_SHARPA_FAKE_TESTS=1 and SHARPA_TEST_SETUP pointing to a
built shared/staged ROS workspace setup.bash. No physical device is used.
"""

import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

import pytest

from conftest import wait_until


pytestmark = pytest.mark.skipif(
    os.environ.get('RUN_SHARPA_FAKE_TESTS') != '1', reason='explicit fake ROS integration opt-in required')


@pytest.fixture
def fake_robot(tmp_path, request):
    side = getattr(request, 'param', 'left')
    setup = Path(os.environ['SHARPA_TEST_SETUP']).resolve()
    assert setup.is_file()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as free:
        free.bind(('127.0.0.1', 0))
        port = free.getsockname()[1]
    env = dict(os.environ, ROS_DOMAIN_ID='66', ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',
               ROS_HOME=str(tmp_path / 'ros_home'), ROS_LOG_DIR=str(tmp_path / 'ros_logs'))
    env.pop('ROS_STATIC_PEERS', None)
    env.pop('PYTHONPATH', None)
    env.pop('LD_LIBRARY_PATH', None)
    command = ['ros2', 'run', 'dual_sharpa_wave', 'compliance_session.py',
               '--hand', side, '--mode', 'fake', '--domain', '66', '--port', str(port),
               '--state-dir', str(tmp_path / 'robot_state')]
    shell = ('source ' + shlex.quote(str(setup)) +
             ' && export PATH=/usr/bin:/bin:"$PATH" && exec ' + shlex.join(command))
    log = tmp_path / 'robot.log'
    stream = log.open('w')
    process = subprocess.Popen(['bash', '-c', shell], env=env, stdout=stream,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        yield ('127.0.0.1', port), process, log
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        stream.close()
        print(log.read_text())


def actual_pair(client, kp, kd):
    state = client.state
    return (state is not None and state['actual']['verified'] and
            state['actual']['kp_ratio'] == pytest.approx(kp) and
            state['actual']['kd_ratio'] == pytest.approx(kd))


@pytest.mark.parametrize('fake_robot', ['right'], indirect=True)
def test_one_window_manual_without_band_or_model_and_mode_switch(qapp, tmp_path, fake_robot, monkeypatch):
    from gforce_emg_ui.hub import DataHub
    from gforce_emg_ui.sources import SimSource, IDLE
    from gforce_emg_ui.ui.main_window import MainWindow
    endpoint, process, log = fake_robot
    monkeypatch.setitem(sys.modules, 'emg_classifier', None)
    hub = DataHub()
    source = SimSource(hub)  # Remains disconnected: no samples, band or model.
    win = MainWindow(hub, source, tmp_path / 'data', robot_control_target=endpoint)
    win.live.set_log_root(tmp_path / 'live_logs')
    win.show()
    panel, client = win.robot_control, win.robot_control.client

    def wait(predicate, timeout=15):
        assert wait_until(predicate, timeout, qapp), json.dumps(client.state) + '\n' + log.read_text()

    try:
        wait(lambda: client.state and client.state['driver']['feedback_fresh']
             and client.state['driver']['state'] == 'normal')
        assert process.poll() is None
        assert client.model.side == client.state['side'] == 'right'
        assert source.state == IDLE and hub.emg.total == 0
        assert win.live.model is None and not win.live.engine.running
        assert win.tabs.currentWidget() is panel
        assert sum(widget.isVisible() for widget in qapp.topLevelWidgets()) == 1
        assert client.state['mode'] == 'manual' and client.state['run_state'] == 'stopped'
        assert client.state['writes_session'] == 0

        panel.kp_edit.setText('0.7')
        panel.kd_edit.setText('0.9')
        panel.apply_button.click()
        wait(lambda: client.state['selected'] == {'kp_ratio': .7, 'kd_ratio': .9})
        wait(panel.start_button.isEnabled)
        panel.start_button.click()
        wait(lambda: actual_pair(client, .7, .9))
        assert '0.7' in panel.labels['verified'].text()

        # EMG selection without a band is explicit and cannot fabricate a prediction.
        wait(panel.mode_combo.isEnabled)
        panel.mode_combo.setCurrentIndex(1)
        wait(lambda: client.state['mode'] == 'emg' and not client.state['pending_mode'])
        assert not client.state['emg']['fresh']
        assert not win.live.robot.armed
        assert actual_pair(client, .7, .9)

        wait(panel.mode_combo.isEnabled)
        panel.mode_combo.setCurrentIndex(0)
        wait(lambda: client.state['mode'] == 'manual' and not client.state['pending_mode'])
        if client.state['run_state'] == 'stopped':
            wait(panel.start_button.isEnabled)
            panel.start_button.click()
        wait(lambda: actual_pair(client, 1., 1.))
        assert source.state == IDLE and hub.emg.total == 0
        assert sum(widget.isVisible() for widget in qapp.topLevelWidgets()) == 1

        wait(panel.apply_button.isEnabled)
        panel.kp_edit.setText('0.6')
        panel.kd_edit.setText('0')
        panel.apply_button.click()
        wait(lambda: actual_pair(client, .6, 0.))
        wait(panel.reset_button.isEnabled)
        panel.reset_button.click()
        wait(lambda: actual_pair(client, 1., 1.))
        wait(panel.hold_button.isEnabled)
        panel.hold_button.click()
        wait(lambda: client.state['hold_pose'])
        wait(panel.home_button.isEnabled)
        count = client.state['writes_session']
        panel.home_button.click()
        wait(lambda: client.state['home']['active'])
        assert client.state['home']['scope'] == 'full_robot'
        assert panel.pause_button.text() == 'Cancel return home'
        # This fixture has one fake hand and no arms. A full-home request must
        # not silently fall back to moving that hand. Cancel while in preflight.
        wait(lambda: 'force limiter' in client.state['home']['detail'])
        panel.pause_button.click()
        wait(lambda: client.state['home']['success'] is False and client.state['run_state'] == 'stopped')
        assert not client.state['hold_pose']
        assert client.state['writes_session'] == count
        assert panel.hold_button.isEnabled()
    finally:
        win.close()
        assert wait_until(lambda: not win.isVisible(), 8, qapp)
        source.shutdown()
    assert client.close_confirmed


def test_same_window_classifier_sends_automatically_and_manual_takes_over(qapp, tmp_path, fake_robot):
    from dataclasses import replace
    from gforce_emg_ui.hub import DataHub
    from gforce_emg_ui.sources import SimSource
    from gforce_emg_ui.ui.main_window import MainWindow
    from test_live_engine import FakePred
    from test_live_ui import fake_model

    class ConstantMedium:
        def predict_windows(self, windows, *, channel_names):
            return [FakePred(True, 'ok', None, 1, 'medium',
                             {'low': 0., 'medium': 1., 'high': 0.}, .5, .5, 1.)
                    for _ in windows]

    endpoint, process, log = fake_robot
    hub = DataHub()
    source = SimSource(hub, contact='good')
    win = MainWindow(hub, source, tmp_path / 'data', robot_control_target=endpoint, hand='left')
    win.live.set_log_root(tmp_path / 'live_logs')
    win.show()
    panel, client, live = win.robot_control, win.robot_control.client, win.live

    def wait(predicate, timeout=15):
        assert wait_until(predicate, timeout, qapp), json.dumps(client.state) + '\n' + log.read_text()

    try:
        wait(lambda: client.state and client.state['driver']['feedback_fresh']
             and client.state['driver']['state'] == 'normal')
        panel.start_button.click()
        wait(lambda: actual_pair(client, 1., 1.))
        wait(panel.hold_button.isEnabled)
        panel.hold_button.click()
        wait(lambda: client.state['hold_pose'] and not client.state['driver']['command_idle'])
        source.connect_device(source.ADDRESS)
        wait(lambda: hub.emg.total > 200)
        live._load_token += 1
        loaded = replace(fake_model(tmp_path, live), clf=ConstantMedium())
        live._on_model_loaded(loaded, live._load_token)
        live.log_check.setChecked(False)
        live.stab_enable.setChecked(True)
        assert live.apply_stabilization()
        live.ack_check.setChecked(True)
        wait(live.btn_start.isEnabled)
        live.btn_start.click()
        wait(lambda: live.engine.running)
        # Merely starting classification in Manual cannot send robot predictions.
        assert not live.robot.armed
        wait(panel.mode_combo.isEnabled)
        panel.mode_combo.setCurrentIndex(1)
        wait(lambda: live.robot.armed and client.state['emg']['fresh'])
        wait(lambda: client.state['mode'] == 'emg' and actual_pair(client, .4, .6), timeout=20)
        assert live.output_box.isHidden()
        assert 'ARMED' not in win.robot_indicator.text()
        assert sum(widget.isVisible() for widget in qapp.topLevelWidgets()) == 1
        screenshot = os.environ.get('SHARPA_GUI_SCREENSHOT')
        if screenshot:
            assert win.grab().save(screenshot)

        wait(panel.mode_combo.isEnabled)
        panel.mode_combo.setCurrentIndex(0)
        wait(lambda: client.state['mode'] == 'manual' and actual_pair(client, 1., 1.))
        assert client.state['run_state'] == 'running'
        wait(lambda: not live.robot.armed)
        # The classifier may remain available; it cannot override Manual.
        source.disconnect_device()
        wait(lambda: not live.engine.running)
        assert client.state['mode'] == 'manual' and actual_pair(client, 1., 1.)
    finally:
        win.close()
        assert wait_until(lambda: not win.isVisible(), 8, qapp)
        source.shutdown()
    assert client.close_confirmed
