"""Gain editing is a configuration operation, tested without ROS or real hardware."""

import importlib.util
import json
from pathlib import Path
import socket
import threading

import pytest
from PyQt5 import QtCore

from conftest import wait_until
from gforce_emg_ui.live import gain_table
from gforce_emg_ui.ui.gain_table_panel import GainTablePanel

KP, KD = [.45, .7, 1.], [.7, .85, 1.]


def reply(*, kp=None, kd=None, armed=False, success=True):
    return {'success': success, 'message': 'Confirmed', 'armed': armed, 'busy': armed,
            'config': {'side': 'right', 'config_hash': 'abcdef123456', 'revision': 'a' * 32,
                       'table': {'levels': [0., .5, 1.],
                                 'kp_ratios': list(kp or gain_table.DEFAULT_KP),
                                 'kd_ratios': list(kd or gain_table.DEFAULT_KD)}}}


@pytest.mark.parametrize('kp,kd,valid', [
    (gain_table.DEFAULT_KP, gain_table.DEFAULT_KD, True), (KP, KD, True),
    ([.6, .4, 1.], KD, False), ([.4, .6, 1.], [.8, .65, 1.], False),
    ([.4, .6, 1.], [.1, .2, 1.], False), ([.4, .6, 1.1], KD, False),
    ([True, .6, 1.], KD, False), ([float('nan'), .6, 1.], KD, False),
    ([.4, .6], KD, False), ([.4, .4, 1.], [.65, .65, 1.], False),
])
def test_profile_validation_matches_bridge(kp, kd, valid):
    if valid:
        gain_table.validate(kp, kd)
    else:
        with pytest.raises(ValueError):
            gain_table.validate(kp, kd)
    bridge_path = Path(__file__).resolve().parents[2] / (
        'ros_ws/src/emg_stiffness_bridge/emg_stiffness_bridge/levels.py')
    if bridge_path.is_file():
        spec = importlib.util.spec_from_file_location('bridge_levels', bridge_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if valid:
            module.LevelTable(kp_ratios=kp, kd_ratios=kd)
        else:
            with pytest.raises(ValueError):
                module.LevelTable(kp_ratios=kp, kd_ratios=kd)


def test_request_uses_configuration_protocol_and_verifies_reply():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
        server.bind(('127.0.0.1', 0))
        server.settimeout(2.)
        seen = []

        def serve():
            raw, peer = server.recvfrom(2048)
            data = json.loads(raw)
            seen.append(data)
            response = reply(kp=KP, kd=KD)
            response.update(proto=gain_table.PROTOCOL, id=data['id'], side=data['side'])
            server.sendto(json.dumps(response).encode(), peer)

        worker = threading.Thread(target=serve)
        worker.start()
        result = gain_table.request_table('127.0.0.1', server.getsockname()[1], 'right',
                                         kp=KP, kd=KD, expected_hash='0123456789ab',
                                         expected_revision='a' * 32)
        worker.join(timeout=3.)
        assert result['success'] and len(seen) == 1
        assert seen[0]['op'] == 'apply' and seen[0]['kp_ratios'] == KP
        assert seen[0]['expected_hash'] == '0123456789ab'
        assert 's' not in seen[0]


@pytest.mark.parametrize('change', [dict(id='wrong_id'), dict(side='left'),
                                  dict(armed=1), dict(config=None)])
def test_bad_response_not_reported_as_applied(change):
    request = dict(id='request_123', side='right')
    response = dict(proto=gain_table.PROTOCOL, **request, **reply())
    response.update(change)
    with pytest.raises(ValueError):
        gain_table.validate_response(response, request)


@pytest.fixture
def panel(qapp):
    calls = []

    def request(host, port, side, **kwargs):
        calls.append((host, port, side, kwargs))
        return reply(kp=kwargs.get('kp'), kd=kwargs.get('kd'))

    store = QtCore.QSettings(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope,
                             'gain-tests', 'panel')
    widget = GainTablePanel(store, lambda: ('127.0.0.1', 6731), request=request)
    widget.show()
    yield widget, calls
    widget.close_client()
    widget.close()


def read(panel, qapp):
    widget, calls = panel
    widget.read_button.click()
    assert wait_until(lambda: not widget._busy, 2., qapp)
    widget._timer.stop()
    assert widget._snapshot is not None


def test_six_fields_read_apply_and_no_automatic_send(panel, qapp):
    widget, calls = panel
    busy = []
    widget.busy_changed.connect(busy.append)
    assert len(widget.kp) == len(widget.kd) == 3 and not calls
    assert widget.draft() == (list(gain_table.DEFAULT_KP), list(gain_table.DEFAULT_KD))
    assert not widget.apply_button.isEnabled()
    read(panel, qapp)
    widget._set_draft(KP, KD)
    assert widget.apply_button.isEnabled()
    widget.apply_button.click()
    assert wait_until(lambda: not widget._busy, 2., qapp)
    assert calls[-1][3]['kp'] == KP and calls[-1][3]['kd'] == KD
    assert calls[-1][3]['expected_hash'] == 'abcdef123456'
    assert calls[-1][3]['expected_revision'] == 'a' * 32
    assert not widget.apply_button.isEnabled()
    assert 'Matches' in widget.draft_label.text()
    assert '0.45, 0.7, 1' in widget.active_label.text()
    assert busy == [True, False, True, False]


def test_invalid_draft_sending_and_armed_bridge_block_apply(panel, qapp):
    widget, calls = panel
    read(panel, qapp)
    widget.kp[0].setValue(.2)
    assert not widget.apply_button.isEnabled() and 'sqrt' in widget.draft_label.text()
    widget._set_draft(KP, KD)
    widget.set_output_active(True)
    assert not widget.apply_button.isEnabled() and not widget.kp[0].isEnabled()
    widget.set_output_active(False)
    widget._snapshot = reply(armed=True)
    widget._edited()
    assert not widget.apply_button.isEnabled()
    assert len(calls) == 1


def test_saved_draft_survives_read_and_target_change_requires_confirmation(panel, qapp):
    widget, calls = panel
    widget._set_draft(KP, KD)
    read(panel, qapp)
    assert widget.draft() == (KP, KD)
    widget.use_active_button.click()
    assert widget.draft()[0] == list(gain_table.DEFAULT_KP)
    widget._set_draft(KP, KD)
    widget.target_changed()
    assert widget._snapshot is None and not widget.apply_button.isEnabled()
    assert widget.draft() == (KP, KD)
    clone = GainTablePanel(widget.store, widget.target, request=widget.request)
    try:
        assert clone.draft() == (KP, KD) and clone._snapshot is None
        assert len(calls) == 1
    finally:
        clone.close_client()


def test_timeout_clears_confirmation(panel, qapp):
    widget, calls = panel
    read(panel, qapp)

    def failed(*args, **kwargs):
        raise TimeoutError('No confirmed reply')

    widget.request = failed
    widget.read_button.click()
    assert wait_until(lambda: not widget._busy, 2., qapp)
    assert widget._snapshot is None and not widget.apply_button.isEnabled()
    assert 'No confirmed reply' in widget.status_label.text()

