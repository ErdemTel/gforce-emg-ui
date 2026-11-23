"""Offline presentation/lease tests. These import neither ROS nor the vendor SDK."""

from concurrent.futures import Future
import copy
import json
from types import SimpleNamespace
import uuid

import pytest

from gforce_emg_ui.live.robot_control_session import ComplianceSession
from gforce_emg_ui.live.robot_control_model import ComplianceGuiModel, ratio


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def state():
    return {
        'schema_version': 1, 'instance_id': str(uuid.uuid4()), 'side': 'left', 'serial': 'CD52943BCD53',
        'run_state': 'stopped', 'mode': 'manual', 'pending_mode': None, 'dry_run': True,
        'hold_pose': False, 'owner_id': None, 'lease_remaining_sec': 0.0,
        'selected': {'kp_ratio': .6, 'kd_ratio': .8},
        'actual': {'kp_ratio': .6, 'kd_ratio': .8, 'verified': True,
                   'kp': [12.] * 22, 'kd': [.4] * 22, 'readback_age_sec': 2.0},
        'driver': {'state': 'normal', 'reason': '', 'feedback_fresh': True, 'command_idle': True},
        'emg': {'candidate_level': None, 'requested_level': None, 'fresh': False}, 'reason': '',
        'table': {'levels': [0, .5, 1], 'kp_ratios': [.4, .6, 1.], 'kd_ratios': [.65, .8, 1.]},
    }


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def model(clock, state):
    model = ComplianceGuiModel('left', clock=clock)
    assert model.observe(json.dumps(state))
    return model


class Transport:
    def __init__(self):
        self.calls = []
        self.available = True

    def ready(self):
        return self.available

    def call(self, request):
        future = Future()
        self.calls.append((request, future))
        return future

    def reply(self, index, state, *, success=True, message='accepted'):
        request, future = self.calls[index]
        snapshot = copy.deepcopy(state)
        snapshot['request_id'] = request['request_id']
        future.set_result(SimpleNamespace(success=success, message=message, state_json=json.dumps(snapshot)))


def test_open_and_state_receipt_do_not_claim_or_start(model, clock):
    transport = Transport()
    session = ComplianceSession(model, transport, clock=clock)
    for _ in range(10):
        session.tick()
    assert transport.calls == []
    assert model.display()['phase'] == 'STOPPED'


def test_request_has_unique_session_request_and_backend_identity(model):
    first = model.request('set_manual', kp_ratio='.2', kd_ratio='.5')
    second = model.request('set_manual', kp_ratio=.2, kd_ratio=.5)
    assert first['client_id'] == second['client_id'] == model.client_id
    assert first['request_id'] != second['request_id']
    uuid.UUID(first['request_id'])
    assert first['expected_instance'] == model.state['instance_id']
    assert (first['kp_ratio'], first['kd_ratio']) == (.2, .5)


@pytest.mark.parametrize('value', [True, False, None, 'nan', float('inf'), -.01, 1.001, 'text'])
def test_invalid_gain_drafts_never_create_requests(model, value):
    with pytest.raises(ValueError, match='from 0 to 1'):
        model.request('set_manual', kp_ratio=value, kd_ratio=.8)


def test_zero_is_valid_explicit_request_and_not_pause(model):
    request = model.request('set_manual', kp_ratio=0, kd_ratio=0)
    assert request['op'] == 'set_manual'
    assert request['kp_ratio'] == request['kd_ratio'] == 0


def test_home_support_readiness_progress_and_no_automatic_requests(model, state):
    assert not model.allowed('return_home')[0]  # Older backend cannot home.
    state['home'] = {'active': False, 'success': None, 'detail': 'Idle'}
    model.observe(state)
    assert not model.allowed('return_home')[0]  # Preview cannot move.
    state['dry_run'] = False
    state['driver']['state'] = 'limit_locked'
    model.observe(state)
    assert model.request('return_home')['op'] == 'return_home'
    state['home'].update(active=True, detail='Returning to zero')
    state['run_state'] = 'homing'
    model.observe(state)
    assert model.display()['phase'] == 'HOMING'
    assert model.display()['reason'] == 'Returning to zero'
    for op in ('start', 'set_mode', 'set_manual', 'reset', 'hold_pose', 'return_home'):
        assert not model.allowed(op)[0]
    assert model.allowed('pause')[0] and model.allowed('heartbeat')[0]
    state['home'].update(active=False, success=True, detail='Home reached')
    state['run_state'], state['driver']['state'] = 'stopped', 'normal'
    model.observe(state)
    assert model.display()['phase'] == 'STOPPED'
    assert model.allowed('hold_pose')[0] and model.allowed('start')[0]


def test_full_home_requires_explicit_backend_capability(model, state):
    state['dry_run'] = False
    state['home'] = {'active': False, 'success': None, 'detail': 'Idle'}
    model.observe(state)
    assert not model.allowed('return_full_home')[0]
    state['home']['scopes'] = ['selected_hand', 'full_robot']
    model.observe(state)
    assert model.request('return_full_home')['op'] == 'return_full_home'
    state['home']['active'] = True
    model.observe(state)
    assert not model.allowed('return_full_home')[0]
    assert model.allowed('pause')[0]


def test_requested_presets_display_without_overwriting_manual_draft(model, state):
    state['table']['kp_ratios'] = [.15, .4, 1.]
    state['table']['kd_ratios'] = [.3, .6, 1.]
    model.observe(state)
    assert model.display()['mapping'] == 'Low: 0.15 / 0.3   ·   Medium: 0.4 / 0.6   ·   High: 1 / 1'
    assert model.state['selected'] == {'kp_ratio': .6, 'kd_ratio': .8}


def test_stale_state_hides_actual_and_blocks_start_but_allows_pause(model, clock):
    clock.now = 2.01
    assert not model.allowed('start')[0]
    assert not model.allowed('set_manual')[0]
    assert 'UNKNOWN' in model.display()['verified']
    assert model.display()['phase'] == 'STATE UNKNOWN'
    assert model.request('pause')['op'] == 'pause'


def test_live_mode_label_does_not_claim_control_is_running(model, state):
    state['dry_run'] = False
    model.observe(state)
    view = model.display()
    assert view['execution'] == 'HARDWARE mode — LIVE gain-write configuration'
    assert view['phase'] == 'STOPPED'


def test_simulation_identity_never_claims_physical_live_writes(model, state):
    state['dry_run'] = False
    state['serial'] = 'SIM-LEFT'
    model.observe(state)
    assert model.display()['execution'] == 'SIMULATION — no physical hand'
    state['serial'] = ''
    model.observe(state)
    assert 'identity unknown' in model.display()['execution']


def test_stale_state_does_not_claim_a_current_pose_hold_or_execution_mode(model, state, clock):
    state['hold_pose'] = True
    state['dry_run'] = False
    model.observe(state)
    clock.now = 3.
    view = model.display()
    assert view['hold'].startswith('Unknown')
    assert view['execution'].startswith('Execution unknown')
    assert view['requested'].startswith('Last reported')
    assert view['mode'].startswith('Last reported')


def test_mapping_is_reported_from_backend_without_changing_manual_selection(model, state):
    state['table']['kp_ratios'] = [.3, .7, .9]
    model.observe(state)
    assert 'Low: 0.3 / 0.65' in model.display()['mapping']
    assert 'Medium: 0.7 / 0.8' in model.display()['mapping']
    assert model.state['selected'] == {'kp_ratio': .6, 'kd_ratio': .8}


def test_emg_requested_pair_comes_from_accepted_level_not_saved_manual_draft(model, state):
    state['mode'] = 'emg'
    state['selected'] = {'kp_ratio': .8, 'kd_ratio': .7}
    state['emg']['requested_level'] = .5
    model.observe(state)
    assert model.display()['requested'] == 'Kp × 0.6   Kd × 0.8 (EMG)'
    assert model.state['selected'] == {'kp_ratio': .8, 'kd_ratio': .7}
    state['emg']['requested_level'] = None
    model.observe(state)
    assert model.display()['requested'] == 'Waiting for an accepted EMG level'


def test_emg_request_with_unknown_mapping_never_uses_manual_draft(model, state):
    state['mode'] = 'emg'
    state['emg']['requested_level'] = .5
    del state['table']
    model.observe(state)
    assert model.display()['requested'] == 'Accepted EMG level 0.5; gain mapping unavailable'


def test_feedback_stale_blocks_actions_even_with_recent_controller_state(model, state):
    state['driver']['feedback_fresh'] = False
    model.observe(state)
    for action in ('start', 'set_manual', 'set_mode', 'reset', 'hold_pose'):
        assert not model.allowed(action)[0]
    assert model.allowed('pause')[0]
    assert model.allowed('release_pose')[0]


def test_preview_prohibits_pose_hold_but_allows_release_and_gain_selection(model, state):
    assert state['dry_run'] is True
    allowed, reason = model.allowed('hold_pose')
    assert not allowed
    assert reason == 'Preview cannot publish position targets'
    with pytest.raises(ValueError, match='Preview cannot publish position targets'):
        model.request('hold_pose', enabled=True)
    assert model.request('hold_pose', enabled=False)['enabled'] is False
    assert model.request('set_manual', kp_ratio=.2, kd_ratio=.5)['op'] == 'set_manual'
    state['dry_run'] = False
    model.observe(state)
    assert model.allowed('hold_pose')[0]


def test_malformed_or_wrong_side_state_is_never_fresh(model, state):
    state['side'] = 'right'
    assert not model.observe(state)
    assert not model.fresh
    assert 'UNKNOWN' in model.display()['verified']
    state['side'] = 'left'
    state['actual']['kp'] = [12.]
    assert not model.observe(state)
    assert '22 joints' in model.invalid_reason


def test_string_gain_in_backend_state_is_rejected(model, state):
    state['selected']['kp_ratio'] = '0.6'
    assert not model.observe(state)


def test_missing_owner_metadata_is_rejected_instead_of_enabling_controls(model, state):
    del state['owner_id']
    assert not model.observe(state)
    assert not model.allowed('start')[0]


def test_another_window_owner_blocks_commands_and_heartbeats(model, state, clock):
    state['owner_id'] = str(uuid.uuid4())
    model.observe(state)
    for action in ('start', 'set_manual', 'pause', 'heartbeat'):
        assert not model.allowed(action)[0]
    transport = Transport()
    ComplianceSession(model, transport, clock=clock).tick()
    assert not transport.calls


def test_acceptance_does_not_replace_verified_gains_or_live_phase(model, state):
    request = model.request('set_manual', kp_ratio=1, kd_ratio=1)
    response = copy.deepcopy(state)
    response['request_id'] = request['request_id']
    response['run_state'] = 'starting'
    response['selected'] = {'kp_ratio': 1., 'kd_ratio': 1.}
    ack = model.acknowledge(request, True, 'queued', json.dumps(response))
    assert ack.accepted
    assert model.display()['phase'] == 'STOPPED'
    assert model.state['selected']['kp_ratio'] == .6
    assert '0.6' in model.display()['verified']


def test_late_response_from_old_instance_cannot_override_restart(model, state):
    request = model.request('start')
    response = copy.deepcopy(state)
    response['request_id'] = request['request_id']
    state['instance_id'] = str(uuid.uuid4())
    model.observe(state)
    ack = model.acknowledge(request, True, 'running', json.dumps(response))
    assert not ack.accepted
    assert 'restarted' in ack.message
    assert model.state['instance_id'] == state['instance_id']


def test_response_request_id_mismatch_is_unconfirmed(model, state):
    request = model.request('start')
    state['request_id'] = str(uuid.uuid4())
    ack = model.acknowledge(request, True, 'ok', json.dumps(state))
    assert not ack.accepted


def test_readback_age_advances_without_claiming_a_new_read(model, clock):
    clock.now = .75
    assert model.display()['readback_age'] == '2.8 s since verification'


def test_reset_is_explicit_one_one_and_emg_handover(model, state):
    request = model.reset_request()
    assert request['op'] == 'set_manual'
    assert request['kp_ratio'] == request['kd_ratio'] == 1.
    state['mode'] = 'emg'
    model.observe(state)
    request = model.reset_request()
    assert (request['op'], request['mode']) == ('set_mode', 'manual')


def test_heartbeat_requires_confirmed_ownership_and_stops_on_close(model, state, clock):
    transport = Transport()
    session = ComplianceSession(model, transport, clock=clock)
    session.tick()
    assert not transport.calls
    state['owner_id'] = model.client_id
    model.observe(state)
    session.tick()
    assert [r['op'] for r, _ in transport.calls] == ['heartbeat']
    session.begin_close()
    session.tick()
    assert [r['op'] for r, _ in transport.calls] == ['heartbeat', 'pause']
    clock.now = 1.1
    session.tick()
    assert len([r for r, _ in transport.calls if r['op'] == 'heartbeat']) == 1


def test_close_supersedes_inflight_start_then_waits_for_pause_confirmation(model, state, clock):
    transport = Transport()
    session = ComplianceSession(model, transport, clock=clock)
    assert session.submit('start')
    session.begin_close()
    session.tick()
    assert [r['op'] for r, _ in transport.calls] == ['start', 'pause']
    assert transport.calls[0][1].done()  # Local wait retired; server work is not cancelled.
    state['run_state'] = 'stopping'
    transport.reply(1, state)
    session.tick()
    assert not session.close_complete
    state['run_state'] = 'stopped'
    model.observe(state)
    session.tick()
    assert session.close_complete and session.close_confirmed


def test_close_timeout_sends_best_effort_pause_without_claiming_completion(model, clock):
    transport = Transport()
    session = ComplianceSession(model, transport, clock=clock)
    session.submit('start')
    session.begin_close()
    clock.now = session.CLOSE_TIMEOUT + .01
    session.tick()
    assert session.close_complete and not session.close_confirmed
    assert [r['op'] for r, _ in transport.calls] == ['start', 'pause']
    assert 'UNCONFIRMED' in model.notice


def test_pending_timeout_retains_future_and_eventually_pauses(model, state, clock):
    transport = Transport()
    session = ComplianceSession(model, transport, clock=clock)
    session.submit('start')
    future = transport.calls[0][1]
    clock.now = 3.1
    session.tick()
    assert session.pending_timed_out and session.pending is not None
    assert not future.cancelled()
    assert not session.submit('set_manual', kp_ratio=.2, kd_ratio=.5)
    transport.reply(0, state)
    session.tick()
    assert transport.calls[-1][0]['op'] == 'pause'


def test_service_unavailable_does_not_create_a_request(model, clock):
    transport = Transport()
    transport.available = False
    session = ComplianceSession(model, transport, clock=clock)
    assert not session.submit('start')
    assert not transport.calls
    assert 'unavailable' in model.notice


def test_pause_intent_survives_service_outage_without_resuming_heartbeats(model, state, clock):
    state['owner_id'] = model.client_id
    state['run_state'] = 'running'
    model.observe(state)
    transport = Transport()
    transport.available = False
    session = ComplianceSession(model, transport, clock=clock)
    session.submit('pause')
    assert session.pause_requested and not transport.calls
    transport.available = True
    clock.now = .6
    session.tick()
    assert [r['op'] for r, _ in transport.calls] == ['pause']
    assert session.pause_requested


def test_rejected_pause_does_not_silently_resume_lease_renewal(model, state, clock):
    state['owner_id'] = model.client_id
    state['run_state'] = 'running'
    model.observe(state)
    transport = Transport()
    session = ComplianceSession(model, transport, clock=clock)
    session.submit('pause')
    transport.reply(0, state, success=False, message='temporarily unavailable')
    session.tick()
    assert session.pause_requested
    assert [r['op'] for r, _ in transport.calls] == ['pause']
    clock.now = .6
    session.tick()
    assert [r['op'] for r, _ in transport.calls] == ['pause']
    session.submit('pause')  # Only a new explicit operator request creates a new ID.
    assert [r['op'] for r, _ in transport.calls] == ['pause', 'pause']
