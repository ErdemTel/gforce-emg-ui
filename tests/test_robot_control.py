"""Operator UDP tests with fake sockets; no Qt, ROS, SDK or network access."""

import copy
import json
import uuid

import pytest

from gforce_emg_ui.live.robot_control import MAX_DATAGRAM, REPLY_PROTO, RobotControlClient


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class FakeSocket:
    def __init__(self, *_args):
        self.sent = []
        self.incoming = []
        self.closed = False
        self.fail_send = False

    def bind(self, address):
        self.bound = address

    def setblocking(self, value):
        self.blocking = value

    def sendto(self, wire, address):
        if self.fail_send:
            raise OSError('injected send failure')
        self.sent.append((wire, address))
        return len(wire)

    def recvfrom(self, size):
        if not self.incoming:
            raise BlockingIOError()
        wire, address = self.incoming.pop(0)
        return wire[:size], address

    def close(self):
        self.closed = True


def state():
    return {
        'schema_version': 1, 'instance_id': str(uuid.uuid4()), 'side': 'right', 'serial': 'SIM-RIGHT',
        'run_state': 'stopped', 'mode': 'manual', 'pending_mode': None, 'dry_run': False,
        'hold_pose': False, 'owner_id': None, 'lease_remaining_sec': 0.,
        'selected': {'kp_ratio': .6, 'kd_ratio': .8},
        'actual': {'kp_ratio': .6, 'kd_ratio': .8, 'kp': [12.] * 22, 'kd': [.4] * 22,
                   'verified': True, 'readback_age_sec': .1},
        'driver': {'state': 'normal', 'reason': '', 'feedback_fresh': True, 'command_idle': True},
        'emg': {'candidate_level': None, 'requested_level': None, 'fresh': False},
        'table': {'levels': [0, .5, 1], 'kp_ratios': [.4, .6, 1.], 'kd_ratios': [.65, .8, 1.]},
        'reason': '',
    }


@pytest.fixture
def setup():
    clock, sock = Clock(), FakeSocket()
    client = RobotControlClient(clock=clock, socket_factory=lambda *_args: sock)
    yield client, clock, sock
    client.close()


def packets(sock, op=None):
    decoded = [json.loads(wire) for wire, _ in sock.sent]
    return decoded if op is None else [item for item in decoded if item['op'] == op]


def enqueue(sock, request, snapshot, *, peer=('127.0.0.1', 6731), **overrides):
    reply = dict(proto=REPLY_PROTO, request_id=request['request_id'], success=True,
                 message='accepted', state=copy.deepcopy(snapshot))
    reply.update(overrides)
    sock.incoming.append((json.dumps(reply).encode(), peer))


def observe(client, sock, snapshot):
    client.tick()
    enqueue(sock, packets(sock, 'observe')[-1], snapshot)
    client.tick()


def test_open_only_observes_without_claiming_or_arming(setup):
    client, clock, sock = setup
    assert sock.sent == []
    assert sock.bound == ('127.0.0.1', 0) and sock.blocking is False
    observe(client, sock, state())
    assert [p['op'] for p in packets(sock)] == ['observe']
    request = packets(sock)[0]
    assert not any(key in request for key in ('client_id', 'expected_instance', 'op_seq'))
    assert client.model.display()['phase'] == 'STOPPED'


def test_observe_runs_at_two_hz_and_state_expiry_blocks_start(setup):
    client, clock, sock = setup
    observe(client, sock, state())
    clock.now = .49
    client.tick()
    assert len(packets(sock, 'observe')) == 1
    clock.now = .5
    client.tick()
    assert len(packets(sock, 'observe')) == 2
    clock.now = 2.01
    client.tick()
    assert not client.model.fresh
    assert not client.submit('start')
    assert not packets(sock, 'start')


@pytest.mark.parametrize('peer', [('127.0.0.2', 6731), ('127.0.0.1', 6732)])
def test_reply_from_wrong_peer_is_ignored(setup, peer):
    client, clock, sock = setup
    client.tick()
    enqueue(sock, packets(sock)[0], state(), peer=peer)
    client.tick()
    assert client.state is None and client.transport.rejected_packets == 1


@pytest.mark.parametrize('change', ['wrong_side', 'wrong_id', 'wrong_proto', 'non_bool_success', 'invalid_state'])
def test_reply_identity_and_schema_are_checked(setup, change):
    client, clock, sock = setup
    client.tick()
    snapshot = state()
    overrides = {}
    if change == 'wrong_side':
        snapshot['side'] = 'left'
    elif change == 'wrong_id':
        overrides['request_id'] = str(uuid.uuid4())
    elif change == 'wrong_proto':
        overrides['proto'] = 'emg-stiffness-state/1'
    elif change == 'non_bool_success':
        overrides['success'] = 1
    else:
        del snapshot['owner_id']
    enqueue(sock, packets(sock)[0], snapshot, **overrides)
    client.tick()
    assert client.state is None
    assert client.transport.rejected_packets == 1


def test_oversize_and_nonfinite_packets_do_not_change_state(setup):
    client, clock, sock = setup
    client.tick()
    sock.incoming.append((b'x' * (MAX_DATAGRAM + 1), client.transport.peer))
    sock.incoming.append((b'{"value":NaN}', client.transport.peer))
    client.tick()
    assert client.state is None and client.transport.rejected_packets == 2


def test_mutation_retries_are_identical_and_bounded(setup):
    client, clock, sock = setup
    observe(client, sock, state())
    assert client.submit('set_manual', kp_ratio=.2, kd_ratio=.5)
    for now in (.31, .71, 1.51, 2.8):
        clock.now = now
        client.tick()
    wires = [wire for wire, _ in sock.sent if json.loads(wire)['op'] == 'set_manual']
    assert len(wires) == 4
    assert len(set(wires)) == 1
    request = json.loads(wires[0])
    assert request['op_seq'] == 0
    assert request['expected_instance'] == client.state['instance_id']
    assert request['side'] == 'right'
    assert client.pending is not None  # UDP sends are not acknowledgments.


def test_successful_mutation_ack_is_not_optimistic_applied_state(setup):
    client, clock, sock = setup
    snapshot = state()
    observe(client, sock, snapshot)
    client.submit('set_manual', kp_ratio=1., kd_ratio=1.)
    response_state = copy.deepcopy(snapshot)
    response_state['selected'] = {'kp_ratio': 1., 'kd_ratio': 1.}
    enqueue(sock, packets(sock, 'set_manual')[-1], response_state)
    client.tick()
    assert client.pending is None
    assert client.state['selected']['kp_ratio'] == .6
    assert '0.6' in client.model.display()['verified']
    assert 'Request accepted' in client.model.notice
    assert client.last_ack['op'] == 'set_manual' and client.last_ack['outcome'] == 'accepted'


def test_backend_restart_rejects_old_mutation_ack(setup):
    client, clock, sock = setup
    snapshot = state()
    observe(client, sock, snapshot)
    client.submit('start')
    new_instance = state()
    enqueue(sock, packets(sock, 'start')[-1], new_instance)
    client.tick()
    assert 'unconfirmed' in client.model.notice
    assert client.state['instance_id'] == snapshot['instance_id']


def test_out_of_order_observation_cannot_replace_newer_state(setup):
    client, clock, sock = setup
    client.tick()
    first = packets(sock, 'observe')[-1]
    clock.now = .6
    client.tick()
    second = packets(sock, 'observe')[-1]
    newest = state()
    enqueue(sock, second, newest)
    client.tick()
    older = copy.deepcopy(newest)
    older['mode'] = 'emg'
    enqueue(sock, first, older)
    client.tick()
    assert client.state['mode'] == 'manual'


def test_each_new_mutation_including_heartbeat_has_increasing_sequence(setup):
    client, clock, sock = setup
    snapshot = state()
    observe(client, sock, snapshot)
    client.submit('set_manual', kp_ratio=.2, kd_ratio=.5)
    enqueue(sock, packets(sock, 'set_manual')[-1], snapshot)
    client.tick()
    snapshot['owner_id'] = client.model.client_id
    client.model.observe(snapshot)
    client.tick()  # Ownership is known, so a heartbeat is now allowed.
    client.submit('start')
    mutations = [item for item in packets(sock) if item['op'] != 'observe']
    assert [item['op_seq'] for item in mutations] == [0, 1, 2]
    assert [item['op'] for item in mutations] == ['set_manual', 'heartbeat', 'start']


def test_pause_and_close_stop_retransmitting_an_earlier_start(setup):
    client, clock, sock = setup
    observe(client, sock, state())
    client.submit('start')
    client.begin_close()
    for now in (.31, .71, 1.51, 4.01):
        clock.now = now
        client.tick()
    assert len(packets(sock, 'start')) == 1
    # The close request is retried using the same identity, never replayed fresh.
    assert len({p['request_id'] for p in packets(sock, 'pause')}) == 1
    start, pause = packets(sock, 'start')[0], packets(sock, 'pause')[0]
    assert pause['op_seq'] > start['op_seq']
    assert client.close_complete and not client.close_confirmed
    before = len(sock.sent)
    client.tick()
    assert len(sock.sent) == before


def test_explicit_pause_supersedes_lost_start_and_ignores_late_reply(setup):
    client, clock, sock = setup
    snapshot = state()
    observe(client, sock, snapshot)
    client.submit('start')
    old_request = packets(sock, 'start')[-1]
    old_future = client.pending[0]
    client.submit('pause')
    pause = packets(sock, 'pause')[-1]
    assert pause['op_seq'] > old_request['op_seq']
    assert old_future.done()
    assert 'outcome unknown' in str(old_future.exception())
    late_state = copy.deepcopy(snapshot)
    late_state['run_state'] = 'running'
    enqueue(sock, old_request, late_state)
    enqueue(sock, pause, snapshot)
    client.tick()
    assert client.pending is None
    assert client.state['run_state'] == 'stopped'
    assert client.transport.rejected_packets == 1
    clock.now = .8
    client.tick()
    assert len(packets(sock, 'start')) == 1


def test_all_rpc_replies_lost_has_terminal_timeout_and_remains_usable(setup):
    client, clock, sock = setup
    snapshot = state()
    observe(client, sock, snapshot)
    client.submit('start')
    old_id = packets(sock, 'start')[-1]['request_id']
    clock.now = 3.01
    client.tick()
    assert client.pending is None and not client.busy
    assert 'outcome unconfirmed' in client.model.notice
    assert old_id not in client.transport.pending
    assert client.last_ack['request_id'] == old_id and client.last_ack['outcome'] == 'unknown'
    client.model.observe(snapshot)
    assert client.submit('pause')
    assert packets(sock, 'pause')[-1]['op_seq'] > packets(sock, 'start')[-1]['op_seq']


def test_no_new_heartbeat_can_overtake_an_unacknowledged_control_request(setup):
    client, clock, sock = setup
    snapshot = state()
    observe(client, sock, snapshot)
    snapshot['owner_id'] = client.model.client_id
    client.model.observe(snapshot)
    client.submit('set_mode', mode='emg')
    clock.now = .2
    client.tick()
    assert not packets(sock, 'heartbeat')
    request = packets(sock, 'set_mode')[-1]
    enqueue(sock, request, snapshot)
    client.tick()
    assert len(packets(sock, 'heartbeat')) == 1
    assert client.last_ack['request_id'] == request['request_id']


def test_pause_rejection_does_not_retry_with_fresh_identity(setup):
    client, clock, sock = setup
    snapshot = state()
    observe(client, sock, snapshot)
    client.submit('pause')
    enqueue(sock, packets(sock, 'pause')[-1], snapshot, success=False, message='rejected')
    client.tick()
    clock.now = 1.1
    client.tick()
    assert len(packets(sock, 'pause')) == 1
    assert client.busy
    assert not packets(sock, 'heartbeat')
    # A new explicit user Start is possible after fresh stopped state is observed.
    client.model.observe(snapshot)
    assert client.submit('start')
    assert len(packets(sock, 'start')) == 1


def test_close_confirmation_refers_to_matching_pause_ack(setup):
    client, clock, sock = setup
    snapshot = state()
    observe(client, sock, snapshot)
    client.begin_close()
    client.tick()
    enqueue(sock, packets(sock, 'pause')[-1], snapshot)
    client.tick()
    assert client.close_complete and client.close_confirmed


def test_send_failure_retains_same_request_for_retry(setup):
    client, clock, sock = setup
    observe(client, sock, state())
    sock.fail_send = True
    assert client.submit('start')
    assert client.pending is not None
    original_id = client.pending[1]['request_id']
    sock.fail_send = False
    clock.now = .31
    client.tick()
    assert packets(sock, 'start')[0]['request_id'] == original_id


@pytest.mark.parametrize('host', ['192.168.0.1', '0.0.0.0', 'example.com', '::1'])
def test_non_loopback_ipv4_targets_rejected_before_opening_socket(host):
    opened = []
    with pytest.raises(ValueError, match='loopback IPv4'):
        RobotControlClient(host=host, socket_factory=lambda *_args: opened.append(True))
    assert not opened


@pytest.mark.parametrize('port', [0, 65536, True, '6731'])
def test_invalid_ports_rejected_before_socket_creation(port):
    with pytest.raises(ValueError, match='port'):
        RobotControlClient(port=port, socket_factory=lambda *_args: pytest.fail('opened socket'))


def test_cleanup_closes_socket_without_claiming_a_pause(setup):
    client, clock, sock = setup
    client.close()
    assert sock.closed
    assert not client.close_confirmed
    client.tick()
    assert sock.sent == []
    assert not client.submit('start')
