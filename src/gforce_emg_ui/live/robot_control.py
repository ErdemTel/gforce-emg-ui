"""Loopback UDP operator client for the existing Qt application.

The robot coordinator owns hardware writes. This module only observes its state
and sends explicit operator requests; importing it does not open a socket.
"""

from concurrent.futures import Future
from dataclasses import dataclass
import ipaddress
import json
import math
import socket
import time
from types import SimpleNamespace
import uuid

from .robot_control_model import ComplianceGuiModel
from .robot_control_session import ComplianceSession


REQUEST_PROTO = 'sharpa-control/1'
REPLY_PROTO = 'sharpa-control-state/1'
MAX_DATAGRAM = 16384


def _reject_constant(value):
    raise ValueError(f'Nonfinite JSON value: {value}')


@dataclass
class _Pending:
    request: dict
    wire: bytes
    future: Future
    sent_at: float
    next_retry: float
    sequence: int = 0
    attempts: int = 1


class UdpOperatorTransport:
    """Bounded nonblocking datagrams with exact peer and request correlation.

Retries preserve the original request ID and payload. The coordinator dedupes
mutations; successful UDP send calls are never treated as acknowledgments.
"""

    RETRY_DELAYS = (.3, .7, 1.5)
    OBSERVE_PERIOD = .5
    OBSERVE_RETENTION = 3.0
    RPC_TIMEOUT = 3.0

    def __init__(self, model, *, host='127.0.0.1', port=6731, clock=time.monotonic,
                 socket_factory=socket.socket):
        host = '127.0.0.1' if host == 'localhost' else host
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            raise ValueError('Robot control supports a loopback IPv4 address only') from None
        if address.version != 4 or not address.is_loopback:
            raise ValueError('Robot control supports a loopback IPv4 address only')
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError('Robot control port must be an integer from 1 to 65535')
        self.model, self.clock = model, clock
        self.peer = (str(address), port)
        self.socket = socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.bind(('127.0.0.1', 0))
            self.socket.setblocking(False)
        except Exception:
            self.socket.close()
            raise
        self.pending = {}
        self.closed = False
        self.next_observe = 0.0
        self.observe_sequence = 0
        self.last_observe_sequence = 0
        self.op_sequence = 0
        self.sent_packets = self.received_packets = self.rejected_packets = 0
        self.last_error = ''

    def ready(self):
        return not self.closed

    def _send(self, pending):
        try:
            count = self.socket.sendto(pending.wire, self.peer)
            if count != len(pending.wire):
                raise OSError('Incomplete UDP datagram send')
            self.sent_packets += 1
            return True
        except OSError as error:
            # Keep the same pending request for retry. Send failure does not prove
            # whether a prior datagram reached the coordinator.
            self.last_error = str(error)
            return False

    def call(self, request):
        if self.closed:
            raise RuntimeError('Robot control socket is closed')
        body = dict(request)
        body.update(proto=REQUEST_PROTO, side=self.model.side)
        if not isinstance(body.get('request_id'), str):
            raise ValueError('Request ID is required')
        uuid.UUID(body['request_id'])
        if body['request_id'] in self.pending:
            raise ValueError('This request ID is already pending')
        if body.get('op') != 'observe':
            body['op_seq'] = self.op_sequence
            self.op_sequence += 1
        wire = json.dumps(body, allow_nan=False, separators=(',', ':')).encode('utf-8')
        if len(wire) > MAX_DATAGRAM:
            raise ValueError('Robot control request is too large')
        future = Future()
        now = self.clock()
        pending = _Pending(body, wire, future, now, now + self.RETRY_DELAYS[0])
        self.pending[body['request_id']] = pending
        self._send(pending)
        return future

    def _observe(self):
        request = {'op': 'observe', 'request_id': str(uuid.uuid4())}
        self.call(request)
        self.observe_sequence += 1
        self.pending[request['request_id']].sequence = self.observe_sequence

    def abandon(self, future, reason='Superseded locally; server outcome unknown'):
        """Retire a local wait only; never claim to cancel server-side work."""
        for request_id, pending in list(self.pending.items()):
            if pending.future is future:
                del self.pending[request_id]
                if not future.done():
                    future.set_exception(RuntimeError(reason))
                return True
        return False

    def _receive(self, data, peer):
        if peer != self.peer or len(data) > MAX_DATAGRAM:
            self.rejected_packets += 1
            return
        try:
            reply = json.loads(data.decode('utf-8'), parse_constant=_reject_constant)
            if not isinstance(reply, dict) or reply.get('proto') != REPLY_PROTO:
                raise ValueError('Wrong reply protocol')
            request_id = reply.get('request_id')
            if not isinstance(request_id, str) or request_id not in self.pending:
                raise ValueError('Unknown reply request ID')
            pending = self.pending[request_id]
            state = reply.get('state')
            if (type(reply.get('success')) is not bool or not isinstance(reply.get('message'), str)
                    or not isinstance(state, dict) or state.get('side') != self.model.side
                    or not isinstance(state.get('instance_id'), str) or not state['instance_id']):
                raise ValueError('Invalid reply identity or state')
            if state.get('request_id', request_id) != request_id:
                raise ValueError('Conflicting response request IDs')
            if pending.request['op'] == 'observe':
                if reply['success'] and pending.sequence > self.last_observe_sequence:
                    if not self.model.observe(state):
                        raise ValueError(self.model.invalid_reason)
                    self.last_observe_sequence = pending.sequence
            else:
                # Delayed command replies cannot replace a newer observation.
                # The model checks expected_instance again when the session reads
                # this result; a backend restart is an unconfirmed command.
                state = dict(state, request_id=request_id)
                pending.future.set_result(SimpleNamespace(
                    success=reply['success'], message=reply['message'],
                    state_json=json.dumps(state, allow_nan=False)))
            del self.pending[request_id]
            self.received_packets += 1
        except (ValueError, TypeError, UnicodeDecodeError) as error:
            self.rejected_packets += 1
            self.last_error = str(error)

    def poll(self, *, stopping=False):
        if self.closed:
            return
        # Bound processing so a packet flood cannot starve the Qt event loop.
        for _ in range(64):
            try:
                data, peer = self.socket.recvfrom(MAX_DATAGRAM + 1)
            except BlockingIOError:
                break
            except OSError as error:
                self.last_error = str(error)
                break
            self._receive(data, peer)
        now = self.clock()
        for request_id, pending in list(self.pending.items()):
            op = pending.request['op']
            if op == 'observe':
                if now - pending.sent_at > self.OBSERVE_RETENTION:
                    del self.pending[request_id]
                continue
            if now - pending.sent_at >= self.RPC_TIMEOUT:
                if not pending.future.done():
                    pending.future.set_exception(TimeoutError(
                        'Robot request timed out; outcome unknown. Reconcile the observed controller state.'))
                del self.pending[request_id]
                continue
            if stopping and op != 'pause':
                continue
            if pending.attempts <= len(self.RETRY_DELAYS) and now >= pending.next_retry:
                self._send(pending)
                pending.attempts += 1
                pending.next_retry = (pending.sent_at + self.RETRY_DELAYS[pending.attempts - 1]
                                      if pending.attempts <= len(self.RETRY_DELAYS) else math.inf)
        if now >= self.next_observe:
            self._observe()
            self.next_observe = now + self.OBSERVE_PERIOD

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.socket.close()
        for pending in self.pending.values():
            if not pending.future.done():
                pending.future.set_exception(RuntimeError('Socket closed; request outcome unconfirmed'))
        self.pending.clear()


class RobotControlClient:
    """Qt-free operator client; call tick() from the existing window's timer.

Construction opens only a local UDP socket. tick() observes state without
claiming control. submit() returns local submission success, never hardware
completion; display/state and verified readback remain the authority.
"""

    def __init__(self, side='right', host='127.0.0.1', port=6731, *, clock=time.monotonic,
                 socket_factory=socket.socket, client_id=None):
        self.model = ComplianceGuiModel(side, clock=clock, client_id=client_id)
        self.transport = UdpOperatorTransport(self.model, host=host, port=port, clock=clock,
                                              socket_factory=socket_factory)
        self.session = ComplianceSession(self.model, self.transport, clock=clock)
        self.closed = False

    @property
    def state(self):
        return self.model.state

    @property
    def pending(self):
        return self.session.pending

    @property
    def last_ack(self):
        return dict(self.session.last_ack) if self.session.last_ack is not None else None

    @property
    def busy(self):
        return self.pending is not None or self.session.pause_requested or self.closing

    @property
    def closing(self):
        return self.session.closing

    @property
    def close_complete(self):
        return self.session.close_complete

    @property
    def close_confirmed(self):
        return self.session.close_confirmed

    def submit(self, op, **fields):
        if self.closed:
            self.model.notice = 'Robot control connection is closed'
            return False
        return self.session.submit(op, **fields)

    def tick(self):
        if self.closed or self.close_complete:
            return
        stopping = self.closing or self.session.pause_requested or self.session.pending_timed_out
        self.transport.poll(stopping=stopping)
        self.session.tick()

    def begin_close(self):
        self.session.begin_close()

    def close(self):
        # This only closes the client. Call begin_close() and keep ticking first
        # when a Pause acknowledgment is needed. No local cleanup claims a stop.
        if not self.closed:
            self.closed = True
            self.transport.close()
