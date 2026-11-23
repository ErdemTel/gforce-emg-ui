"""Pure request/lease lifecycle, adapted from the authored compliance GUI session.

Transport-independent: no ROS, Qt, Tk, or Sharpa SDK.
"""

import time
import math


class ComplianceSession:
    """Nonblocking request/lease lifecycle, testable without Tk or ROS.

The transport supplies ready() and call(dict)->Future. A timeout never cancels
an already submitted command or reports that it did not execute.
"""

    REQUEST_TIMEOUT = 3.0
    CLOSE_TIMEOUT = 4.0
    HEARTBEAT_PERIOD = 1.0

    def __init__(self, model, transport, *, clock=time.monotonic):
        self.model, self.transport, self.clock = model, transport, clock
        self.pending = None
        self.heartbeat = None
        self.next_heartbeat = 0.0
        self.pending_timed_out = False
        self.pause_requested = False
        self.closing = False
        self.close_deadline = None
        self.close_complete = False
        self.close_confirmed = False
        self.pause_accepted = False
        self.pause_state_generation = None
        self.next_pause_attempt = 0.0
        self.last_pause_request = None
        self.last_ack = None

    def _send(self, request):
        if not self.transport.ready():
            raise ValueError('Compliance service is unavailable')
        self.pending = (self.transport.call(request), request, self.clock())
        if request['op'] == 'pause':
            self.last_pause_request = request
        self.pending_timed_out = False
        self.model.notice = f'Waiting for {request["op"].replace("_", " ")} acknowledgment'

    def _supersede_waits(self):
        """A newer sequenced Pause supersedes waiting, not server-side work."""
        abandon = getattr(self.transport, 'abandon', None)
        for item in (self.pending, self.heartbeat):
            if item is not None:
                future = item[0]
                if abandon is not None:
                    abandon(future, 'Superseded by newer Pause; earlier server outcome unknown')
                elif not future.done():
                    future.set_exception(RuntimeError('Superseded locally; earlier server outcome unknown'))
        self.pending = self.heartbeat = None
        self.pending_timed_out = False

    def submit(self, op, **fields):
        if self.closing:
            return False
        if op == 'pause':
            self._supersede_waits()
            self.pause_requested = True
            self.pause_accepted = False
            self.next_pause_attempt = self.clock()
            self.model.notice = 'Pause requested; earlier operations may still be draining on the robot'
            self.tick()
            return True
        if (op == 'start' and self.pause_requested and self.pending is None
                and self.model.allowed('start')[0]):
            # Explicit restart is possible only after fresh backend state says
            # stopped/error and the normal Start readiness checks pass.
            self.pause_requested = False
        if self.pending is not None or self.pause_requested:
            self.model.notice = 'Wait for the pending request, or Pause updates'
            return False
        try:
            request = self.model.reset_request() if op == 'reset' else self.model.request(op, **fields)
            self._send(request)
        except (ValueError, RuntimeError) as error:
            self.model.notice = str(error)
            return False
        return True

    def begin_close(self):
        if self.closing:
            return
        self._supersede_waits()
        self.closing, self.pause_requested = True, True
        self.pause_accepted = False
        self.next_pause_attempt = self.clock()
        self.last_pause_request = None
        self.close_deadline = self.clock() + self.CLOSE_TIMEOUT
        self.model.notice = 'Closing: waiting for Pause acknowledgment; no further heartbeats'

    def _read_response(self, future, request):
        try:
            response = future.result()
            return self.model.acknowledge(request, response.success, response.message, response.state_json)
        except Exception as error:
            self.model.notice = f'Request outcome unconfirmed: {error}'
            return None

    def tick(self):
        if self.close_complete:
            return
        now = self.clock()
        if self.heartbeat is not None:
            future, request, sent = self.heartbeat
            if future.done():
                try:
                    response = future.result()
                    # Do not replace the operator's last action notice on a healthy heartbeat.
                    old_notice = self.model.notice
                    ack = self.model.acknowledge(request, response.success, response.message, response.state_json)
                    if ack.accepted:
                        self.model.notice = old_notice
                except Exception as error:
                    self.model.notice = f'Lease renewal unconfirmed: {error}'
                self.heartbeat = None
            elif now - sent > self.REQUEST_TIMEOUT:
                self.model.notice = 'Lease renewal timed out; controller state determines whether updates are active'

        if self.pending is not None:
            future, request, sent = self.pending
            if future.done():
                ack = self._read_response(future, request)
                self.last_ack = {
                    'request_id': request['request_id'], 'op': request['op'],
                    'expected_instance': request['expected_instance'],
                    'accepted': bool(ack and ack.accepted),
                    'outcome': ('accepted' if ack and ack.accepted else
                                'rejected' if ack and ack.run_state else 'unknown'),
                    'message': ack.message if ack else self.model.notice,
                    'run_state': ack.run_state if ack else '',
                }
                self.pending, self.pending_timed_out = None, False
                if request['op'] == 'pause':
                    self.pause_accepted = bool(ack and ack.accepted)
                    self.pause_requested = not self.pause_accepted
                    # A rejected command is not retried with a fresh identity.
                    # Preserve stop intent without renewing the lease; an explicit
                    # Pause click may make a new request after reviewing the state.
                    self.next_pause_attempt = math.inf
                    self.pause_state_generation = self.model.state_generation
                    if self.closing and self.pause_accepted and ack.run_state == 'stopped':
                        self.close_confirmed = self.close_complete = True
            elif now - sent > self.REQUEST_TIMEOUT and not self.pending_timed_out:
                self.pending_timed_out = True
                self.pause_requested = True
                self.model.notice = ('Request timed out; outcome unknown. Heartbeats stopped; '
                                     'waiting to send Pause after its acknowledgment')

        if (self.pause_requested and self.pending is None and not self.close_complete
                and now >= self.next_pause_attempt):
            self.next_pause_attempt = now + 0.5
            try:
                self._send(self.model.request('pause'))
            except (ValueError, RuntimeError) as error:
                self.model.notice = f'Pause unconfirmed: {error}'

        if (self.closing and self.pause_accepted and self.model.fresh
                and self.model.state_generation > self.pause_state_generation
                and self.model.state['run_state'] == 'stopped'):
            self.close_confirmed = self.close_complete = True
        if self.closing and not self.close_complete and now >= self.close_deadline:
            # A late Start acknowledgment can still arrive. Submit one final Pause
            # without claiming it ran, then stop renewing the short owner lease.
            try:
                if self.last_pause_request is None and self.transport.ready():
                    self.transport.call(self.model.request('pause'))
            except (ValueError, RuntimeError):
                pass
            self.model.notice = 'Pause UNCONFIRMED at close; no further lease heartbeats will be sent'
            self.close_complete = True

        if (not self.closing and not self.pause_requested and not self.pending_timed_out
                and self.pending is None
                and self.model.owns_lease and self.heartbeat is None and now >= self.next_heartbeat):
            try:
                if self.transport.ready():
                    request = self.model.request('heartbeat')
                    self.heartbeat = (self.transport.call(request), request, now)
                    self.next_heartbeat = now + self.HEARTBEAT_PERIOD
            except (ValueError, RuntimeError) as error:
                self.model.notice = f'Lease renewal unavailable: {error}'

