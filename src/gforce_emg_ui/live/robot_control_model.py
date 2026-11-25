"""Pure presentation and request rules for the unified robot panel.

Adapted from the newly authored dual_sharpa_wave.compliance.gui_model.
Shared behavior is tested without ROS, Qt, Tk, or vendor SDK imports.

Only backend state can describe an applied gain or running controller. The GUI
does not import ROS, the Sharpa SDK, or the EMG application from this module.
"""

from dataclasses import dataclass
import json
import math
import re
import time
import uuid


RUN_STATES = frozenset(('stopped', 'starting', 'running', 'stopping', 'homing', 'error'))
OPS = frozenset(('start', 'pause', 'set_mode', 'set_manual', 'hold_pose', 'return_home',
                 'return_full_home', 'heartbeat', 'status'))


def ratio(value):
    if isinstance(value, bool):
        raise ValueError('Kp and Kd ratios must be numbers from 0 to 1')
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError('Kp and Kd ratios must be numbers from 0 to 1') from None
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError('Kp and Kd ratios must be numbers from 0 to 1')
    return number


def _number(value, *, minimum=0):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= minimum)


def _parse_state(raw, side):
    try:
        state = json.loads(raw) if isinstance(raw, str) else raw
        if (not isinstance(state, dict) or type(state.get('schema_version')) is not int
                or state['schema_version'] != 1):
            raise ValueError('Unsupported controller state')
        if state.get('side') != side:
            raise ValueError('Controller hand does not match this window')
        if not isinstance(state.get('instance_id'), str) or not state['instance_id']:
            raise ValueError('Controller instance is missing')
        if not isinstance(state.get('serial'), str):
            raise ValueError('Controller serial is missing')
        if state.get('run_state') not in RUN_STATES or state.get('mode') not in ('manual', 'emg'):
            raise ValueError('Controller phase or mode is invalid')
        if state.get('pending_mode') not in (None, 'manual', 'emg'):
            raise ValueError('Pending mode is invalid')
        if type(state.get('dry_run')) is not bool or type(state.get('hold_pose')) is not bool:
            raise ValueError('Controller execution state is invalid')
        if ('owner_id' not in state or
                (state['owner_id'] is not None and not isinstance(state['owner_id'], str))):
            raise ValueError('Controller owner is invalid')
        selected, actual, driver, emg = (state[k] for k in ('selected', 'actual', 'driver', 'emg'))
        if not all(isinstance(item, dict) for item in (selected, actual, driver, emg)):
            raise ValueError('Controller state fields are invalid')
        for key in ('kp_ratio', 'kd_ratio'):
            if not _number(selected[key]) or selected[key] > 1:
                raise ValueError('Requested gain ratios are invalid')
        if type(actual.get('verified')) is not bool or type(driver.get('feedback_fresh')) is not bool:
            raise ValueError('Feedback verification is missing')
        if type(emg.get('fresh')) is not bool:
            raise ValueError('EMG freshness is missing')
        for key in ('kp_ratio', 'kd_ratio'):
            if actual.get(key) is not None and not _number(actual[key]):
                raise ValueError('Device gain readback is invalid')
        if actual.get('readback_age_sec') is not None and not _number(actual['readback_age_sec']):
            raise ValueError('Device readback age is invalid')
        if actual['verified']:
            for key in ('kp', 'kd'):
                values = actual.get(key)
                if not isinstance(values, list) or len(values) != 22 or not all(_number(v) for v in values):
                    raise ValueError('Verified readback must contain all 22 joints')
        if not isinstance(driver.get('state'), str) or not isinstance(state.get('reason', ''), str):
            raise ValueError('Controller status is invalid')
        for key in ('candidate_level', 'requested_level'):
            if emg.get(key) is not None and (isinstance(emg[key], bool) or emg[key] not in (0, 0.5, 1)):
                raise ValueError('EMG level is invalid')
        if 'table' in state:
            table = state['table']
            if (not isinstance(table, dict) or table.get('levels') != [0, .5, 1]
                    or any(not isinstance(table.get(key), list) or len(table[key]) != 3
                           or not all(_number(value) and value <= 1 for value in table[key])
                           for key in ('kp_ratios', 'kd_ratios'))):
                raise ValueError('EMG preset mapping is invalid')
        if 'home' in state:
            home = state['home']
            if (not isinstance(home, dict) or type(home.get('active')) is not bool
                    or (home.get('success') is not None and type(home['success']) is not bool)
                    or not isinstance(home.get('detail'), str)):
                raise ValueError('Home return state is invalid')
        # Detach from caller-owned dictionaries and reject NaN anywhere in the state.
        return json.loads(json.dumps(state, allow_nan=False))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(str(error)) from None


@dataclass(frozen=True)
class Acknowledgment:
    accepted: bool
    message: str
    run_state: str = ''


class ComplianceGuiModel:
    def __init__(self, side, *, clock=time.monotonic, client_id=None, stale_after=2.0):
        if side not in ('left', 'right'):
            raise ValueError('side must be left or right')
        self.side, self.clock, self.stale_after = side, clock, stale_after
        self.client_id = client_id or str(uuid.uuid4())
        uuid.UUID(self.client_id)
        self.state = None
        self.state_generation = 0
        self.received_at = -math.inf
        self.invalid_reason = ''
        self.notice = 'Waiting for controller state; opening this window sends no commands'

    def observe(self, raw):
        try:
            state = _parse_state(raw, self.side)
        except ValueError as error:
            self.invalid_reason = str(error)
            return False
        old_instance = self.state['instance_id'] if self.state else None
        self.state, self.received_at, self.invalid_reason = state, self.clock(), ''
        self.state_generation += 1
        if old_instance and old_instance != state['instance_id']:
            self.notice = 'Controller restarted; review the state and explicitly Start control again'
        return True

    @property
    def fresh(self):
        return bool(self.state is not None and not self.invalid_reason
                    and 0 <= self.clock() - self.received_at <= self.stale_after)

    @property
    def owns_lease(self):
        return bool(self.fresh and self.state['owner_id'] == self.client_id)

    def allowed(self, action):
        state = self.state
        if state is None:
            return False, 'Waiting for controller state'
        if state['owner_id'] not in (None, self.client_id):
            return False, 'Another operator window owns this controller'
        # Stopping updates is still useful when motion feedback has gone stale.
        if action in ('pause', 'release_pose'):
            return True, ''
        if not self.fresh:
            return False, self.invalid_reason or 'Controller state is stale'
        if action in ('heartbeat', 'status'):
            return True, ''
        if state.get('home', {}).get('active'):
            return False, 'Home return is in progress; wait or cancel it with Pause'
        if action in ('return_home', 'return_full_home'):
            if 'home' not in state:
                return False, 'Restart the updated backend to enable home return'
            if action == 'return_full_home' and 'full_robot' not in state['home'].get('scopes', []):
                return False, 'Restart the updated full-home backend; both arms and hands are required'
            if state['dry_run']:
                return False, 'Preview cannot return home'
            if state['driver']['state'] not in ('normal', 'limit_locked'):
                return False, 'Home return needs a normal or limit-locked hand driver'
        if action == 'hold_pose' and state['dry_run']:
            return False, 'Preview cannot publish position targets'
        if not state['driver']['feedback_fresh']:
            return False, 'Hand feedback is stale; no new control request will be sent'
        if action == 'start':
            if state['run_state'] not in ('stopped', 'error'):
                return False, 'Controller is already running or changing state'
            if state['driver']['state'] != 'normal':
                return False, state['driver'].get('reason') or 'Hand driver is not ready'
        if action == 'set_manual' and state['mode'] != 'manual':
            return False, 'Select Manual before applying independent gains'
        if action == 'hold_pose' and state['driver']['state'] != 'normal':
            return False, 'Hand driver is not ready for a pose hold'
        return True, ''

    def request(self, op, **fields):
        if op not in OPS:
            raise ValueError('Unknown compliance operation')
        action = 'release_pose' if op == 'hold_pose' and fields.get('enabled') is False else op
        allowed, reason = self.allowed(action)
        if not allowed:
            raise ValueError(reason)
        if op == 'set_manual':
            fields = {'kp_ratio': ratio(fields.get('kp_ratio')), 'kd_ratio': ratio(fields.get('kd_ratio'))}
        elif op == 'set_mode':
            if fields.get('mode') not in ('manual', 'emg'):
                raise ValueError('Mode must be Manual or EMG')
            fields = {'mode': fields['mode']}
        elif op == 'hold_pose':
            if type(fields.get('enabled')) is not bool:
                raise ValueError('Pose hold must be explicitly enabled or released')
            fields = {'enabled': fields['enabled']}
        elif fields:
            raise ValueError('Unexpected request fields')
        return dict(op=op, client_id=self.client_id, request_id=str(uuid.uuid4()),
                    expected_instance=self.state['instance_id'], **fields)

    def reset_request(self):
        allowed, reason = self.allowed('reset')
        if not allowed:
            raise ValueError(reason)
        # A running EMG-to-manual handover explicitly selects the backend's 1/1 pair.
        if self.state['mode'] == 'emg':
            return self.request('set_mode', mode='manual')
        return self.request('set_manual', kp_ratio=1.0, kd_ratio=1.0)

    def acknowledge(self, request, success, message, state_json):
        try:
            state = _parse_state(state_json, self.side)
            if state.get('request_id') != request['request_id']:
                raise ValueError('Response does not match the pending request')
            if (state['instance_id'] != request['expected_instance'] or self.state is None
                    or self.state['instance_id'] != request['expected_instance']):
                raise ValueError('Controller restarted; request outcome is unconfirmed')
        except ValueError as error:
            result = Acknowledgment(False, str(error))
        else:
            accepted = success is True
            result = Acknowledgment(accepted, str(message), state['run_state'])
        # Responses acknowledge commands. Topic state alone drives the live display,
        # preventing a delayed response from overwriting a newer topic snapshot.
        self.notice = (f'Request accepted: {result.message}; see controller state below'
                       if result.accepted else f'Request not confirmed: {result.message}')
        return result

    def display(self):
        state = self.state
        if state is None:
            return {'identity': f'{self.side.title()} hand · serial unknown', 'execution': 'Execution unknown',
                    'phase': 'Waiting for controller', 'mode': 'Unknown', 'requested': 'Unknown',
                    'verified': 'UNKNOWN', 'readback_age': 'Unknown', 'feedback': 'Unknown',
                    'emg': 'No EMG state', 'reason': self.invalid_reason or 'Waiting for controller state',
                    'hold': 'Pose hold unknown', 'mapping': 'Waiting for the configured level mapping'}
        elapsed = max(0.0, self.clock() - self.received_at)
        actual = state['actual']
        verified = 'UNKNOWN'
        age_text = 'Unknown'
        if self.fresh and actual['verified']:
            if actual.get('kp_ratio') is None or actual.get('kd_ratio') is None:
                verified = 'Nonuniform gains (22 joints verified)'
            else:
                verified = f'Kp × {actual["kp_ratio"]:.4g}   Kd × {actual["kd_ratio"]:.4g}'
            age = actual.get('readback_age_sec')
            age_text = 'Age unavailable' if age is None else f'{age + elapsed:.1f} s since verification'
        elif not self.fresh:
            verified = 'UNKNOWN — controller state stale'
        selected = state['selected']
        phase = state['run_state'].upper()
        reason = state.get('reason') or state['driver'].get('reason') or 'No pending operation'
        if state.get('home', {}).get('active'):
            reason = state['home']['detail']
        if state.get('pending_mode'):
            reason = f'Switching to {state["pending_mode"].upper()}: {reason}'
        if not self.fresh:
            phase = 'STATE UNKNOWN'
            reason = self.invalid_reason or f'No controller state for {elapsed:.1f} s'
        elif state['owner_id'] not in (None, self.client_id):
            reason = f'Another operator window owns control. {reason}'
        emg = state['emg']
        level = emg.get('candidate_level')
        emg_text = 'No level' if level is None else {0: 'Low (0)', .5: 'Medium (0.5)', 1: 'High (1)'}[level]
        serial = state['serial']
        if serial.startswith('SIM-'):
            execution = 'SIMULATION — no physical hand'
            if state['dry_run']:
                execution += ' · Preview'
        elif state['dry_run']:
            execution = 'PREVIEW mode — no gain writes'
        elif re.fullmatch(r'[0-9A-Fa-f]{12}', serial):
            execution = 'HARDWARE mode — LIVE gain-write configuration'
        else:
            execution = 'LIVE requested — device identity unknown'
        table = state.get('table')
        mapping = ('Unavailable' if table is None else '   ·   '.join(
            f'{label}: {kp:.3g} / {kd:.3g}'
            for label, kp, kd in zip(('Low', 'Medium', 'High'), table['kp_ratios'], table['kd_ratios'])))
        source = state['mode'].upper()
        requested = f'Kp × {selected["kp_ratio"]:.4g}   Kd × {selected["kd_ratio"]:.4g}'
        if state['mode'] == 'emg':
            accepted_level = emg.get('requested_level')
            if accepted_level is None:
                requested = 'Waiting for an accepted EMG level'
            elif table is None:
                requested = f'Accepted EMG level {accepted_level:g}; gain mapping unavailable'
            else:
                index = table['levels'].index(accepted_level)
                requested = (f'Kp × {table["kp_ratios"][index]:.4g}   '
                             f'Kd × {table["kd_ratios"][index]:.4g} (EMG)')
        hold = 'Holding current pose' if state['hold_pose'] else 'Pose released for teleop'
        if not self.fresh:
            execution = f'Execution unknown · last reported: {execution}'
            source = f'Last reported: {source}'
            requested = f'Last reported: {requested}'
            hold = f'Unknown · last reported: {hold}'
            mapping = f'Last reported: {mapping}'
        return {'identity': f'{self.side.title()} hand · serial {state["serial"] or "unknown"}',
                'execution': execution,
                'phase': phase, 'mode': source,
                'requested': requested,
                'verified': verified, 'readback_age': age_text,
                'feedback': ('Fresh' if self.fresh and state['driver']['feedback_fresh'] else 'STALE / unavailable'),
                'emg': f'{emg_text} · {"fresh" if self.fresh and emg["fresh"] else "stale / unavailable"}',
                'reason': reason, 'hold': hold,
                'mapping': mapping}
