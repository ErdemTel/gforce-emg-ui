"""Gain-table editing client. No ROS or hand SDK dependency; never sends a gain command."""

import ipaddress
import json
import math
import re
import socket
import time
import uuid

PROTOCOL = 'emg-stiffness-config/1'
LEVELS = (0., .5, 1.)
DEFAULT_KP = (.4, .6, 1.)
DEFAULT_KD = (.65, .8, 1.)
STORE_KEY = 'stiffness_gains/draft'


def validate(kp, kd):
    """Same envelope as the bridge; the bridge validates again before changing its table."""
    for label, values, low in (('Kp', kp, .2), ('Kd', kd, .1)):
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(f'{label} needs three multipliers')
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or
               not math.isfinite(v) or not low <= v <= 1. for v in values):
            raise ValueError(f'{label} multipliers must be between {low:g} and 1')
        if any(a > b for a, b in zip(values, values[1:])):
            raise ValueError(f'{label} must not decrease from low to high stiffness')
    for level, p, d in zip(LEVELS, kp, kd):
        if not .8 <= d / math.sqrt(p) <= 1.25:
            raise ValueError(f'Level {level:g}: Kd / sqrt(Kp) must be between 0.8 and 1.25')
    for i in range(2):
        if math.isclose(kp[i], kp[i + 1], rel_tol=1e-4) and math.isclose(
                kd[i], kd[i + 1], rel_tol=1e-4):
            raise ValueError('Adjacent stiffness levels must have different gain pairs')


def validate_response(message, request):
    if not isinstance(message, dict) or message.get('proto') != PROTOCOL or any(
            message.get(k) != request[k] for k in ('id', 'side')):
        raise ValueError('Unexpected gain-table reply')
    if any(not isinstance(message.get(k), bool) for k in ('success', 'armed', 'busy')):
        raise ValueError('Invalid bridge status in gain-table reply')
    if not isinstance(message.get('message'), str) or len(message['message']) > 1024:
        raise ValueError('Invalid bridge message')
    config = message.get('config')
    if config is None and not message['success']:
        return message
    if not isinstance(config, dict) or config.get('side') != request['side']:
        raise ValueError('Bridge returned the wrong hand')
    if not isinstance(config.get('config_hash'), str) or not re.fullmatch(
            r'[0-9a-f]{12}', config['config_hash']):
        raise ValueError('Bridge configuration hash missing')
    if not isinstance(config.get('revision'), str) or not re.fullmatch(
            r'[0-9a-f]{32}', config['revision']):
        raise ValueError('Bridge configuration revision missing')
    table = config.get('table', {})
    if not isinstance(table, dict) or table.get('levels') != list(LEVELS):
        raise ValueError('Bridge returned unexpected stiffness levels')
    validate(table.get('kp_ratios'), table.get('kd_ratios'))
    return message


def request_table(host, port, side, *, kp=None, kd=None, expected_hash='', expected_revision='',
                  cancel=None, timeout=4.):
    """One request, no automatic retries. A timeout leaves an APPLY outcome unconfirmed."""
    address = ipaddress.ip_address(host)
    if address.is_multicast or address.is_unspecified or str(address) == '255.255.255.255':
        raise ValueError('Robot IP must be a unicast address')
    if side not in ('right', 'left') or not 0 < port <= 65535:
        raise ValueError('Select a hand and valid robot UDP port')
    message = {'proto': PROTOCOL, 'id': uuid.uuid4().hex, 'side': side,
               'op': 'read' if kp is None else 'apply'}
    if kp is not None:
        validate(kp, kd)
        if not re.fullmatch(r'[0-9a-f]{12}', expected_hash) or not re.fullmatch(
                r'[0-9a-f]{32}', expected_revision):
            raise ValueError('Read the bridge configuration before applying')
        message.update(kp_ratios=list(kp), kd_ratios=list(kd), expected_hash=expected_hash,
                       expected_revision=expected_revision)
    raw = json.dumps(message, allow_nan=False).encode()
    family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
    peer = socket.getaddrinfo(host, port, family, socket.SOCK_DGRAM, 0,
                              socket.AI_NUMERICHOST)[0][4]
    with socket.socket(family, socket.SOCK_DGRAM) as sock:
        sock.connect(peer)  # Only receive datagrams from the selected bridge endpoint.
        sock.settimeout(.2)
        sock.send(raw)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not (cancel and cancel.is_set()):
            try:
                raw = sock.recv(8193)
            except socket.timeout:
                continue
            if len(raw) > 8192:
                raise ValueError('Gain-table reply too large')
            return validate_response(json.loads(raw), message)
    raise TimeoutError('No confirmed reply. Check that the updated bridge is running; '
                       'read it before retrying an apply.')
