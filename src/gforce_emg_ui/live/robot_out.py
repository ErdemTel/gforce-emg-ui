"""Optional robot output (Qt-free): the shown s as UDP datagrams, protocol emg-stiffness/1.

  arm()             -> new session id, seq 0, one daemon thread 'live-robot-out' per session
  sender thread     -> every 1 / rate_hz s: snap = engine.snapshot(); snap.stable(now) is
                       exactly the value the tab shows (stabilized, or raw while
                       stabilization is OFF); kind 'change' if (s, stabilized) differs from
                       the previous datagram, else 'heartbeat'; non-blocking sendto() (errors
                       counted, never raised); non-blocking drain of the robot's
                       emg-stiffness-state/1 datagrams arriving on the same socket
  disarm() / close()-> only signal the thread: it sends kind 'stop' (s null) as its last
                       datagram, closes its socket and exits. close(timeout) is the only
                       call that waits (bounded join). No value datagram goes out once
                       the stop is signalled: a caller that disarms before changing the
                       model or the stabilizer never has the new state sent as a value.

No valid output (s = --) is sent as s null / valid false, never as 0: the robot decides
what to do with it (emg_stiffness_bridge holds its gains). seq is written by the sender
thread only. Targets are IP literals (no DNS on the GUI or sender thread); multicast,
broadcast and unspecified addresses are refused. Contract: design section 5.1-5.2 in the
parent repository (docs/design_emg_stiffness_2026-11-02.md) and the golden vectors
tests/data/emg-stiffness-v1-golden.json.
"""

import collections
import ipaddress
import json
import math
import numbers
import re
import socket
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, fields
from typing import Callable, List, Optional, Tuple

from .engine import RUNNING
from .stabilization import OFF, describe

PROTOCOL = "emg-stiffness/1"
STATE_PROTOCOL = "emg-stiffness-state/1"
DEFAULT_PORT = 6731
SOURCE = "gforce-emg-ui"
LEVELS = (0.0, 0.5, 1.0)
HEARTBEAT, CHANGE, STOP = "heartbeat", "change", "stop"
KINDS = (HEARTBEAT, CHANGE, STOP)
MAX_DATAGRAM_BYTES = 2048
MAX_STATE_BYTES = 8192          # state datagrams are ~0.4 kB; leaves room for unknown keys
STATE_TIMEOUT_S = 2.0           # no state datagram for longer: "robot state unknown"
SEQ_MAX = 2 ** 53               # largest integer every JSON reader keeps exact
THREAD_NAME = "live-robot-out"
SETTINGS_SCHEMA = 1
STORE_KEY, SCHEMA_KEY = "robot_output/settings", "robot_output/schema"
LIMITS = {"port": (1, 65535), "rate_hz": (5, 50)}
PREV_JOIN_S = 1.0               # a new session first waits this long for the old 'stop'
STOP_TRIES = 3                  # 'stop' retries if sendto fails (e.g. full send buffer)
DRAIN_MAX = 32                  # state datagrams read per period (the robot sends ~2 Hz)
_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{8,64}")
_STATE_TYPES = {"proto": str, "side": str, "armed": bool, "dry_run": bool, "latched": bool,
                "phase": str, "requested_s": float, "target_s": float, "applied_s": float,
                "detail": str, "control_state": str, "teleop_active": bool, "seq": int,
                "session": str}


class SettingsError(ValueError):
    pass


def host_error(host) -> str:
    """'' if host is a usable unicast IP literal, else why not (actionable)."""
    if not isinstance(host, str) or not host:
        return f"robot IP must be an IP address such as 127.0.0.1, got {host!r}"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return (f"robot IP {host!r} is not an IP address literal (e.g. 127.0.0.1 or ::1; host "
                f"names are never resolved)")
    mapped = getattr(ip, "ipv4_mapped", None)
    for a in (ip, mapped):
        if a is None:
            continue
        if a.is_unspecified:
            return f"robot IP {host} is unspecified: give the robot PC's address"
        if a.is_multicast:
            return f"robot IP {host} is a multicast address: give the robot PC's address"
        if a == ipaddress.IPv4Address("255.255.255.255"):
            return f"robot IP {host} is the broadcast address: give the robot PC's address"
    if mapped is not None:                        # one spelling per target (and socket family)
        return f"robot IP {host} is IPv4-mapped: give it as {mapped}"
    return ""


def format_target(host: str, port: int) -> str:
    """'ip:port', with [brackets] around an IPv6 address."""
    return f"[{host}]:{port}" if ":" in str(host) else f"{host}:{port}"


@dataclass(frozen=True)
class RobotOutputSettings:
    """Target and send rate. The armed state is never part of the settings (or stored)."""
    host: str = "127.0.0.1"         # IP literal; default = the robot PC over loopback
    port: int = DEFAULT_PORT
    rate_hz: int = 20               # datagrams per second (heartbeat; a change waits <= 1/rate)

    def errors(self) -> List[str]:
        out = []
        err = host_error(self.host)
        if err:
            out.append(err)
        for name, (lo, hi) in LIMITS.items():
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                out.append(f"{name} must be a whole number, got {v!r}")
            elif not lo <= v <= hi:
                out.append(f"{name} {v} outside {lo}–{hi}")
        return out

    @property
    def target(self) -> str:
        return format_target(self.host, self.port)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d) -> "RobotOutputSettings":
        """Complete, validated settings or SettingsError (never partly applied)."""
        if not isinstance(d, dict):
            raise SettingsError(f"settings must be a mapping, got {type(d).__name__}")
        names = {f.name for f in fields(cls)}
        missing, extra = names - set(d), set(d) - names
        if missing or extra:
            raise SettingsError(f"settings keys: missing {sorted(missing)}, "
                                f"unknown {sorted(extra)}")
        kw = {}
        for f in fields(cls):
            v = d[f.name]
            if f.type in (int, "int") and isinstance(v, float) and v.is_integer():
                v = int(v)
            kw[f.name] = v
        s = cls(**kw)
        errs = s.errors()
        if errs:
            raise SettingsError("; ".join(errs))
        return s


def parse_target(text: str) -> Tuple[str, int]:
    """'HOST:PORT' (IPv6: '[HOST]:PORT') -> (host, port), or ValueError saying why."""
    t = str(text).strip()
    eg = f"e.g. 127.0.0.1:{DEFAULT_PORT} or [::1]:{DEFAULT_PORT}"
    if t.startswith("["):
        host, sep, port = t[1:].partition("]:")
    else:
        host, sep, port = t.rpartition(":")
        if ":" in host:
            raise ValueError(f"{t!r}: put an IPv6 address in brackets ({eg})")
    if not sep or not host:
        raise ValueError(f"{t!r} is not HOST:PORT ({eg})")
    if not (port.isascii() and port.isdigit()):
        raise ValueError(f"{t!r}: port {port!r} is not a number ({eg})")
    errs = RobotOutputSettings(host, int(port)).errors()
    if errs:
        raise ValueError("; ".join(errs))
    return host, int(port)


def load_settings(store) -> Tuple[Optional[RobotOutputSettings], str]:
    """(stored settings or None, note) from a QSettings-like store (value / setValue).
    Missing -> (None, ""); unusable -> (None, why)."""
    raw = store.value(STORE_KEY)
    if raw is None:
        return None, ""
    try:
        schema = int(store.value(SCHEMA_KEY))
        if schema != SETTINGS_SCHEMA:
            raise ValueError(f"schema {schema}, expected {SETTINGS_SCHEMA}")
        return RobotOutputSettings.from_dict(json.loads(raw)), ""
    except (TypeError, ValueError) as e:
        return None, f"Stored robot output settings ignored ({e}); defaults shown."


def save_settings(store, settings: RobotOutputSettings):
    """Target and rate only: the armed state is never stored."""
    errs = settings.errors()
    if errs:
        raise SettingsError("; ".join(errs))
    store.setValue(SCHEMA_KEY, SETTINGS_SCHEMA)
    store.setValue(STORE_KEY, json.dumps(settings.to_dict(), sort_keys=True))
    sync = getattr(store, "sync", None)
    if sync is not None:
        sync()


# -- datagrams ------------------------------------------------------------------------

def _number(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v)


def _whole(name: str, v, hi: int = SEQ_MAX) -> int:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, bool) or not isinstance(v, numbers.Integral) or not 0 <= v <= hi:
        raise ValueError(f"{name} must be a whole number 0..{hi}, got {v!r}")
    return int(v)


def _level(s) -> float:
    if _number(s):
        for level in LEVELS:
            if abs(s - level) <= 1e-9:
                return level
    raise ValueError(f"s must be 0, 0.5, 1 or None (no valid command), got {s!r}")


def _text(name: str, v, n: int) -> str:
    if not isinstance(v, str) or len(v) > n:
        raise ValueError(f"{name} must be a string of at most {n} characters, got {v!r}")
    return v


def encode_command(*, session: str, seq: int, kind: str, s: Optional[float], stabilized: bool,
                   simulated: bool, sent_wall: float, sent_mono: float,
                   stab_ms: Optional[dict] = None, age_ms: Optional[float] = None,
                   status: Optional[str] = None, reason: Optional[str] = None,
                   run_id: Optional[str] = None, revision: Optional[int] = None,
                   accepted_pred_id: Optional[int] = None,
                   last_pred_id: Optional[int] = None) -> bytes:
    """One emg-stiffness/1 datagram (UTF-8 JSON, <= 2048 bytes). s None = no valid command:
    sent as null with valid false, never 0. Every rule of the contract is checked here, so
    a datagram the robot would reject or misread is a ValueError instead."""
    if not isinstance(session, str) or not _SESSION_ID.fullmatch(session):
        raise ValueError(f"session must be 8-64 characters [A-Za-z0-9_-], got {session!r}")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    level = None if s is None else _level(s)
    if kind == STOP and level is not None:
        raise ValueError("a 'stop' datagram carries no level (s None)")
    for name, v in (("stabilized", stabilized), ("simulated", simulated)):
        if not isinstance(v, bool):
            raise ValueError(f"{name} must be True/False, got {v!r}")
    for name, v in (("sent_wall", sent_wall), ("sent_mono", sent_mono)):
        if not _number(v):
            raise ValueError(f"{name} must be a finite number, got {v!r}")
    msg = {"proto": PROTOCOL, "source": SOURCE, "session": session, "seq": _whole("seq", seq),
           "kind": kind, "s": level, "valid": level is not None, "stabilized": stabilized,
           "simulated": simulated, "sent_wall": float(sent_wall), "sent_mono": float(sent_mono)}
    if stab_ms is not None:
        if not isinstance(stab_ms, dict) or set(stab_ms) != {"up", "down", "startup"}:
            raise ValueError(f"stab_ms must have exactly up / down / startup, got {stab_ms!r}")
        msg["stab_ms"] = {k: _whole(f"stab_ms.{k}", stab_ms[k])
                          for k in ("up", "down", "startup")}
    elif stabilized:
        raise ValueError("stab_ms (up / down / startup confirmation ms) is required when "
                         "stabilized")
    if age_ms is not None and not (_number(age_ms) and age_ms >= 0):
        raise ValueError(f"age_ms must be a finite number >= 0 or None, got {age_ms!r}")
    msg["age_ms"] = None if age_ms is None else round(float(age_ms), 1) + 0.0   # no -0.0
    if status is not None:
        msg["status"] = _text("status", status, 64)
    if reason is not None:
        msg["reason"] = _text("reason", reason, 200)
    msg["run_id"] = None if run_id is None else _text("run_id", run_id, 64)
    for name, v in (("revision", revision), ("accepted_pred_id", accepted_pred_id),
                    ("last_pred_id", last_pred_id)):
        msg[name] = None if v is None else _whole(name, v)
    data = json.dumps(msg, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")
    if len(data) > MAX_DATAGRAM_BYTES:
        raise ValueError(f"datagram is {len(data)} bytes, limit {MAX_DATAGRAM_BYTES}")
    return data


def _why(snap, st) -> str:
    """Why there is no valid command (diagnostics only)."""
    if snap.state != RUNNING:
        return f"{snap.state}: {snap.reason}" if snap.reason else str(snap.state)
    if st is None:
        return "no stabilizer state"
    if st.accepted is not None:
        return "no fresh evidence"
    if st.status == OFF:
        return st.reason or "no valid prediction yet"
    return describe(st)


def command_fields(snap, now: float) -> dict:
    """encode_command() keywords for the output shown in `snap` (a LiveSnapshot) at `now`:
    s = snap.stable(now).s, i.e. stabilized, or raw while stabilization is OFF; else None."""
    st = snap.stabilization
    shown = snap.stable(now)
    s = None if shown is None else shown.s
    age = None
    if st is not None and st.last_evidence_mono is not None:
        a = (now - st.last_evidence_mono) * 1000
        age = max(0.0, a) if math.isfinite(a) else None
    run_id = snap.run_id
    f = {"s": s, "stabilized": bool(st is not None and st.enabled), "age_ms": age,
         "reason": "" if s is not None else _why(snap, st)[:200],
         "run_id": run_id if isinstance(run_id, str) and len(run_id) <= 64 else None,
         "revision": None if st is None else st.revision,
         "accepted_pred_id": None if s is None else st.accepted_pred_id,
         "last_pred_id": None if st is None else st.last_pred_id}
    if st is not None:
        f["status"] = str(st.status)[:64]
        if st.enabled:
            c = st.settings
            f["stab_ms"] = {"up": c.up_ms, "down": c.down_ms, "startup": c.startup_ms}
    return f


def parse_state(data) -> Optional[dict]:
    """A well-formed emg-stiffness-state/1 datagram as a dict of its listed keys, or None
    (malformed or another protocol: the caller ignores it, nothing is raised). Unknown keys
    are ignored; applied_s / requested_s / target_s are numbers or None."""
    if not isinstance(data, (bytes, bytearray)) or len(data) > MAX_STATE_BYTES:
        return None
    try:
        msg = json.loads(bytes(data).decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    if not isinstance(msg, dict) or msg.get("proto") != STATE_PROTOCOL:
        return None
    out = {}
    for key, kind in _STATE_TYPES.items():
        if key not in msg:
            return None
        v = msg[key]
        if kind is bool:
            ok = isinstance(v, bool)
        elif kind is int:
            ok = isinstance(v, int) and not isinstance(v, bool) and v >= 0
        elif kind is float:
            ok = v is None or _number(v)
        else:
            ok = isinstance(v, str) and len(v) <= (200 if key == "detail" else 64)
        if not ok:
            return None
        out[key] = v
    return out


def _udp_socket(family):
    return socket.socket(family, socket.SOCK_DGRAM)


def _same_peer(addr, target: Tuple[str, int]) -> bool:
    try:
        return (int(addr[1]) == target[1]
                and ipaddress.ip_address(addr[0]) == ipaddress.ip_address(target[0]))
    except (TypeError, ValueError, IndexError):
        return False


# -- sender -----------------------------------------------------------------------------

class _Session:
    """One armed period. GUI thread: stop, reason. Sender thread: everything else. Fields
    read by stats() change under RobotOutput._lock."""

    def __init__(self, sid: str, settings: RobotOutputSettings, sock, local, prev):
        self.id, self.settings, self.sock, self.local = sid, settings, sock, local
        self.addr = (settings.host, settings.port)
        self.prev = prev                          # previous session's thread (its 'stop' first)
        self.stop = threading.Event()
        self.reason = ""
        self.seq = 0                              # next seq: the sender thread only
        self.sent = self.errors = 0
        self.last_error = ""
        self.last = None                          # (s, stabilized) of the last value datagram
        self.last_s, self.last_kind = None, ""
        self.ended = False
        self.state, self.state_mono = None, None  # newest robot state (parse_state) + receipt
        self.state_counts = collections.Counter()
        self.reported = set()                     # error kinds already printed once


class RobotOutput:
    """UDP sender for one LiveEngine. arm / disarm / configure / stats / close are for the
    GUI thread and never block, except close(timeout). The sender thread only reads the
    engine through snapshot(); it never touches the engine's state or any widget."""

    def __init__(self, engine, settings: RobotOutputSettings, *, simulated: bool,
                 clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time,
                 socket_factory: Optional[Callable] = None):
        errs = settings.errors()
        if errs:
            raise ValueError("; ".join(errs))
        self.engine = engine
        self.simulated = bool(simulated)          # synthetic EMG: the robot ignores it by default
        self._settings = settings
        self._clock, self._wall = clock, wall
        self._socket_factory = socket_factory or _udp_socket
        self._lock = threading.Lock()
        self._sess: Optional[_Session] = None
        self._threads: List[threading.Thread] = []
        self._closed = False
        self.sessions = 0

    # -- GUI thread ------------------------------------------------------------------
    @property
    def settings(self) -> RobotOutputSettings:
        return self._settings

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._sess is not None and not self._sess.stop.is_set()

    @property
    def session(self) -> Optional[str]:
        """Id of the armed session, else None."""
        with self._lock:
            s = self._sess
            return s.id if s is not None and not s.stop.is_set() else None

    @property
    def threads(self) -> List[threading.Thread]:
        with self._lock:
            return [t for t in self._threads if t.is_alive()]

    def configure(self, settings: RobotOutputSettings):
        """Target / rate for the next arm(). ValueError if invalid, RuntimeError while armed."""
        errs = settings.errors()
        if errs:
            raise ValueError("; ".join(errs))
        with self._lock:
            if self._sess is not None and not self._sess.stop.is_set():
                raise RuntimeError("disarm the robot output before changing its target")
            self._settings = settings

    def arm(self) -> str:
        """Start a new session (new id, seq 0) on its own sender thread and return its id;
        the id of the current one if already armed. OSError if the UDP socket cannot be
        opened (nothing started), RuntimeError after close()."""
        with self._lock:
            if self._closed:
                raise RuntimeError("robot output is closed")
            if self._sess is not None and not self._sess.stop.is_set():
                return self._sess.id
            cfg = self._settings
            sock, local = self._open(cfg)
            alive = [t for t in self._threads if t.is_alive()]
            sess = _Session(uuid.uuid4().hex[:12], cfg, sock, local,
                            alive[-1] if alive else None)
            t = threading.Thread(target=self._run, args=(sess,), name=THREAD_NAME, daemon=True)
            try:
                t.start()
            except RuntimeError:                  # cannot start a thread: nothing armed
                sock.close()
                raise
            self._sess = sess
            self._threads = alive + [t]
            self.sessions += 1
            return sess.id

    def disarm(self, reason: str = "disarmed") -> bool:
        """Signal the sender: it sends kind 'stop' (reason included) as its last datagram and
        exits. Never blocks. False if nothing was armed."""
        with self._lock:
            s = self._sess
            if s is None or s.stop.is_set():
                return False
            s.reason = str(reason)[:200] or "disarmed"
            s.stop.set()
            return True

    def close(self, timeout: float = 1.0, reason: str = "shutdown") -> bool:
        """Disarm (the sender still sends its 'stop'), refuse further arm(), then wait at
        most `timeout` s in total for the sender threads. Idempotent; the only call that
        waits. True if no sender thread is left."""
        with self._lock:
            self._closed = True
        self.disarm(reason)
        with self._lock:
            threads = list(self._threads)
        end = time.monotonic() + max(0.0, float(timeout))
        for t in threads:
            t.join(max(0.0, end - time.monotonic()))
        with self._lock:
            self._threads = [t for t in self._threads if t.is_alive()]
            return not self._threads

    def stats(self) -> dict:
        """Counters of the current (or last) session for display; robot = the newest robot
        state if one arrived within STATE_TIMEOUT_S, else None ("robot state unknown")."""
        now = self._clock()
        with self._lock:
            s = self._sess
            out = {"armed": False, "active": False, "session": None,
                   "target": self._settings.target, "local": None, "sent": 0, "errors": 0,
                   "last_error": "", "last_s": None, "last_kind": "", "seq": 0,
                   "stop_reason": "", "robot": None, "robot_age_s": None,
                   "back_channel": {}, "simulated": self.simulated, "sessions": self.sessions}
            if s is None:
                return out
            age = None if s.state_mono is None else now - s.state_mono
            out.update(armed=not s.stop.is_set(), active=not s.ended, session=s.id,
                       target=s.settings.target, local=s.local, sent=s.sent, errors=s.errors,
                       last_error=s.last_error, last_s=s.last_s, last_kind=s.last_kind,
                       seq=s.seq, stop_reason=s.reason, robot_age_s=age,
                       robot=dict(s.state) if age is not None and age <= STATE_TIMEOUT_S
                       else None, back_channel=dict(s.state_counts))
            return out

    # -- sender thread ---------------------------------------------------------------
    def _open(self, cfg: RobotOutputSettings):
        """Non-blocking UDP socket for the target's family. Bound to loopback for a loopback
        target, so nothing from the network reaches the back-channel port."""
        ip = ipaddress.ip_address(cfg.host)
        v6 = ip.version == 6
        sock = self._socket_factory(socket.AF_INET6 if v6 else socket.AF_INET)
        try:
            sock.setblocking(False)
            if hasattr(socket, "SIO_UDP_CONNRESET"):  # Windows: no recvfrom error after an
                sock.ioctl(socket.SIO_UDP_CONNRESET, False)   # ICMP 'port unreachable'
            if ip.is_loopback:
                sock.bind(("::1" if v6 else "127.0.0.1", 0))
            else:
                sock.bind(("::" if v6 else "0.0.0.0", 0))
            local = sock.getsockname()
        except BaseException:
            sock.close()
            raise
        return sock, (str(local[0]), int(local[1]))

    def _run(self, sess: _Session):
        try:
            if sess.prev is not None:
                sess.prev.join(PREV_JOIN_S)       # the old session's 'stop' goes out first
            period = 1.0 / sess.settings.rate_hz
            while not sess.stop.is_set():
                try:
                    out = self._datagram(sess, stop=False)
                    # Disarm before a settings / model change (GUI) + this check: nothing
                    # built from a snapshot taken after the disarm is sent as a value.
                    if out is not None and not sess.stop.is_set():
                        self._send(sess, *out)
                    self._drain(sess)
                except Exception as e:            # a bug: count it, keep the session alive
                    self._error(sess, "sender", e, trace=True)
                sess.stop.wait(period)
            self._send_stop(sess)
        finally:
            try:
                sess.sock.close()
            except OSError:
                pass
            with self._lock:
                sess.ended = True

    def _datagram(self, sess: _Session, stop: bool):
        """(bytes, kind, (s, stabilized)) from a fresh snapshot, or None (error counted)."""
        now = self._clock()
        try:
            f = command_fields(self.engine.snapshot(), now)
        except Exception as e:
            self._error(sess, "snapshot", e, trace=True)
            if not stop:
                return None
            f = {"s": None, "stabilized": False}
        if stop:
            f.update(s=None, accepted_pred_id=None, reason=sess.reason or "disarmed")
            kind = STOP
        else:
            kind = CHANGE if (f["s"], f["stabilized"]) != sess.last else HEARTBEAT
        try:
            data = encode_command(session=sess.id, seq=sess.seq, kind=kind,
                                  simulated=self.simulated, sent_wall=self._wall(),
                                  sent_mono=self._clock(), **f)
        except ValueError as e:
            self._error(sess, "encode", e, trace=True)
            if not stop:
                return None
            f = {"s": None, "stabilized": False, "reason": f["reason"]}   # always encodable
            data = encode_command(session=sess.id, seq=sess.seq, kind=STOP,
                                  simulated=self.simulated, sent_wall=self._wall(),
                                  sent_mono=self._clock(), **f)
        return data, kind, (f["s"], f["stabilized"])

    def _send(self, sess: _Session, data: bytes, kind: str, key) -> bool:
        try:
            sess.sock.sendto(data, sess.addr)
        except OSError as e:                      # incl. a full send buffer: never raised
            self._error(sess, "send", e)
            return False
        with self._lock:
            sess.seq += 1                         # only datagrams handed to the network
            sess.sent += 1
            sess.last_kind = kind
            sess.last_s = key[0]
            if kind != STOP:
                sess.last = key
        return True

    def _send_stop(self, sess: _Session):
        out = self._datagram(sess, stop=True)
        for _ in range(STOP_TRIES):
            if self._send(sess, *out):
                return
            time.sleep(0.01)                      # the robot also times out by itself

    def _drain(self, sess: _Session):
        """Read the robot's state datagrams without blocking; keep the newest of this session."""
        for _ in range(DRAIN_MAX):
            try:
                data, addr = sess.sock.recvfrom(65536)
            except (BlockingIOError, InterruptedError):
                return
            except OSError as e:                  # e.g. an ICMP report of an earlier send
                self._error(sess, "receive", e)   # (nothing listens there): count, go on
                continue
            now = self._clock()
            peer = _same_peer(addr, sess.addr)
            st = parse_state(data) if peer else None
            with self._lock:
                c = sess.state_counts
                if not peer:
                    c["foreign"] += 1             # not from the target address: never shown
                elif st is None:
                    c["malformed"] += 1
                elif st["session"] != sess.id:
                    c["other_session"] += 1
                elif (sess.state is not None and st["seq"] <= sess.state["seq"]
                      and now - sess.state_mono <= STATE_TIMEOUT_S):
                    c["out_of_order"] += 1
                else:
                    c["received"] += 1
                    sess.state, sess.state_mono = st, now

    def _error(self, sess: _Session, what: str, e: BaseException, trace: bool = False):
        with self._lock:
            sess.errors += 1
            sess.last_error = f"{what}: {type(e).__name__}: {e}"
            first = what not in sess.reported
            sess.reported.add(what)
        if first:                                 # once per kind and session: no log flood
            if trace:
                traceback.print_exc()
            else:
                print(f"[robot output] {sess.settings.target}: {what} failed: "
                      f"{type(e).__name__}: {e}")
