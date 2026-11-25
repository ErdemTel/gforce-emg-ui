"""Robot output (live/robot_out.py): SOFTWARE tests with test-double classifiers, fake clocks
and loopback UDP. Nothing here talks to a robot. The golden vectors are the contract shared
with the robot bridge (parent repository: docs/protocol/emg-stiffness-v1-golden.json and
docs/design_emg_stiffness_2026-11-02.md section 5); the checker below reads them
independently of the encoder, so the encoder is tested against the rules, not itself."""

import errno
import json
import math
import re
import socket
import threading
import time
from pathlib import Path

import pytest

from conftest import wait_until
from gforce_emg_ui.live import robot_out
from gforce_emg_ui.live.robot_out import (CHANGE, DEFAULT_PORT, HEARTBEAT, PROTOCOL, SCHEMA_KEY,
                                          SETTINGS_SCHEMA, STATE_PROTOCOL, STOP, STORE_KEY,
                                          THREAD_NAME, RobotOutput, RobotOutputSettings,
                                          SettingsError, command_fields, encode_command,
                                          load_settings, parse_state, parse_target,
                                          save_settings)
from gforce_emg_ui.live.stabilization import StabilizerSettings
from test_live_stabilization import ON, Rig

GOLDEN_PATH = Path(__file__).parent / "data" / "emg-stiffness-v1-golden.json"
PARENT_GOLDEN = Path(__file__).resolve().parents[2] / "docs/protocol/emg-stiffness-v1-golden.json"
GOLDEN = json.loads(GOLDEN_PATH.read_text())
SESSION_ID = re.compile(r"[A-Za-z0-9_-]{8,64}")
STAB = {"up": 100, "down": 100, "startup": 100}


# -- independent reading of the golden rules ------------------------------------------------

class Reject(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def check(raw: bytes) -> dict:
    """What a receiver must decide for one datagram: the golden 'expect' fields, or
    Reject(reason) with the golden reason code."""
    if len(raw) > GOLDEN["max_datagram_bytes"]:
        raise Reject("size")
    try:
        m = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise Reject("json")
    if not isinstance(m, dict):
        raise Reject("json")
    if m.get("proto") != PROTOCOL:
        raise Reject("proto")
    rules = {
        "source": lambda v: isinstance(v, str),
        "session": lambda v: isinstance(v, str) and SESSION_ID.fullmatch(v) is not None,
        "seq": lambda v: _int(v) and 0 <= v <= 2 ** 53,
        "kind": lambda v: v in ("heartbeat", "change", "stop"),
        "s": lambda v: v is None or _num(v),
        "valid": lambda v: isinstance(v, bool),
        "stabilized": lambda v: isinstance(v, bool),
        "simulated": lambda v: isinstance(v, bool),
        "sent_wall": _num,
        "sent_mono": _num,
    }
    optional = {
        "stab_ms": lambda v: isinstance(v, dict) and all(
            _int(v.get(k)) and v[k] >= 0 for k in ("up", "down", "startup")),
        "age_ms": lambda v: v is None or (_num(v) and v >= 0),
        "status": lambda v: isinstance(v, str) and len(v) <= 64,
        "reason": lambda v: isinstance(v, str) and len(v) <= 200,
        "run_id": lambda v: v is None or (isinstance(v, str) and len(v) <= 64),
        "revision": lambda v: v is None or _int(v),
        "accepted_pred_id": lambda v: v is None or _int(v),
        "last_pred_id": lambda v: v is None or _int(v),
    }
    for k, ok in rules.items():
        if k not in m or not ok(m[k]):
            raise Reject("type")
    for k, ok in optional.items():
        if k in m and not ok(m[k]):
            raise Reject("type")
    if m["stabilized"] and "stab_ms" not in m:
        raise Reject("type")
    if m["source"] not in ("gforce-emg-ui", "send_stiffness"):
        raise Reject("source")
    s = m["s"]
    if s is not None:
        near = [level for level in GOLDEN["levels"] if abs(s - level) <= 1e-9]
        if not near:
            raise Reject("level")
        s = float(near[0])
    if m["valid"] != (s is not None) or (m["kind"] == "stop" and m["valid"]):
        raise Reject("consistency")
    return {"s": s, "valid": m["valid"], "kind": m["kind"], "stabilized": m["stabilized"],
            "simulated": m["simulated"]}


def raw_vector(v) -> bytes:
    if "datagram" in v:
        return json.dumps(v["datagram"]).encode()
    if "utf8" in v:
        return v["utf8"].encode()
    if "hex" in v:
        return bytes.fromhex(v["hex"].replace(" ", ""))
    rep = v["utf8_repeat"]
    return (rep["unit"] * rep["count"]).encode()


def encode(**kw) -> bytes:
    base = dict(session="a1b2c3d4e5f6", seq=0, kind=CHANGE, s=0.5, stabilized=True,
                simulated=False, sent_wall=1793630000.125, sent_mono=5021.375, stab_ms=STAB)
    base.update(kw)
    return encode_command(**base)


# -- contract --------------------------------------------------------------------------------

@pytest.mark.skipif(not PARENT_GOLDEN.is_file(), reason="parent repository not checked out")
def test_golden_copy_is_identical_to_the_parent_repository():
    assert GOLDEN_PATH.read_bytes() == PARENT_GOLDEN.read_bytes()


def test_the_checker_reproduces_every_golden_vector():
    assert GOLDEN["max_datagram_bytes"] == robot_out.MAX_DATAGRAM_BYTES
    assert tuple(GOLDEN["levels"]) == robot_out.LEVELS
    for v in GOLDEN["valid"]:
        assert check(raw_vector(v)) == v["expect"], v["name"]
    for v in GOLDEN["invalid"] + GOLDEN["invalid_raw"]:
        with pytest.raises(Reject) as e:
            check(raw_vector(v))
        assert e.value.reason == v["reason"], v["name"]
        assert v["reason"] in GOLDEN["reason_codes"]


def test_encoder_reproduces_the_golden_ui_datagrams():
    """Same keys and values as the golden UI datagrams; added keys are only null optionals."""
    allowed = {"age_ms", "run_id", "revision", "accepted_pred_id", "last_pred_id"}
    for v in GOLDEN["valid"]:
        d = v["datagram"]
        if d["source"] != robot_out.SOURCE:
            continue
        kw = {k: d[k] for k in ("session", "seq", "kind", "s", "stabilized", "simulated",
                                "sent_wall", "sent_mono", "stab_ms", "age_ms", "status",
                                "reason", "run_id", "revision", "accepted_pred_id",
                                "last_pred_id") if k in d}
        out = json.loads(encode_command(**kw))
        for k, value in d.items():
            if k == "future_field":
                continue
            assert out[k] == value and type(out[k]) is type(value) or k == "s", (v["name"], k)
        want = v["expect"]["s"]
        assert out["s"] is None if want is None else (out["s"] == want
                                                      and isinstance(out["s"], float)), v["name"]
        assert all(out[k] is None for k in set(out) - set(d) - {"future_field"}), v["name"]
        assert set(out) - set(d) <= allowed


@pytest.mark.parametrize("kind", [HEARTBEAT, CHANGE, STOP])
@pytest.mark.parametrize("s", [None, 0.0, 0.5, 1.0, 0, 1, 0.5 + 1e-10])
@pytest.mark.parametrize("stabilized", [True, False])
@pytest.mark.parametrize("simulated", [True, False])
def test_every_encoder_output_conforms_to_the_golden_rules(kind, s, stabilized, simulated):
    if kind == STOP and s is not None:
        with pytest.raises(ValueError):
            encode(kind=kind, s=s)
        return
    raw = encode(kind=kind, s=s, stabilized=stabilized, simulated=simulated,
                 stab_ms=STAB if stabilized else None)
    got = check(raw)
    assert got == {"s": None if s is None else float(round(s * 2) / 2), "valid": s is not None,
                   "kind": kind, "stabilized": stabilized, "simulated": simulated}
    m = json.loads(raw)
    assert m["source"] == "gforce-emg-ui" and ("stab_ms" in m) == stabilized
    if s is None:
        assert m["s"] is None and b'"s":null' in raw and b'"s":0' not in raw


def test_encoder_extremes_still_conform():
    for kw in (dict(seq=2 ** 53, reason="ü" * 200, status="S" * 64, run_id="r" * 64,
                    age_ms=0.0, revision=2 ** 53, accepted_pred_id=0, last_pred_id=2 ** 53,
                    session="x" * 64),
               dict(seq=0, session="-_aZ09-_", stab_ms={"up": 0, "down": 0, "startup": 0},
                    age_ms=1e9, sent_wall=0.0, sent_mono=-5.0),
               dict(age_ms=-0.0, s=None, kind=STOP, reason="")):
        raw = encode(**kw)
        assert len(raw) <= GOLDEN["max_datagram_bytes"]
        check(raw)
        assert b"-0.0" not in raw                          # age -0.0 is written as 0.0


@pytest.mark.parametrize("bad", [
    dict(s=0.75), dict(s=2.0), dict(s=True), dict(s="0.5"), dict(s=float("nan")),
    dict(kind=STOP, s=0.5), dict(kind="pulse"), dict(stab_ms=None),
    dict(stab_ms={"up": -1, "down": 100, "startup": 100}), dict(stab_ms={"up": 100}),
    dict(stab_ms={**STAB, "extra": 1}), dict(stab_ms={**STAB, "up": True}),
    dict(stab_ms={**STAB, "up": 100.5}), dict(age_ms=-5), dict(age_ms=float("nan")),
    dict(age_ms="1"), dict(reason="x" * 201), dict(reason=3), dict(status="S" * 65),
    dict(run_id="r" * 65), dict(run_id=5), dict(session="abc"), dict(session="abc def/ghi"),
    dict(session="x" * 65), dict(seq=-1), dict(seq=1.5), dict(seq=True), dict(seq=2 ** 53 + 1),
    dict(stabilized=1), dict(simulated=None), dict(sent_mono="NaN"),
    dict(sent_wall=float("inf")), dict(revision=-1), dict(accepted_pred_id=True),
    dict(last_pred_id=1.5)])
def test_encoder_refuses_what_a_receiver_would_reject(bad):
    with pytest.raises(ValueError):
        encode(**bad)


# -- the shown output -> datagram fields -------------------------------------------------------

def test_invalid_output_is_null_never_zero_in_every_state():
    rig = Rig(ON, warm="low")
    try:
        f = command_fields(rig.eng.snapshot(), 0.95)       # warming up: no window yet
        assert f["s"] is None and f["reason"].startswith("WARMING_UP")
        rig.push("low", 1.0)                               # LOW confirming (startup 100 ms)
        f = command_fields(rig.eng.snapshot(), 1.0)
        assert f["s"] is None and f["reason"].startswith("Confirming initial class: LOW")
        rig.push("low", 1.1)
        f = command_fields(rig.eng.snapshot(), 1.1)
        assert f["s"] == 0.0 and f["reason"] == "" and f["accepted_pred_id"] == 2
        f = command_fields(rig.eng.snapshot(), 1.7)        # read-time freshness: stale
        assert f["s"] is None and f["reason"] == "no fresh evidence"
        assert f["accepted_pred_id"] is None and f["age_ms"] == pytest.approx(600)
        rig.push("low", 1.15, bad=True)                    # invalid prediction
        f = command_fields(rig.eng.snapshot(), 1.15)
        assert f["s"] is None and f["reason"].startswith("INVALID_INPUT")
        rig.eng.stop("user_stop")
        f = command_fields(rig.eng.snapshot(), 1.15)
        assert f["s"] is None and f["reason"] == "STOPPED: user_stop"
        for kind in (HEARTBEAT, CHANGE, STOP):
            raw = encode(kind=kind, **f)
            assert b'"s":null' in raw and b'"valid":false' in raw
    finally:
        rig.eng.shutdown()


def test_fields_follow_stabilization_and_carry_join_keys():
    rig = Rig(StabilizerSettings(enabled=True, up_ms=150, down_ms=120, startup_ms=110),
              warm="low")
    try:
        rig.push("low", 1.0)
        rig.push("low", 1.12)
        f = command_fields(rig.eng.snapshot(), 1.3)
        st = rig.st()
        assert f["s"] == 0.0 and f["stabilized"] is True and f["status"] == "HOLDING"
        assert f["stab_ms"] == {"up": 150, "down": 120, "startup": 110}
        assert (f["run_id"], f["revision"]) == ("rig", st.revision)
        assert f["accepted_pred_id"] == st.accepted_pred_id == 2 and f["last_pred_id"] == 2
        assert f["age_ms"] == pytest.approx(180)
        check(encode(**f))
    finally:
        rig.eng.shutdown()
    rig = Rig(warm="low")                                  # OFF: the raw output, unchanged
    try:
        rig.push("low", 1.0)
        rig.push("high", 1.05)
        snap = rig.eng.snapshot()
        f = command_fields(snap, 1.05)
        assert f["s"] == snap.current(1.05).prediction.s_discrete == 1.0
        assert f["stabilized"] is False and "stab_ms" not in f and f["status"] == "OFF"
        check(encode(**f))
    finally:
        rig.eng.shutdown()


# -- settings ---------------------------------------------------------------------------------

class DictStore:
    """QSettings stand-in; like the INI backend it returns stored values as strings."""

    def __init__(self):
        self.d, self.synced = {}, 0

    def value(self, key):
        return self.d.get(key)

    def setValue(self, key, v):
        self.d[key] = v if isinstance(v, str) else str(v)

    def sync(self):
        self.synced += 1


def test_settings_defaults_and_validation():
    d = RobotOutputSettings()
    assert (d.host, d.port, d.rate_hz) == ("127.0.0.1", DEFAULT_PORT, 20) and d.errors() == []
    assert DEFAULT_PORT == 6731 and d.target == "127.0.0.1:6731"
    for host in ("192.168.1.20", "10.0.0.5", "::1", "fe80::1", "2001:db8::7", "127.0.0.2"):
        assert RobotOutputSettings(host).errors() == [], host
    assert RobotOutputSettings("::1", 7000).target == "[::1]:7000"
    bad_hosts = {"localhost": "not an IP address", "robot-pc.local": "not an IP address",
                 "": "IP address", " 127.0.0.1": "not an IP address", "127.1": "not an IP",
                 "224.0.0.1": "multicast", "239.255.0.1": "multicast", "ff02::1": "multicast",
                 "255.255.255.255": "broadcast", "0.0.0.0": "unspecified",
                 "::": "unspecified", "::ffff:224.0.0.5": "multicast",
                 "::ffff:127.0.0.1": "give it as 127.0.0.1"}
    for host, why in bad_hosts.items():
        errs = RobotOutputSettings(host).errors()
        assert len(errs) == 1 and why in errs[0], (host, errs)
    for kw in (dict(port=0), dict(port=65536), dict(port=True), dict(port="6731"),
               dict(port=6731.0), dict(rate_hz=4), dict(rate_hz=51), dict(rate_hz=None)):
        assert RobotOutputSettings(**kw).errors(), kw
    assert RobotOutputSettings(rate_hz=5).errors() == RobotOutputSettings(rate_hz=50).errors() == []
    good = d.to_dict()
    assert good == {"host": "127.0.0.1", "port": 6731, "rate_hz": 20}
    assert RobotOutputSettings.from_dict({**good, "port": 7000.0}).port == 7000
    for broken in ({**good, "host": "localhost"}, {k: v for k, v in good.items() if k != "port"},
                   {**good, "armed": True}, ["not", "a", "dict"], {**good, "rate_hz": 2.5}):
        with pytest.raises(SettingsError):
            RobotOutputSettings.from_dict(broken)


def test_settings_persist_without_the_armed_state():
    store = DictStore()
    assert load_settings(store) == (None, "")
    s = RobotOutputSettings("192.168.1.20", 7000, rate_hz=25)
    save_settings(store, s)
    assert store.synced == 1 and store.d[SCHEMA_KEY] == str(SETTINGS_SCHEMA)
    assert json.loads(store.d[STORE_KEY]) == {"host": "192.168.1.20", "port": 7000, "rate_hz": 25}
    assert load_settings(store) == (s, "")
    with pytest.raises(SettingsError):
        save_settings(store, RobotOutputSettings("localhost"))
    assert load_settings(store) == (s, "")                    # refused: nothing written
    for raw in ("{not json", json.dumps({"host": "127.0.0.1"}), json.dumps([1]),
                json.dumps({**s.to_dict(), "host": "224.0.0.1"})):
        store.d[STORE_KEY] = raw
        loaded, note = load_settings(store)
        assert loaded is None and "ignored" in note, raw
    store.d[STORE_KEY], store.d[SCHEMA_KEY] = json.dumps(s.to_dict()), "99"
    assert load_settings(store)[0] is None and "schema" in load_settings(store)[1]
    del store.d[SCHEMA_KEY]
    assert load_settings(store)[0] is None


def test_parse_target_accepts_ip_literals_with_a_port_only():
    assert parse_target("127.0.0.1:6731") == ("127.0.0.1", 6731)
    assert parse_target(" 192.168.1.20:7000 ") == ("192.168.1.20", 7000)
    assert parse_target("[::1]:6731") == ("::1", 6731)
    for bad, why in (("localhost:6731", "not an IP address"), ("127.0.0.1", "HOST:PORT"),
                     ("127.0.0.1:", "not a number"), ("127.0.0.1:x", "not a number"),
                     ("127.0.0.1:0", "port 0"), ("127.0.0.1:65536", "port 65536"),
                     ("::1:6731", "brackets"), ("[::1]", "HOST:PORT"), (":6731", "HOST:PORT"),
                     ("224.0.0.1:6731", "multicast"), ("0.0.0.0:6731", "unspecified"),
                     ("127.0.0.1:-1", "not a number"), ("127.0.0.1:٣", "not a number")):
        with pytest.raises(ValueError) as e:
            parse_target(bad)
        assert why in str(e.value), (bad, str(e.value))


# -- state back-channel -----------------------------------------------------------------------

def test_parse_state_reads_the_golden_example_and_ignores_malformed():
    ex = GOLDEN["state_example"]
    assert ex["proto"] == STATE_PROTOCOL
    assert parse_state(json.dumps(ex).encode()) == ex
    assert parse_state(json.dumps({**ex, "kp": [1.0] * 22}).encode()) == ex   # unknown keys
    assert parse_state(json.dumps({**ex, "applied_s": None}).encode())["applied_s"] is None
    broken = [b"{not json", b"\xff\xfe\x00{", b"[1, 2]", b"x" * 9000,
              json.dumps({**ex, "proto": "emg-stiffness/1"}).encode(),
              json.dumps({k: v for k, v in ex.items() if k != "phase"}).encode(),
              json.dumps({**ex, "armed": 1}).encode(), json.dumps({**ex, "seq": -1}).encode(),
              json.dumps({**ex, "seq": True}).encode(),
              json.dumps({**ex, "applied_s": "0.5"}).encode(),
              json.dumps({**ex, "applied_s": True}).encode(),
              json.dumps({**ex, "detail": "d" * 201}).encode(),
              json.dumps({**ex, "session": 7}).encode(),
              b'{"proto": "emg-stiffness-state/1", "applied_s": NaN}', "text", None]
    for raw in broken:
        assert parse_state(raw) is None, raw


# -- the sender on loopback -------------------------------------------------------------------

class Receiver:
    """The robot side on loopback: every datagram must pass check(); can answer with state."""

    def __init__(self, family=socket.AF_INET, host="127.0.0.1"):
        self.sock = socket.socket(family, socket.SOCK_DGRAM)
        self.sock.bind((host, 0))
        self.sock.settimeout(0.02)
        self.host, self.port = host, self.sock.getsockname()[1]
        self.msgs, self.addrs, self.times = [], [], []

    def collect(self, cond=lambda msgs: False, timeout=3.0) -> bool:
        end = time.monotonic() + timeout
        while True:
            try:
                raw, addr = self.sock.recvfrom(65536)
            except socket.timeout:
                raw = None
            if raw is not None:
                check(raw)
                self.msgs.append(json.loads(raw))
                self.addrs.append(addr)
                self.times.append(time.monotonic())
            if cond(self.msgs):
                return True
            if time.monotonic() > end:
                return False

    def reply(self, state: dict, addr=None):
        self.sock.sendto(json.dumps(state).encode(), addr or self.addrs[-1])

    def close(self):
        self.sock.close()


def sender_threads():
    return [t for t in threading.enumerate() if t.name == THREAD_NAME]


@pytest.fixture
def rx():
    r = Receiver()
    yield r
    r.close()


@pytest.fixture(autouse=True)
def _no_sender_left():
    yield
    assert wait_until(lambda: not sender_threads(), 3), "a 'live-robot-out' thread leaked"


def accepted_low(settings=ON):
    rig = Rig(settings, warm="low")
    rig.push("low", 1.0)
    rig.push("low", 1.1)
    assert rig.eng.snapshot().stable(1.1).s == 0.0
    return rig


def sender(rig, rx, rate_hz=50, simulated=False, **kw):
    return RobotOutput(rig.eng, RobotOutputSettings(rx.host, rx.port, rate_hz=rate_hz),
                       simulated=simulated, clock=rig.clock, **kw)


def test_disarmed_sends_nothing_and_starts_no_thread(rx):
    rig = accepted_low()
    out = sender(rig, rx)
    try:
        assert not rx.collect(lambda m: m, timeout=0.3) and rx.msgs == []
        assert not sender_threads() and not out.armed and out.session is None
        st = out.stats()
        assert st["session"] is None and st["sent"] == 0 and st["robot"] is None
        assert not out.disarm("nothing armed")
    finally:
        assert out.close(1.0)
        rig.eng.shutdown()
    assert rx.msgs == []


@pytest.mark.parametrize("rate_hz", [20, 50])
def test_armed_session_heartbeats_at_rate_with_consecutive_seq(rx, rate_hz):
    rig = accepted_low()
    out = sender(rig, rx, rate_hz=rate_hz)
    try:
        sid = out.arm()
        assert out.arm() == sid and out.armed                 # arming twice: same session
        assert SESSION_ID.fullmatch(sid) and len(sid) >= 8
        n = int(rate_hz * 0.6)
        assert rx.collect(lambda m: len(m) >= n)
        m = rx.msgs
        assert [x["seq"] for x in m] == list(range(len(m)))
        assert {x["session"] for x in m} == {sid}
        assert m[0]["kind"] == CHANGE and all(x["kind"] == HEARTBEAT for x in m[1:])
        assert all(x["s"] == 0.0 and x["valid"] and x["stabilized"] and not x["simulated"]
                   and x["stab_ms"] == STAB and x["run_id"] == "rig" for x in m)
        rate = (len(m) - 1) / (m[-1]["sent_wall"] - m[0]["sent_wall"])
        assert 0.6 * rate_hz <= rate <= 1.05 * rate_hz, rate
        assert wait_until(lambda: out.stats()["sent"] >= n, 1)   # counted just after sendto()
        st = out.stats()
        assert st["armed"] and st["errors"] == 0 and st["last_s"] == 0.0
        assert st["target"] == f"127.0.0.1:{rx.port}" and st["local"][0] == "127.0.0.1"
    finally:
        assert out.close(1.0)
        rig.eng.shutdown()


def test_an_accepted_change_is_the_next_datagram_and_kind_change(rx):
    rig = accepted_low()
    out = sender(rig, rx)
    try:
        out.arm()
        assert rx.collect(lambda m: len(m) >= 3)
        rig.push("medium", 1.2)                               # pending, still LOW
        rig.push("medium", 1.3)                               # MEDIUM accepted (up 100 ms)
        t0 = time.monotonic()
        assert rig.eng.snapshot().stable(1.3).s == 0.5
        assert rx.collect(lambda m: m[-1]["s"] == 0.5)
        first = next(i for i, x in enumerate(rx.msgs) if x["s"] == 0.5)
        assert rx.times[first] - t0 < 0.3                     # within one period (+ slack)
        assert rx.msgs[first]["kind"] == CHANGE
        assert rx.msgs[first]["accepted_pred_id"] == rig.st().accepted_pred_id
        assert rx.collect(lambda m: len(m) >= first + 4)
        assert all(x["kind"] == HEARTBEAT and x["s"] == 0.5 for x in rx.msgs[first + 1:])
        assert all(x["kind"] == HEARTBEAT for x in rx.msgs[1:first])
    finally:
        assert out.close(1.0)
        rig.eng.shutdown()


def test_invalid_output_is_sent_as_null_never_zero(rx):
    rig = Rig(ON, warm="low")
    out = sender(rig, rx)

    def phase(s, n=4):
        """The next datagram with this s is a change, and the following ones keep it."""
        k = len(rx.msgs)
        assert rx.collect(lambda m: any(x["s"] == s for x in m[k:]))
        i = next(i for i in range(k, len(rx.msgs)) if rx.msgs[i]["s"] == s)
        assert rx.msgs[i]["kind"] == CHANGE
        assert rx.collect(lambda m: len(m) >= i + n)
        assert all(x["s"] == s for x in rx.msgs[i:])
    try:
        out.arm()
        phase(None)                                           # warming up: null, not 0
        assert rx.msgs[0]["reason"].startswith("WARMING_UP")
        rig.push("low", 1.0)
        rig.push("low", 1.1)
        phase(0.0)
        rig.push("low", 1.15, bad=True)                       # invalid prediction
        phase(None)
        rig.push("low", 1.2)
        rig.push("low", 1.3)
        phase(0.0)
        rig.clock.t = 1.81                                    # evidence older than stale_s
        phase(None)
        assert rx.msgs[-1]["reason"] == "no fresh evidence"
        for x in rx.msgs:
            assert x["valid"] == (x["s"] is not None)
            assert x["s"] is None or x["status"] in ("HOLDING", "PENDING")
            assert x["s"] is not None or x["reason"]          # always says why
    finally:
        assert out.close(1.0)
        rig.eng.shutdown()


def test_disarm_only_signals_and_stop_is_the_last_datagram(rx):
    rig = accepted_low()
    out = sender(rig, rx)
    try:
        out.arm()
        assert rx.collect(lambda m: len(m) >= 3)
        t0 = time.perf_counter()
        assert out.disarm("operator")
        assert time.perf_counter() - t0 < 0.05 and not out.armed
        assert wait_until(lambda: not out.threads, 2)
        assert rx.collect(lambda m: m[-1]["kind"] == STOP)
        assert not rx.collect(lambda m: m[-1]["kind"] != STOP, timeout=0.25)   # nothing after
        stop = rx.msgs[-1]
        assert stop["s"] is None and stop["valid"] is False and stop["reason"] == "operator"
        assert stop["stabilized"] and stop["stab_ms"] == STAB   # still well-formed when ON
        assert [x["kind"] for x in rx.msgs].count(STOP) == 1
        assert [x["seq"] for x in rx.msgs] == list(range(len(rx.msgs)))
        st = out.stats()
        assert not st["armed"] and not st["active"] and st["last_kind"] == STOP
        assert st["last_s"] is None and st["stop_reason"] == "operator"
        assert not out.disarm("again")
    finally:
        assert out.close(1.0)
        rig.eng.shutdown()


def test_rearm_starts_a_new_session_after_the_old_stop(rx):
    rig = accepted_low()
    out = sender(rig, rx)
    try:
        a = out.arm()
        assert rx.collect(lambda m: len(m) >= 2)
        out.disarm("operator")
        b = out.arm()                                         # old thread may still run
        assert a != b and SESSION_ID.fullmatch(b)
        assert rx.collect(lambda m: sum(x["session"] == b for x in m) >= 4)
        sessions = [x["session"] for x in rx.msgs]
        first_b = sessions.index(b)
        assert rx.msgs[first_b - 1] == next(x for x in rx.msgs if x["kind"] == STOP)
        assert set(sessions[:first_b]) == {a} and set(sessions[first_b:]) == {b}
        new = [x for x in rx.msgs if x["session"] == b]
        assert [x["seq"] for x in new] == list(range(len(new))) and new[0]["kind"] == CHANGE
        assert out.stats()["sessions"] == 2 and out.session == b
    finally:
        assert out.close(1.0)
        rig.eng.shutdown()
    assert rx.collect(lambda m: m[-1]["kind"] == STOP and m[-1]["session"] == b)
    assert rx.msgs[-1]["reason"] == "shutdown"


class FailingSocket:
    """A real loopback socket whose sendto always fails (unreachable network, full buffer)."""

    def __init__(self, family, error):
        self._sock = socket.socket(family, socket.SOCK_DGRAM)
        self.error = error
        self.attempts = 0

    def sendto(self, data, addr):
        self.attempts += 1
        raise self.error

    def __getattr__(self, name):
        return getattr(self._sock, name)


class RecordingSocket(FailingSocket):
    """A real socket that records the sending thread and fails the first `fail` sends."""

    def __init__(self, family, fail=0):
        super().__init__(family, BlockingIOError(errno.EAGAIN, "send buffer full"))
        self.fail, self.threads, self.kinds = fail, [], []

    def sendto(self, data, addr):
        self.threads.append(threading.current_thread().name)
        self.kinds.append(json.loads(data)["kind"])
        if self.fail > 0:
            self.fail -= 1
            raise self.error
        return self._sock.sendto(data, addr)


def test_every_datagram_is_sent_by_the_sender_thread_and_a_failed_change_is_resent(rx):
    rig = accepted_low()
    socks = []

    def factory(family):
        socks.append(RecordingSocket(family, fail=3))
        return socks[-1]
    out = sender(rig, rx, socket_factory=factory)
    try:
        out.arm()
        assert rx.collect(lambda m: len(m) >= 3)
        assert socks[0].kinds[:4] == [CHANGE] * 4                # not sent: still a change
        first = rx.msgs[0]
        assert first["kind"] == CHANGE and first["seq"] == 0 and first["s"] == 0.0
        rig.push("medium", 1.2)
        rig.push("medium", 1.3)                                  # accepted from the test thread
        assert rx.collect(lambda m: m[-1]["s"] == 0.5)
        out.disarm("operator")                                   # GUI-thread call: signal only
        assert rx.collect(lambda m: m[-1]["kind"] == STOP)
        assert wait_until(lambda: not out.threads, 2)
        assert set(socks[0].threads) == {THREAD_NAME}            # incl. 'stop': one writer
        assert [x["seq"] for x in rx.msgs] == list(range(len(rx.msgs)))
        assert out.stats()["errors"] == 3 and out.stats()["sent"] == len(rx.msgs)
    finally:
        assert out.close(1.0)
        rig.eng.shutdown()


@pytest.mark.parametrize("error", [OSError(errno.ENETUNREACH, "Network is unreachable"),
                                   BlockingIOError(errno.EAGAIN, "send buffer full")])
def test_send_errors_are_counted_never_raised(rx, error, capsys):
    rig = accepted_low()
    socks = []

    def factory(family):
        socks.append(FailingSocket(family, error))
        return socks[-1]
    out = sender(rig, rx, socket_factory=factory)
    try:
        out.arm()
        assert wait_until(lambda: out.stats()["errors"] >= 5, 3)
        st = out.stats()
        assert st["armed"] and st["active"] and st["sent"] == 0 and st["seq"] == 0
        assert type(error).__name__ in st["last_error"] and len(out.threads) == 1
        assert out.disarm("operator")
        assert wait_until(lambda: not out.threads, 2)        # the failing 'stop' is retried
        assert socks[0].attempts >= 5 + robot_out.STOP_TRIES
        assert rx.msgs == [] and not rx.collect(lambda m: m, timeout=0.1)
        assert capsys.readouterr().out.count("send failed") == 1   # printed once per session
    finally:
        assert out.close(1.0)
        rig.eng.shutdown()


def test_socket_open_failure_raises_on_arm_and_arms_nothing(rx):
    rig = accepted_low()

    def factory(family):
        raise OSError(errno.EMFILE, "Too many open files")
    out = sender(rig, rx, socket_factory=factory)
    try:
        with pytest.raises(OSError):
            out.arm()
        assert not out.armed and not out.threads and out.stats()["session"] is None
    finally:
        assert out.close(1.0)
        rig.eng.shutdown()


def test_back_channel_state_is_checked_shown_and_expires(rx):
    rig = accepted_low()
    out = sender(rig, rx)
    foreign = Receiver()
    try:
        sid = out.arm()
        assert rx.collect(lambda m: m)
        me = rx.addrs[0]
        assert out.stats()["local"] == (me[0], me[1])
        state = {**GOLDEN["state_example"], "session": sid, "seq": 1}
        rx.reply(state, me)
        assert wait_until(lambda: out.stats()["robot"] is not None, 2)
        robot = out.stats()["robot"]
        assert (robot["applied_s"], robot["phase"], robot["armed"], robot["dry_run"],
                robot["side"]) == (0.5, "idle", True, True, "right")
        rx.sock.sendto(b"{not json", me)
        rx.reply({**state, "session": "someoneelse1", "seq": 5, "phase": "error"}, me)
        foreign.reply({**state, "seq": 6, "phase": "error"}, me)     # not the target address
        rx.reply({**state, "seq": 0, "phase": "error"}, me)          # older than the shown one
        rx.reply({**state, "seq": 2, "phase": "holding", "applied_s": None, "new_key": 1}, me)
        assert wait_until(lambda: out.stats()["back_channel"].get("received") == 2, 2)
        assert wait_until(lambda: sum(out.stats()["back_channel"].values()) == 6, 2)
        st = out.stats()
        assert st["robot"]["phase"] == "holding" and st["robot"]["applied_s"] is None
        assert "new_key" not in st["robot"]
        assert st["back_channel"] == {"received": 2, "malformed": 1, "other_session": 1,
                                      "foreign": 1, "out_of_order": 1}
        assert st["errors"] == 0
        rig.clock.t += robot_out.STATE_TIMEOUT_S + 0.1           # none for > 2 s: unknown
        st = out.stats()
        assert st["robot"] is None and st["robot_age_s"] > robot_out.STATE_TIMEOUT_S
    finally:
        foreign.close()
        assert out.close(1.0)
        rig.eng.shutdown()


def test_thread_is_named_outside_the_engine_and_close_is_bounded(rx):
    rig = accepted_low()
    out = sender(rig, rx, rate_hz=5)                          # long period: close must wake it
    try:
        out.arm()
        assert rx.collect(lambda m: m)
        assert [t.name for t in out.threads] == [THREAD_NAME] and len(sender_threads()) == 1
        assert all(t.daemon for t in sender_threads())
        assert THREAD_NAME not in [t.name for t in rig.eng.threads]
        t0 = time.perf_counter()
        assert out.close(1.0)
        assert time.perf_counter() - t0 < 0.5 and not sender_threads()
        assert rx.collect(lambda m: m[-1]["kind"] == STOP)
        assert rx.msgs[-1]["reason"] == "shutdown"
        with pytest.raises(RuntimeError):
            out.arm()
        assert out.close(0.1) and not out.armed               # idempotent
    finally:
        out.close(1.0)
        rig.eng.shutdown()


def test_configure_only_while_disarmed(rx):
    rig = accepted_low()
    out = sender(rig, rx)
    try:
        with pytest.raises(ValueError):
            RobotOutput(rig.eng, RobotOutputSettings("localhost"), simulated=False)
        with pytest.raises(ValueError):
            out.configure(RobotOutputSettings("224.0.0.1"))
        out.arm()
        with pytest.raises(RuntimeError):
            out.configure(RobotOutputSettings(rx.host, rx.port, rate_hz=10))
        out.disarm()
        out.configure(RobotOutputSettings(rx.host, rx.port, rate_hz=10))
        assert out.settings.rate_hz == 10
    finally:
        assert out.close(1.0)
        rig.eng.shutdown()


def test_simulated_source_and_off_mode_are_flagged(rx):
    rig = Rig(warm="low")                                     # stabilization OFF
    rig.push("low", 1.0)
    out = sender(rig, rx, simulated=True)
    try:
        out.arm()
        assert rx.collect(lambda m: len(m) >= 3)
        assert all(x["simulated"] and not x["stabilized"] and "stab_ms" not in x
                   and x["s"] == 0.0 and x["status"] == "OFF" for x in rx.msgs)
    finally:
        assert out.close(1.0)
        rig.eng.shutdown()
    assert rx.collect(lambda m: m[-1]["kind"] == STOP) and rx.msgs[-1]["simulated"]


def test_sender_runs_concurrently_with_the_engine(rx):
    """The sender reads snapshot() on its own thread while the worker publishes."""
    rig = accepted_low(StabilizerSettings(enabled=True, up_ms=0, down_ms=0, startup_ms=0))
    out = sender(rig, rx)
    try:
        out.arm()
        t = 1.1
        for k in range(60):
            t = round(t + 0.05, 3)
            rig.push(("low", "medium", "high")[(k // 4) % 3], t)
        assert rx.collect(lambda m: len(m) >= 20)
        out.disarm("done")
        assert rx.collect(lambda m: m[-1]["kind"] == STOP)
        assert [x["seq"] for x in rx.msgs] == list(range(len(rx.msgs)))
        assert {x["s"] for x in rx.msgs} >= {0.0, 0.5, 1.0}
        changes = [x for x in rx.msgs if x["kind"] == CHANGE]
        assert len(changes) >= 3 and out.stats()["errors"] == 0
        for prev, x in zip(rx.msgs, rx.msgs[1:]):
            if x["kind"] == HEARTBEAT:
                assert (x["s"], x["stabilized"]) == (prev["s"], prev["stabilized"])
            elif x["kind"] == CHANGE:
                assert (x["s"], x["stabilized"]) != (prev["s"], prev["stabilized"])
    finally:
        assert out.close(1.0)
        rig.eng.shutdown()


def _ipv6_loopback():
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as s:
            s.bind(("::1", 0))
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _ipv6_loopback(), reason="no IPv6 loopback on this machine")
def test_ipv6_loopback_target():
    rx6 = Receiver(socket.AF_INET6, "::1")
    rig = accepted_low()
    out = sender(rig, rx6)
    try:
        out.arm()
        assert rx6.collect(lambda m: len(m) >= 2)
        assert out.stats()["target"] == f"[::1]:{rx6.port}" and out.stats()["local"][0] == "::1"
        rx6.reply({**GOLDEN["state_example"], "session": out.session}, rx6.addrs[0])
        assert wait_until(lambda: out.stats()["robot"] is not None, 2)
    finally:
        assert out.close(1.0)
        rig.eng.shutdown()
        rx6.close()
