"""Reliability / performance regression tests, including fault injection."""

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import make_emg, wait_until
from gforce_emg_ui.hub import DataHub
from gforce_emg_ui.recorder import SAVED, Recorder


# -- subscriber isolation / async consumers ------------------------------------

def test_subscriber_exception_is_isolated(tmp_path):
    hub = DataHub()

    def bad(block):
        raise RuntimeError("decoder bug (injected)")
    hub.subscribe_emg(bad, name="bad-decoder")
    rec = Recorder(hub, tmp_path)
    s = rec.start("iso")
    for i in range(5):
        hub.push_emg(make_emg(i * 16))
    s.request_stop()
    assert s.wait(5) and s.state == SAVED
    assert s.counts["emg"]["written"] == 80
    assert hub.emg.total == 80
    assert hub.subscriber_errors["bad-decoder"] == 5


def test_async_subscriber_keeps_work_off_the_callback_thread():
    hub = DataHub()
    threads, seen = set(), []

    def slow_inference(block):
        threads.add(threading.get_ident())
        time.sleep(0.02)
        seen.append(int(block.sample_index[0]))
        if len(seen) == 2:
            raise ValueError("model error (injected)")
    sub = hub.subscribe_emg_async(slow_inference, maxsize=4, name="infer")
    t0 = time.perf_counter()
    for i in range(50):
        hub.push_emg(make_emg(i * 16))
    push_time = time.perf_counter() - t0
    assert push_time < 0.2                        # 50 x 20 ms of work did not block pushes
    assert wait_until(lambda: sub.pending == 0, 5)
    sub.stop()
    assert threading.get_ident() not in threads
    assert sub.dropped > 0 and sub.processed + sub.dropped == 50
    assert sub.errors == 1
    assert seen == sorted(seen)                   # order preserved


# -- reproducible fault injection ----------------------------------------------

def test_simulator_faults_are_reproducible():
    from gforce_emg_ui.sources import SimFaults, SimGenerator
    faults = SimFaults(lost_prob=0.05, index_jump_every=5, index_jump=7)

    def run(seed):
        g = SimGenerator(seed=seed, rate=1000, package=16, faults=faults)
        hub = DataHub()
        blocks = [g.next_emg() for _ in range(40)]
        for b in blocks:
            hub.push_emg(b)
        return blocks, hub
    a, ha = run(11)
    b, hb = run(11)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.sample_index, y.sample_index)
        np.testing.assert_array_equal(x.lost, y.lost)
        np.testing.assert_array_equal(np.isnan(x.data), np.isnan(y.data))
        np.testing.assert_array_equal(np.nan_to_num(x.data), np.nan_to_num(y.data))
    assert ha.lost_samples == hb.lost_samples > 0
    assert ha.missing_samples == hb.missing_samples == 7 * 7   # jumps after blocks 5,10,..,35


# -- connection states (simulator) ---------------------------------------------

def test_sim_connection_states_failed_connect_and_reconnect(qapp):
    from gforce_emg_ui.sources import (CONNECTING, FAILED, IDLE, RECONNECTING, STREAMING,
                                       SimFaults, SimSource)
    hub = DataHub()
    src = SimSource(hub, seed=2, faults=SimFaults(fail_connects=1), reconnect_delay=0.2)
    states = []
    src.state_changed.connect(states.append)
    assert src.state == IDLE
    src.connect_device(SimSource.ADDRESS)
    assert wait_until(lambda: src.state == FAILED, 5, qapp)
    assert not src.is_connected and hub.emg.total == 0      # cleaned up
    src.connect_device(SimSource.ADDRESS)
    assert wait_until(lambda: src.state == STREAMING and hub.emg.total > 0, 5, qapp)
    seg = hub.emg_segment
    src.inject_disconnect()
    assert wait_until(lambda: src.state == RECONNECTING, 2, qapp)
    assert wait_until(lambda: src.state == STREAMING and hub.emg_segment == seg + 1, 5, qapp)
    src.disconnect_device()
    assert wait_until(lambda: src.state == IDLE, 5, qapp)
    assert states[:3] == [CONNECTING, FAILED, CONNECTING]
    assert RECONNECTING in states


# -- connection states (real SdkSource against a fake SDK) ---------------------

class _FakeSensor:
    def __init__(self, sdk, fail_at=None):
        self.sdk = sdk
        self.fail_at = set(fail_at or ())
        self.calls = []
        self.isReady = False
        self.hasInited = False
        self.isDataTransfering = False
        self.BLEDevice = SimpleNamespace(Name="gForceUltra", Address="AA:BB", RSSI=-50)
        self.on_state_change = None

    def _state(self, st):
        if self.on_state_change:
            self.on_state_change(self, st)

    def connect(self):
        self.calls.append("connect")
        if "connect" in self.fail_at:
            return False
        self.isReady = True
        return True

    def init(self, *a):
        self.calls.append("init")
        if "init" in self.fail_at:
            return False
        self.hasInited = True
        return True

    def startDataNotification(self):
        self.calls.append("start")
        if "start" in self.fail_at:
            return False
        self.isDataTransfering = True
        return True

    def stopDataNotification(self):
        self.calls.append("stop")
        self.isDataTransfering = False
        return True

    def disconnect(self):
        self.calls.append("disconnect")
        was = self.isReady
        self.isReady = self.hasInited = self.isDataTransfering = False
        if was:
            self._state(self.sdk.DeviceStateEx.Disconnected)
        return True

    def drop_link(self):
        self.isReady = self.hasInited = self.isDataTransfering = False
        self._state(self.sdk.DeviceStateEx.Disconnected)

    def getParam(self, key):
        return {"EMG_SAMPLE_RATE": "1000", "NTF": "NTF_EMG|ON|NTF_IMU|ON",
                "FILTER": "FILTER_50HZ|ON|FILTER_60HZ|ON|FILTER_HPF|ON|FILTER_LPF|ON"}.get(key, "Error: x")

    def setParam(self, key, value):
        self.calls.append(f"set {key}={value}")
        return "OK"

    def getDeviceInfo(self):
        return SimpleNamespace(ModelName="OB6000", FirmwareVersion="V1.0.4", HardwareVersion="1.40",
                               EmgChannelCount=8, ImuChannelCount=6, MTUSize=247)

    def getBatteryLevel(self):
        return 90

    def setAutoReconnect(self, on):
        pass


def _fake_sdk(sensors):
    import sensor as real

    def submit(fn, *a, **k):
        f = Future()
        f.set_result(fn(*a, **k))
        return f

    class Controller:
        isEnable = True
        isScanning = False

        def __init__(self):
            self.on_sensor_scan_result = None

        def requireSensor(self, device):
            return sensors.pop(0)

        def startScan(self, p):
            pass

        def stopScan(self):
            pass

        def getBLEBackendName(self):
            return "fake"

        def terminate(self):
            pass

    sdk = SimpleNamespace(SensorController=Controller, submit=submit,
                          DataType=real.DataType, DeviceStateEx=real.DeviceStateEx)
    return sdk


def _sdk_source(sensors):
    from gforce_emg_ui.sources import SdkSource
    sdk = _fake_sdk(sensors)
    for s in sensors:
        s.sdk = sdk
    src = SdkSource(DataHub(), sdk=sdk, reconnect_delays=(0.0, 0.0))
    src._on_scan([SimpleNamespace(Name="gForceUltra", Address="AA:BB", RSSI=-50)])
    states = []
    src.state_changed.connect(states.append)
    return src, states


@pytest.mark.parametrize("fail_at,expected_cleanup", [
    ("connect", []),
    ("init", ["disconnect"]),
    ("start", ["disconnect"]),
])
def test_sdk_failed_connect_cleans_up(qapp, fail_at, expected_cleanup):
    from gforce_emg_ui.sources import CONNECTING, FAILED, RECONNECTING
    sensor = _FakeSensor(None, fail_at=[fail_at])
    src, states = _sdk_source([sensor])
    src.connect_device("AA:BB")
    assert src.state == FAILED
    assert src._sensor is None and not src.is_connected
    assert not sensor.isReady and not sensor.isDataTransfering
    i = sensor.calls.index({"connect": "connect", "init": "init", "start": "start"}[fail_at])
    assert [c for c in sensor.calls[i + 1:] if not c.startswith("set ")] == expected_cleanup
    assert states == [CONNECTING, FAILED]
    assert RECONNECTING not in states            # our own cleanup is not a link loss


def test_sdk_unexpected_disconnect_reconnects(qapp):
    from gforce_emg_ui.sources import CONNECTING, IDLE, RECONNECTING, STREAMING
    first, second = _FakeSensor(None), _FakeSensor(None)
    src, states = _sdk_source([first, second])
    src.connect_device("AA:BB")
    assert src.state == STREAMING
    sid = src._stream_id
    first.drop_link()
    assert src.state == STREAMING and src._sensor is second
    assert src._stream_id == sid + 1
    assert states == [CONNECTING, STREAMING, RECONNECTING, STREAMING]
    src.disconnect_device()
    assert src.state == IDLE
    assert states[-2:] == ["disconnecting", IDLE]


def test_sdk_reconnect_gives_up_after_retries(qapp):
    from gforce_emg_ui.sources import FAILED, RECONNECTING
    first = _FakeSensor(None)
    retries = [_FakeSensor(None, fail_at=["connect"]) for _ in range(2)]
    src, states = _sdk_source([first] + retries)
    src.connect_device("AA:BB")
    first.drop_link()
    assert src.state == FAILED and src._sensor is None
    assert states.count(RECONNECTING) >= 1


# -- redraw avoidance ------------------------------------------------------------

def test_dashboards_skip_redraw_without_new_data_or_settings(qapp):
    from gforce_emg_ui.ui.emg_detail import DetailDashboard
    from gforce_emg_ui.ui.emg_equal import EqualScaleDashboard
    from gforce_emg_ui.ui.imu_view import ImuDashboard
    from conftest import make_imu
    hub = DataHub()
    hub.push_emg(make_emg(0, n=500))
    hub.push_imu(make_imu(0, n=20))
    for dash in (EqualScaleDashboard(hub), DetailDashboard(hub), ImuDashboard(hub)):
        assert dash.refresh() is True
        assert dash.refresh() is False, type(dash).__name__
        assert dash.refresh() is False
    eq = EqualScaleDashboard(hub)
    eq.refresh()
    eq.window_combo.setCurrentIndex(0)
    assert eq.refresh() is True
    assert eq.refresh() is False
    hub.push_emg(make_emg(500))
    assert eq.refresh() is True


# -- benchmark command -----------------------------------------------------------

def test_benchmark_command_runs(tmp_path):
    out = tmp_path / "bench.json"
    r = subprocess.run([sys.executable, "-m", "gforce_emg_ui.bench", "--quick", "--json", str(out)],
                       capture_output=True, text=True, timeout=120,
                       env={**__import__("os").environ, "QT_QPA_PLATFORM": "offscreen"})
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(out.read_text())
    for key in ("hub_push_us", "recorder", "ring_latest_us", "dashboard_refresh_ms", "simulated_stream"):
        assert key in res
    assert res["recorder"]["reconciled"] is True
    assert "not hardware" in r.stdout.lower()


def test_stale_queued_state_signal_does_not_override_current_state(qapp, tmp_path):
    """State signals emitted from SDK threads are queued and can arrive late
    (seen on hardware: 'idle' from a finished disconnect delivered after the
    next 'connecting'). The UI must act on the source's current state."""
    from gforce_emg_ui.sources import IDLE, SimSource
    from gforce_emg_ui.ui.main_window import MainWindow
    hub = DataHub()
    src = SimSource(hub, seed=1)
    win = MainWindow(hub, src, tmp_path)
    try:
        src.connect_device(SimSource.ADDRESS)
        assert wait_until(lambda: hub.emg.total > 0, 5, qapp)
        win._toggle_recording()
        src.state_changed.emit(IDLE)                 # stale signal
        qapp.processEvents()
        assert win.recorder.active                   # recording not stopped
        assert "Streaming" in win.state_label.text()
        assert win.btn_disconnect.isEnabled()
    finally:
        if win.recorder.active:
            win._toggle_recording()
        win.recorder.session.wait(5)
        src.disconnect_device()
        win.close()
