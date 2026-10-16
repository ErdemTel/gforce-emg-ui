"""Data sources feeding the DataHub: the real gForceUltra band (via sensor-sdk)
and a simulator with the same interface for development without hardware.

All Qt signals may be emitted from worker threads; Qt queues them to the UI thread.
"""

import threading
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PyQt5 import QtCore

from .config import (DEFAULT_EMG_RATE, EMG_CHANNELS, FILTERS, IMU_CHANNELS,
                     PACKAGE_SAMPLE_COUNT, POWER_REFRESH_MS, SCAN_PERIOD_MS)
from .hub import DataHub, EmgBlock, ImuBlock


# Connection states (Source.state, emitted via state_changed)
IDLE = "idle"
SCANNING = "scanning"
CONNECTING = "connecting"
STREAMING = "streaming"
RECONNECTING = "reconnecting"     # link lost unexpectedly; retrying automatically
DISCONNECTING = "disconnecting"
FAILED = "failed"                 # connect or all reconnect attempts failed; cleaned up


class Source(QtCore.QObject):
    state_changed = QtCore.pyqtSignal(str)
    devices_found = QtCore.pyqtSignal(list)    # [(name, address, rssi), ...]
    status = QtCore.pyqtSignal(str)
    connected = QtCore.pyqtSignal(dict)        # device summary
    disconnected = QtCore.pyqtSignal()
    battery = QtCore.pyqtSignal(int)           # percent
    params = QtCore.pyqtSignal(dict)           # {"emg_rate": int, "filters": {key: bool}}
    error = QtCore.pyqtSignal(str)

    is_simulation = False             # True for synthetic data (never real recordings)

    def __init__(self, hub: DataHub, emg_rate: int = DEFAULT_EMG_RATE):
        super().__init__()
        self.hub = hub
        self.target_emg_rate = emg_rate
        self.settings_locked = False
        self.state = IDLE
        self._state_lock = threading.Lock()

    def _set_state(self, state: str):
        with self._state_lock:
            if state == self.state:
                return
            self.state = state
        self.state_changed.emit(state)

    def lock_settings(self, locked: bool):
        """While locked (recording), firmware filters and sample rate cannot change."""
        self.settings_locked = bool(locked)

    def _check_unlocked(self, what: str) -> bool:
        if self.settings_locked:
            self.error.emit(f"{what} is locked while recording")
            return False
        return True

    @property
    def is_connected(self) -> bool:
        return self.state == STREAMING

    def backend_name(self) -> str:
        return ""

    def start_scan(self): ...
    def stop_scan(self): ...
    def connect_device(self, address: str): ...
    def disconnect_device(self): ...
    def set_emg_rate(self, rate: int) -> bool: ...
    def set_filter(self, key: str, on: bool) -> bool: ...
    def shutdown(self): ...


# ---------------------------------------------------------------------------
# Real band
# ---------------------------------------------------------------------------

class SdkSource(Source):
    """The real band. Blocking SDK calls run on the SDK runner thread (sdk.submit).

    Connection lifecycle: IDLE/FAILED -> CONNECTING -> STREAMING. A failed connect
    stops/disconnects whatever was opened and ends in FAILED. An unexpected link
    loss while STREAMING goes to RECONNECTING and retries with `reconnect_delays`
    (each success starts a new stream_id, i.e. a new data segment); if all
    attempts fail the state is FAILED. A user disconnect ends in IDLE.
    """

    def __init__(self, hub: DataHub, emg_rate: int = DEFAULT_EMG_RATE,
                 package_samples: int = PACKAGE_SAMPLE_COUNT, sdk=None,
                 reconnect_delays=(1.0, 2.0, 4.0, 8.0)):
        super().__init__(hub, emg_rate)
        if sdk is None:
            import sensor as sdk
        self._sdk = sdk
        self._DataType = sdk.DataType
        self._package_samples = package_samples
        self.reconnect_delays = tuple(reconnect_delays)
        self._ctrl = sdk.SensorController()
        self._ctrl.on_sensor_scan_result = self._on_scan
        self._devices = {}
        self._sensor = None
        self._address = None
        self._user_disconnect = False
        self._unhandled_types = set()
        self._stream_id = 0

    def backend_name(self) -> str:
        try:
            return self._ctrl.getBLEBackendName()
        except Exception:
            return ""

    # -- scanning -------------------------------------------------------------
    def start_scan(self):
        if not self._ctrl.isEnable:
            self.error.emit("Bluetooth is disabled (no adapter or dongle usable)")
            return
        if self.state not in (IDLE, FAILED, SCANNING):
            return
        if not self._ctrl.isScanning:
            self._ctrl.startScan(SCAN_PERIOD_MS)
        self._set_state(SCANNING)
        self.status.emit("Scanning...")

    def stop_scan(self):
        if self._ctrl.isScanning:
            self._ctrl.stopScan()
        if self.state == SCANNING:
            self._set_state(IDLE)

    def _on_scan(self, device_list):
        for d in device_list:
            self._devices[d.Address] = d
        self.devices_found.emit([(d.Name, d.Address, d.RSSI) for d in device_list])

    # -- connection -----------------------------------------------------------
    def connect_device(self, address: str):
        if self.state in (CONNECTING, STREAMING, RECONNECTING, DISCONNECTING):
            return
        self._address = address
        self._user_disconnect = False
        self._set_state(CONNECTING)
        self._sdk.submit(self._connect_worker, address)

    def _connect_worker(self, address: str, reconnect: bool = False) -> bool:
        """Runs on the SDK runner thread. Returns True when streaming."""
        sensor = None
        try:
            device = self._devices.get(address)
            if device is None:
                raise RuntimeError(f"unknown device {address}; scan first")
            if self._ctrl.isScanning:
                self._ctrl.stopScan()
            sensor = self._ctrl.requireSensor(device)
            if sensor is None:
                raise RuntimeError("failed to create sensor profile")
            sensor.on_sensor_notify_data = self._on_data
            sensor.on_state_change = self._on_state
            sensor.on_power_changed = lambda s, p: p >= 0 and self.battery.emit(int(p))
            sensor.on_error_callback = lambda s, msg: self.error.emit(f"SDK: {msg}")
            sensor.on_device_info_update = lambda s, info: self._emit_params(s)

            self.status.emit(f"{'Reconnecting' if reconnect else 'Connecting'} to {device.Name}...")
            if not sensor.isReady and not sensor.connect():
                raise RuntimeError(f"failed to connect to {device.Name}")
            if not sensor.hasInited and not sensor.init(self._package_samples, POWER_REFRESH_MS):
                raise RuntimeError(f"failed to initialize {device.Name}")

            rate = sensor.getParam("EMG_SAMPLE_RATE")
            if str(rate) != str(self.target_emg_rate):
                res = sensor.setParam("EMG_SAMPLE_RATE", str(self.target_emg_rate))
                if str(res).startswith("Error"):
                    self.error.emit(f"EMG_SAMPLE_RATE={self.target_emg_rate}: {res}")
            self._ensure_notifications(sensor, ("NTF_EMG", "NTF_IMU"))

            if not reconnect:
                self.hub.reset()
            self._stream_id += 1          # new stream -> new data segment
            if not sensor.isDataTransfering and not sensor.startDataNotification():
                raise RuntimeError("failed to start data stream")
            if self._user_disconnect:     # user gave up while we were connecting
                raise RuntimeError("cancelled")

            self._sensor = sensor
            self._set_state(STREAMING)
            info = sensor.getDeviceInfo()
            self.connected.emit({
                "name": device.Name,
                "address": device.Address,
                "model": getattr(info, "ModelName", ""),
                "firmware": getattr(info, "FirmwareVersion", ""),
                "hardware": getattr(info, "HardwareVersion", ""),
                "emg_channels": getattr(info, "EmgChannelCount", 0),
                "imu_channels": getattr(info, "ImuChannelCount", 0),
                "mtu": getattr(info, "MTUSize", 0),
                "backend": self.backend_name(),
                "package_samples": self._package_samples,
            })
            self._emit_params(sensor)
            power = sensor.getBatteryLevel()
            if power >= 0:
                self.battery.emit(int(power))
            return True
        except Exception as e:
            self.error.emit(f"{'Reconnect' if reconnect else 'Connect'} failed: {e}")
        self._cleanup(sensor)
        if not reconnect:
            self._set_state(IDLE if self._user_disconnect else FAILED)
        return False

    def _cleanup(self, sensor):
        """Undo a partial connection. The sensor is not self._sensor, so the
        Disconnected callback this triggers is not treated as a link loss."""
        if sensor is None:
            return
        try:
            if sensor.isDataTransfering:
                sensor.stopDataNotification()
        except Exception as e:
            self.error.emit(f"Cleanup: stop stream failed: {e}")
        try:
            if sensor.isReady:
                sensor.disconnect()
        except Exception as e:
            self.error.emit(f"Cleanup: disconnect failed: {e}")
        if sensor is self._sensor:
            self._sensor = None

    def _reconnect_loop(self, address: str):
        for delay in self.reconnect_delays:
            if self._user_disconnect:
                return
            time.sleep(delay)
            if self._user_disconnect:
                return
            if self._connect_worker(address, reconnect=True):
                return
        self._set_state(FAILED)
        self.disconnected.emit()

    def _ensure_notifications(self, sensor, keys):
        state = str(sensor.getParam("NTF"))
        if state.startswith("Error"):
            return
        items = state.split("|")
        current = {items[i]: items[i + 1] for i in range(0, len(items) - 1, 2)}
        for key in keys:
            if current.get(key) == "OFF":
                res = sensor.setParam(key, "ON")
                if str(res).startswith("Error"):
                    self.error.emit(f"{key}=ON: {res}")

    def _emit_params(self, sensor):
        # May run on the SDK callback thread, where blocking getParam is not
        # allowed; always hop to the SDK runner thread.
        self._sdk.submit(self._read_params, sensor)

    def _read_params(self, sensor):
        try:
            out = {"filters": {}}
            rate = str(sensor.getParam("EMG_SAMPLE_RATE"))
            if not rate.startswith("Error"):
                out["emg_rate"] = int(rate)
            filt = str(sensor.getParam("FILTER"))
            if filt and not filt.startswith("Error"):
                items = filt.split("|")
                parsed = {items[i]: items[i + 1] for i in range(0, len(items) - 1, 2)}
                out["filters"] = {k: parsed.get(k) == "ON" for k in FILTERS if k in parsed}
            self.params.emit(out)
        except Exception as e:
            self.error.emit(f"Reading parameters failed: {e}")

    def disconnect_device(self):
        if self.state in (IDLE, FAILED, DISCONNECTING):
            return
        self._user_disconnect = True
        sensor = self._sensor
        if sensor is None:              # connecting / waiting to reconnect: just stop
            self._set_state(IDLE)
            self.disconnected.emit()
            return
        self._set_state(DISCONNECTING)

        def work():
            try:
                if sensor.isDataTransfering:
                    sensor.stopDataNotification()
                sensor.disconnect()
            except Exception as e:
                self.error.emit(f"Disconnect failed: {e}")
            if self.state == DISCONNECTING:    # no Disconnected callback arrived
                self._sensor = None
                self._set_state(IDLE)
                self.disconnected.emit()
        self._sdk.submit(work)

    def _on_state(self, sensor, state):
        # SDK callback thread: no blocking SDK calls here
        if state != self._sdk.DeviceStateEx.Disconnected or sensor is not self._sensor:
            return
        self._sensor = None
        if self._user_disconnect:
            self._set_state(IDLE)
            self.disconnected.emit()
        elif self.state == STREAMING:
            self._set_state(RECONNECTING)
            self.status.emit("Link lost, reconnecting...")
            self._sdk.submit(self._reconnect_loop, self._address)

    def set_emg_rate(self, rate: int) -> bool:
        if not self._check_unlocked("EMG sample rate"):
            return False
        self.target_emg_rate = rate
        sensor = self._sensor
        if sensor is None:
            return True

        def work():
            res = sensor.setParam("EMG_SAMPLE_RATE", str(rate))
            if str(res).startswith("Error"):
                self.error.emit(f"EMG_SAMPLE_RATE={rate}: {res}")
            self._read_params(sensor)
        self._sdk.submit(work)
        return True

    def set_filter(self, key: str, on: bool) -> bool:
        if not self._check_unlocked("Firmware filter"):
            return False
        sensor = self._sensor
        if sensor is None:
            return False

        def work():
            res = sensor.setParam(key, "ON" if on else "OFF")
            if str(res).startswith("Error"):
                self.error.emit(f"{key}: {res}")
            self._read_params(sensor)
        self._sdk.submit(work)
        return True

    # -- data (SDK callback thread) -------------------------------------------
    def _on_data(self, sensor, data_list):
        # Sample buffers are borrowed from native memory that the SDK reuses:
        # copy everything we need before returning.
        host_time, host_mono = time.time(), time.monotonic()
        DT = self._DataType
        for data in data_list:
            try:
                if data is None or data.channelCount == 0 or data.sampleCount == 0:
                    continue
                dt = data.dataType
                if dt == DT.NTF_EMG:
                    self.hub.push_emg(self._emg_block(data, host_time, host_mono, self._stream_id))
                elif dt == DT.NTF_IMU:
                    self.hub.push_imu(self._imu_block(data, host_time, host_mono, self._stream_id))
                elif dt not in self._unhandled_types:
                    self._unhandled_types.add(dt)
                    print(f"[SdkSource] ignoring data type {dt!r}")
            except Exception:
                traceback.print_exc()

    @staticmethod
    def _convert(arr, channels):
        """(values, valid, lost) with missing channels and lost samples as NaN / invalid."""
        ch, n = arr.shape
        k = min(ch, channels)
        lost = arr["isLost"][0] != 0
        values = np.full((channels, n), np.nan, dtype=np.float32)
        valid = np.zeros((channels, n), dtype=bool)
        valid[:k] = arr["isLost"][:k] == 0
        values[:k] = np.where(valid[:k], arr["data"][:k], np.nan)
        return values, valid, lost

    @classmethod
    def _emg_block(cls, data, host_time, host_mono, stream_id) -> EmgBlock:
        arr = data.as_numpy()
        values, valid, lost = cls._convert(arr, EMG_CHANNELS)
        k = min(arr.shape[0], EMG_CHANNELS)
        imp = np.full(EMG_CHANNELS, np.nan)
        imp[:k] = arr["impedance"][:k, -1]
        return EmgBlock(
            data=values,
            valid=valid,
            sample_index=arr["sampleIndex"][0].astype(np.int64),
            device_time=arr["absTimeStampInSec"][0].copy(),
            lost=lost,
            host_time=host_time,
            host_mono=host_mono,
            sample_rate=float(data.sampleRate),
            impedance=imp,
            lost_packages=int(data.lostPackageCount),
            stream_id=stream_id,
        )

    @classmethod
    def _imu_block(cls, data, host_time, host_mono, stream_id) -> ImuBlock:
        arr = data.as_numpy()
        values, valid, _ = cls._convert(arr, IMU_CHANNELS)
        return ImuBlock(
            data=values,
            valid=valid,
            sample_index=arr["sampleIndex"][0].astype(np.int64),
            device_time=arr["absTimeStampInSec"][0].copy(),
            host_time=host_time,
            host_mono=host_mono,
            sample_rate=float(data.sampleRate),
            stream_id=stream_id,
        )

    def shutdown(self, timeout: float = 30.0):
        """Disconnect and let queued SDK work finish before terminate(): terminating
        during an in-flight disconnect crashed in native code (segfault on exit)."""
        self._user_disconnect = True          # also cancels a connect / reconnect loop
        try:
            self.stop_scan()
        except Exception as e:
            print(f"[SdkSource] stop scan error: {e}")

        def work():
            sensor = self._sensor             # read when it runs (after queued work)
            if sensor is None:
                return
            try:
                if sensor.isDataTransfering:
                    sensor.stopDataNotification()
                if sensor.isReady:
                    sensor.disconnect()
            except Exception as e:
                print(f"[SdkSource] disconnect on shutdown failed: {e}")
        try:
            # one FIFO runner thread: this also waits for SDK work already queued
            self._sdk.submit(work).result(timeout)
        except Exception as e:
            print(f"[SdkSource] shutdown wait: {e}")
        try:
            self._ctrl.terminate()
        except Exception as e:
            print(f"[SdkSource] terminate error: {e}")


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

@dataclass
class SimFaults:
    """Reproducible fault injection for the simulator (same seed -> same faults)."""
    lost_prob: float = 0.0              # per-sample probability of an SDK-style lost sample
    missing_channels: tuple = ()        # EMG channels never delivered
    index_jump_every: int = 0           # every N EMG blocks, skip `index_jump` sample indices
    index_jump: int = 0
    fail_connects: int = 0              # number of connect attempts that fail


class SimGenerator:
    """Deterministic synthetic 8-ch EMG + 6-axis IMU + impedance, block by block.

    Cycles through 'gestures' that each activate a different channel pattern.
    Contact scenarios: "default" (CH6 marginal, CH8 loose, as before) or "good"
    (all 8 channels <= 500 kOhm; simulation only, for exercising workflows that
    require good contact).
    """

    IMU_RATE = 50
    CONTACT_SCENARIOS = {
        "default": (180, 220, 260, 300, 350, 700, 240, 1400),
        "good": (180, 220, 260, 300, 350, 280, 240, 320),
    }

    def __init__(self, seed=None, rate: int = DEFAULT_EMG_RATE,
                 package: int = PACKAGE_SAMPLE_COUNT, faults: Optional[SimFaults] = None,
                 filters: Optional[dict] = None, contact: str = "default"):
        self.rng = np.random.default_rng(seed)
        self.rate = rate
        self.package = package
        self.faults = faults or SimFaults()
        self.filters = filters if filters is not None else {k: False for k in FILTERS}
        fixed = np.random.default_rng(7)
        self._gestures = fixed.uniform(0.0, 1.0, (6, EMG_CHANNELS)) ** 3
        self._offsets = fixed.uniform(-150, 150, EMG_CHANNELS)
        self._base_imp = np.array(self.CONTACT_SCENARIOS[contact], float) * 1000
        self.stream_id = 1
        self.restart()

    def restart(self, stream_id: Optional[int] = None):
        """Start a new stream: like the band after a reconnect, indices restart at 0."""
        if stream_id is not None:
            self.stream_id = stream_id
        self.emg_idx = 0
        self.imu_idx = 0
        self._imu_carry = 0.0
        self._blocks = 0
        self._imp = self._base_imp.copy()
        self._pending_imu: Optional[ImuBlock] = None

    def next_emg(self, host_time: Optional[float] = None,
                 host_mono: Optional[float] = None) -> EmgBlock:
        host_time = time.time() if host_time is None else host_time
        host_mono = time.monotonic() if host_mono is None else host_mono
        rate, n, f = self.rate, self.package, self.faults
        idx = np.arange(self.emg_idx, self.emg_idx + n, dtype=np.int64)
        ts = idx / rate
        phase = ts % 4.0                                   # 2.5 s contraction, 1.5 s rest
        g = int(ts[0] // 4.0) % len(self._gestures)
        act = np.clip(np.sin(np.pi * np.clip(phase, 0, 2.5) / 2.5), 0, None) * (phase < 2.5)
        envelope = 8.0 + 220.0 * self._gestures[g][:, None] * act[None, :]
        emg = self.rng.standard_normal((EMG_CHANNELS, n)) * envelope
        if not self.filters.get("FILTER_HPF"):
            emg += self._offsets[:, None]
        if not self.filters.get("FILTER_50HZ"):
            emg += 15.0 * np.sin(2 * np.pi * 50.0 * ts)[None, :]
        if self.emg_idx % rate < n:                       # impedance updates ~1 Hz
            self._imp = self._base_imp * self.rng.uniform(0.95, 1.05, EMG_CHANNELS)
        lost = self.rng.random(n) < f.lost_prob
        valid = np.ones((EMG_CHANNELS, n), bool)
        valid[:, lost] = False
        imp = self._imp.copy()
        for c in f.missing_channels:
            valid[c] = False
            imp[c] = np.nan
        block = EmgBlock(
            data=np.where(valid, emg, np.nan).astype(np.float32),
            valid=valid, sample_index=idx, device_time=ts, lost=lost,
            host_time=host_time, host_mono=host_mono, sample_rate=float(rate),
            impedance=imp, lost_packages=int(lost.any()), stream_id=self.stream_id)
        self.emg_idx += n
        self._blocks += 1
        if f.index_jump_every and self._blocks % f.index_jump_every == 0:
            self.emg_idx += f.index_jump
        self._make_imu(n, host_time, host_mono)
        return block

    def _make_imu(self, n_emg, host_time, host_mono):
        self._imu_carry += n_emg * self.IMU_RATE / self.rate
        m = int(self._imu_carry)
        if not m:
            self._pending_imu = None
            return
        self._imu_carry -= m
        ti = (self.imu_idx + np.arange(m)) / self.IMU_RATE
        acc = np.vstack([0.3 * np.sin(0.5 * ti), 0.3 * np.cos(0.5 * ti),
                         np.full(m, 0.95)]) + self.rng.normal(0, 0.01, (3, m))
        gyro = np.vstack([20 * np.cos(0.5 * ti), -20 * np.sin(0.5 * ti),
                          5 * np.sin(0.2 * ti)]) + self.rng.normal(0, 0.5, (3, m))
        self._pending_imu = ImuBlock(
            data=np.vstack([acc, gyro]).astype(np.float32), valid=np.ones((6, m), bool),
            sample_index=np.arange(self.imu_idx, self.imu_idx + m, dtype=np.int64),
            device_time=ti, host_time=host_time, host_mono=host_mono,
            sample_rate=float(self.IMU_RATE), stream_id=self.stream_id)
        self.imu_idx += m

    def take_imu(self) -> Optional[ImuBlock]:
        b, self._pending_imu = self._pending_imu, None
        return b


class SimSource(Source):
    """Simulator with the Source interface and state machine, paced in real time.

    Fault injection: SimFaults (lost samples, missing channels, index jumps,
    failing connects) and inject_disconnect() for a link loss followed by an
    automatic reconnect (new stream_id, indices restart at 0).
    """

    ADDRESS = "SIM:00:00:00:00:00"
    is_simulation = True

    def __init__(self, hub: DataHub, emg_rate: int = DEFAULT_EMG_RATE,
                 package_samples: int = PACKAGE_SAMPLE_COUNT, seed=None,
                 faults: Optional[SimFaults] = None, reconnect_delay: float = 1.0,
                 contact: str = "default"):
        super().__init__(hub, emg_rate)
        if contact not in SimGenerator.CONTACT_SCENARIOS:
            raise ValueError(f"unknown simulated contact scenario {contact!r}")
        self._package = package_samples
        self._seed = seed
        self.contact = contact
        self.faults = faults or SimFaults()
        self.reconnect_delay = reconnect_delay
        self._thread = None
        self._stop = threading.Event()
        self._drop = threading.Event()
        self._filters = {k: False for k in FILTERS}
        self._gen: Optional[SimGenerator] = None
        self._stream_id = 0

    def backend_name(self) -> str:
        return "simulator"

    def start_scan(self):
        if self.state not in (IDLE, FAILED, SCANNING):
            return
        self._set_state(SCANNING)
        self.status.emit("Scanning (simulator)...")
        QtCore.QTimer.singleShot(300, lambda: self.devices_found.emit(
            [("gForceUltra-SIM", self.ADDRESS, -42)]))

    def stop_scan(self):
        if self.state == SCANNING:
            self._set_state(IDLE)

    def connect_device(self, address: str):
        if self.state in (CONNECTING, STREAMING, RECONNECTING, DISCONNECTING):
            return
        self._set_state(CONNECTING)
        if self.faults.fail_connects > 0:
            self.faults.fail_connects -= 1
            self.error.emit("Connect failed: simulated failure")
            self._set_state(FAILED)          # nothing was started: nothing to clean up
            return
        self.hub.reset()
        self._stream_id += 1
        self._gen = SimGenerator(self._seed, self.target_emg_rate, self._package,
                                 self.faults, self._filters, self.contact)
        self._gen.stream_id = self._stream_id
        self._stop.clear()
        self._drop.clear()
        self._thread = threading.Thread(target=self._run, name="sim-source", daemon=True)
        self._thread.start()
        self._set_state(STREAMING)
        self.connected.emit({"name": "gForceUltra-SIM", "address": self.ADDRESS,
                             "model": "gForceUltra (simulated)" if self.contact == "default"
                             else f"gForceUltra (simulated, {self.contact} contact)",
                             "firmware": "-",
                             "hardware": "-", "emg_channels": EMG_CHANNELS,
                             "imu_channels": IMU_CHANNELS, "mtu": 0,
                             "backend": "simulator", "package_samples": self._package})
        self._emit_params()
        self.battery.emit(87)

    def inject_disconnect(self):
        """Simulate an unexpected link loss (the source reconnects by itself)."""
        self._drop.set()

    def disconnect_device(self):
        if self._thread is None:
            if self.state != IDLE:
                self._set_state(IDLE)
            return
        self._set_state(DISCONNECTING)
        self._stop.set()
        self._thread.join(timeout=2)
        self._thread = None
        self._set_state(IDLE)
        self.disconnected.emit()

    def set_emg_rate(self, rate: int) -> bool:
        if not self._check_unlocked("EMG sample rate"):
            return False
        self.target_emg_rate = rate
        if self._gen is not None:
            self._gen.rate = rate
        self._emit_params()
        return True

    def set_filter(self, key: str, on: bool) -> bool:
        if not self._check_unlocked("Firmware filter"):
            return False
        self._filters[key] = on
        self._emit_params()
        return True

    def _emit_params(self):
        self.params.emit({"emg_rate": self.target_emg_rate, "filters": dict(self._filters)})

    def shutdown(self):
        self._stop.set()

    def _run(self):
        gen = self._gen
        next_t = time.monotonic()
        while not self._stop.is_set():
            if self._drop.is_set():
                self._drop.clear()
                self._set_state(RECONNECTING)
                self.status.emit("Link lost, reconnecting...")
                if self._stop.wait(self.reconnect_delay):
                    return
                self._stream_id += 1
                gen.restart(stream_id=self._stream_id)
                self._set_state(STREAMING)
                next_t = time.monotonic()
            block = gen.next_emg()
            self.hub.push_emg(block)
            imu = gen.take_imu()
            if imu is not None:
                self.hub.push_imu(imu)
            next_t += block.data.shape[1] / block.sample_rate
            delay = next_t - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            elif delay < -1.0:      # fell far behind (e.g. machine suspended)
                next_t = time.monotonic()
