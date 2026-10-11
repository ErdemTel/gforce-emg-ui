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


class Source(QtCore.QObject):
    devices_found = QtCore.pyqtSignal(list)    # [(name, address, rssi), ...]
    status = QtCore.pyqtSignal(str)
    connected = QtCore.pyqtSignal(dict)        # device summary
    disconnected = QtCore.pyqtSignal()
    battery = QtCore.pyqtSignal(int)           # percent
    params = QtCore.pyqtSignal(dict)           # {"emg_rate": int, "filters": {key: bool}}
    error = QtCore.pyqtSignal(str)

    def __init__(self, hub: DataHub, emg_rate: int = DEFAULT_EMG_RATE):
        super().__init__()
        self.hub = hub
        self.target_emg_rate = emg_rate
        self.settings_locked = False

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
        raise NotImplementedError

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
    def __init__(self, hub: DataHub, emg_rate: int = DEFAULT_EMG_RATE,
                 package_samples: int = PACKAGE_SAMPLE_COUNT):
        super().__init__(hub, emg_rate)
        import sensor
        self._sdk = sensor
        self._DataType = sensor.DataType
        self._package_samples = package_samples
        self._ctrl = sensor.SensorController()
        self._ctrl.on_sensor_scan_result = self._on_scan
        self._devices = {}
        self._sensor = None
        self._unhandled_types = set()
        self._stream_id = 0

    @property
    def is_connected(self) -> bool:
        s = self._sensor
        return s is not None and s.isReady

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
        if not self._ctrl.isScanning:
            self._ctrl.startScan(SCAN_PERIOD_MS)
        self.status.emit("Scanning...")

    def stop_scan(self):
        if self._ctrl.isScanning:
            self._ctrl.stopScan()

    def _on_scan(self, device_list):
        for d in device_list:
            self._devices[d.Address] = d
        self.devices_found.emit([(d.Name, d.Address, d.RSSI) for d in device_list])

    # -- connection -----------------------------------------------------------
    def connect_device(self, address: str):
        self._sdk.submit(self._connect_worker, address)

    def _connect_worker(self, address: str):
        try:
            device = self._devices.get(address)
            if device is None:
                self.error.emit(f"Unknown device {address}; scan first")
                return
            self.stop_scan()
            sensor = self._ctrl.requireSensor(device)
            if sensor is None:
                self.error.emit("Failed to create sensor profile")
                return
            sensor.on_sensor_notify_data = self._on_data
            sensor.on_state_change = self._on_state
            sensor.on_power_changed = lambda s, p: p >= 0 and self.battery.emit(int(p))
            sensor.on_error_callback = lambda s, msg: self.error.emit(f"SDK: {msg}")
            sensor.on_device_info_update = lambda s, info: self._emit_params(s)

            self.status.emit(f"Connecting to {device.Name}...")
            if not sensor.isReady and not sensor.connect():
                self.error.emit(f"Failed to connect to {device.Name}")
                return
            if not sensor.hasInited and not sensor.init(self._package_samples, POWER_REFRESH_MS):
                self.error.emit(f"Failed to initialize {device.Name}")
                return
            self._sensor = sensor

            rate = sensor.getParam("EMG_SAMPLE_RATE")
            if str(rate) != str(self.target_emg_rate):
                res = sensor.setParam("EMG_SAMPLE_RATE", str(self.target_emg_rate))
                if str(res).startswith("Error"):
                    self.error.emit(f"EMG_SAMPLE_RATE={self.target_emg_rate}: {res}")
            self._ensure_notifications(sensor, ("NTF_EMG", "NTF_IMU"))

            self.hub.reset()
            self._stream_id += 1
            if not sensor.isDataTransfering and not sensor.startDataNotification():
                self.error.emit("Failed to start data stream")
                return

            info = sensor.getDeviceInfo()
            summary = {
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
            }
            self.connected.emit(summary)
            self._emit_params(sensor)
            power = sensor.getBatteryLevel()
            if power >= 0:
                self.battery.emit(int(power))
        except Exception as e:
            traceback.print_exc()
            self.error.emit(f"Connect failed: {e}")

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
        sensor = self._sensor
        if sensor is None:
            return

        def work():
            try:
                if sensor.isDataTransfering:
                    sensor.stopDataNotification()
                sensor.disconnect()
            except Exception as e:
                self.error.emit(f"Disconnect failed: {e}")
        self._sdk.submit(work)

    def _on_state(self, sensor, state):
        if state == self._sdk.DeviceStateEx.Disconnected:
            if sensor is self._sensor:
                self._sensor = None
            self.disconnected.emit()

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

    def shutdown(self):
        try:
            self.stop_scan()
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
    """

    IMU_RATE = 50

    def __init__(self, seed=None, rate: int = DEFAULT_EMG_RATE,
                 package: int = PACKAGE_SAMPLE_COUNT, faults: Optional[SimFaults] = None,
                 filters: Optional[dict] = None):
        self.rng = np.random.default_rng(seed)
        self.rate = rate
        self.package = package
        self.faults = faults or SimFaults()
        self.filters = filters if filters is not None else {k: False for k in FILTERS}
        fixed = np.random.default_rng(7)
        self._gestures = fixed.uniform(0.0, 1.0, (6, EMG_CHANNELS)) ** 3
        self._offsets = fixed.uniform(-150, 150, EMG_CHANNELS)
        self._base_imp = np.array([180, 220, 260, 300, 350, 700, 240, 1400], float) * 1000
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
    """Simulator with the Source interface, paced in real time on a thread."""

    ADDRESS = "SIM:00:00:00:00:00"

    def __init__(self, hub: DataHub, emg_rate: int = DEFAULT_EMG_RATE,
                 package_samples: int = PACKAGE_SAMPLE_COUNT, seed=None,
                 faults: Optional[SimFaults] = None):
        super().__init__(hub, emg_rate)
        self._package = package_samples
        self._seed = seed
        self.faults = faults or SimFaults()
        self._thread = None
        self._stop = threading.Event()
        self._filters = {k: False for k in FILTERS}
        self._gen: Optional[SimGenerator] = None

    @property
    def is_connected(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def backend_name(self) -> str:
        return "simulator"

    def start_scan(self):
        self.status.emit("Scanning (simulator)...")
        QtCore.QTimer.singleShot(300, lambda: self.devices_found.emit(
            [("gForceUltra-SIM", self.ADDRESS, -42)]))

    def stop_scan(self):
        pass

    def connect_device(self, address: str):
        if self.is_connected:
            return
        self.hub.reset()
        self._gen = SimGenerator(self._seed, self.target_emg_rate, self._package,
                                 self.faults, self._filters)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="sim-source", daemon=True)
        self._thread.start()
        self.connected.emit({"name": "gForceUltra-SIM", "address": self.ADDRESS,
                             "model": "gForceUltra (simulated)", "firmware": "-",
                             "hardware": "-", "emg_channels": EMG_CHANNELS,
                             "imu_channels": IMU_CHANNELS, "mtu": 0,
                             "backend": "simulator", "package_samples": self._package})
        self._emit_params()
        self.battery.emit(87)

    def disconnect_device(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None
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
