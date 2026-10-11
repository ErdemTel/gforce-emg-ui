"""Central data hub: receives blocks from a source (real band or simulator),
keeps ring buffers for display, tracks stream statistics, and fans blocks out
to subscribers (recorder, future real-time decoders).

push_* runs on the source thread and must stay fast: copy into ring buffers,
update counters, call subscribers. Subscribers that do heavy work must queue it.
"""

import collections
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .config import (DEFAULT_EMG_RATE, EMG_BUFFER_SECONDS, EMG_CHANNELS, EMG_RATES,
                     IMU_BUFFER_SECONDS, IMU_CHANNELS)
from .ringbuffer import RingBuffer


@dataclass
class EmgBlock:
    data: np.ndarray            # (8, n) float32
    sample_index: np.ndarray    # (n,) int64, device sample counter
    device_time: np.ndarray     # (n,) float64, SDK absolute timestamp (s)
    lost: np.ndarray            # (n,) bool, sample was lost in transit (zero-filled)
    host_time: float            # time.time() when the batch reached the host
    sample_rate: float
    impedance: Optional[np.ndarray] = None   # (8,) Ohm, latest per channel
    lost_packages: int = 0


@dataclass
class ImuBlock:
    data: np.ndarray            # (6, n) float32: ACC xyz, GYRO xyz
    sample_index: np.ndarray
    device_time: np.ndarray
    host_time: float
    sample_rate: float


class _RateMeter:
    def __init__(self, window_s: float = 2.0):
        self._window = window_s
        self._events = collections.deque()
        self._lock = threading.Lock()

    def add(self, t: float, n: int):
        with self._lock:
            self._events.append((t, n))
            while self._events and t - self._events[0][0] > self._window:
                self._events.popleft()

    def rate(self, now: float) -> float:
        with self._lock:
            while self._events and now - self._events[0][0] > self._window:
                self._events.popleft()
            if len(self._events) < 2:
                return 0.0
            span = now - self._events[0][0]
            # The first batch's samples arrived before the window start
            return sum(n for _, n in list(self._events)[1:]) / max(span, 1e-3)

    def clear(self):
        with self._lock:
            self._events.clear()


class DataHub:
    def __init__(self):
        self.emg = RingBuffer(EMG_CHANNELS, EMG_BUFFER_SECONDS * max(EMG_RATES))
        self.imu = RingBuffer(IMU_CHANNELS, IMU_BUFFER_SECONDS * 1000)
        self.emg_rate: float = DEFAULT_EMG_RATE
        self.imu_rate: float = 50.0
        self.impedance = np.full(EMG_CHANNELS, np.nan)
        self.last_emg_host_time: float = 0.0
        self.lost_packages = 0
        self.lost_samples = 0
        self._emg_meter = _RateMeter()
        self._imu_meter = _RateMeter()
        self._emg_subscribers: List[Callable[[EmgBlock], None]] = []
        self._imu_subscribers: List[Callable[[ImuBlock], None]] = []

    # -- subscription ---------------------------------------------------------
    def subscribe_emg(self, fn: Callable[[EmgBlock], None]):
        self._emg_subscribers.append(fn)

    def unsubscribe_emg(self, fn):
        if fn in self._emg_subscribers:
            self._emg_subscribers.remove(fn)

    def subscribe_imu(self, fn: Callable[[ImuBlock], None]):
        self._imu_subscribers.append(fn)

    def unsubscribe_imu(self, fn):
        if fn in self._imu_subscribers:
            self._imu_subscribers.remove(fn)

    # -- source side ----------------------------------------------------------
    def push_emg(self, block: EmgBlock):
        if block.sample_rate and block.sample_rate != self.emg_rate:
            self.emg_rate = float(block.sample_rate)
            self.emg.clear()
            self._emg_meter.clear()
        self.emg.append(block.data)
        if block.impedance is not None:
            self.impedance = block.impedance
        self.lost_packages += block.lost_packages
        self.lost_samples += int(np.count_nonzero(block.lost))
        self.last_emg_host_time = block.host_time
        self._emg_meter.add(block.host_time, block.data.shape[1])
        for fn in list(self._emg_subscribers):
            fn(block)

    def push_imu(self, block: ImuBlock):
        if block.sample_rate and block.sample_rate != self.imu_rate:
            self.imu_rate = float(block.sample_rate)
            self.imu.clear()
            self._imu_meter.clear()
        self.imu.append(block.data)
        self._imu_meter.add(block.host_time, block.data.shape[1])
        for fn in list(self._imu_subscribers):
            fn(block)

    # -- UI side --------------------------------------------------------------
    def reset(self):
        self.emg.clear()
        self.imu.clear()
        self.impedance = np.full(EMG_CHANNELS, np.nan)
        self.lost_packages = 0
        self.lost_samples = 0
        self.last_emg_host_time = 0.0
        self._emg_meter.clear()
        self._imu_meter.clear()

    def emg_actual_rate(self) -> float:
        return self._emg_meter.rate(time.time())

    def imu_actual_rate(self) -> float:
        return self._imu_meter.rate(time.time())
