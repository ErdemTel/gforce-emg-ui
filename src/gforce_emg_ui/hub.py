"""Central data hub: receives blocks from a source (real band or simulator),
keeps ring buffers for display, tracks stream statistics, and fans blocks out
to subscribers (recorder, future real-time decoders).

push_* runs on the source thread and must stay fast: copy into ring buffers,
update counters, call subscribers. Subscribers that do heavy work must queue it.

Data integrity rules:
  * invalid samples (lost in transit, missing channels) are NaN in block.data and
    False in block.valid; they are never replaced by zeros
  * each block is assigned a segment: a continuous run of sample indices from one
    stream at one rate. A new segment starts on a new stream_id (reconnection),
    a rate change, an index reset, or a forward jump larger than MAX_GAP_S
  * a small forward jump inside a segment is shown as an explicit NaN gap in the
    display buffer and counted in missing_samples; nothing is fabricated for recording
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

MAX_GAP_S = 1.0     # larger forward index jumps start a new segment instead of a NaN gap


@dataclass
class EmgBlock:
    data: np.ndarray            # (8, n) float32, NaN where not valid
    valid: np.ndarray           # (8, n) bool, False for lost samples / missing channels
    sample_index: np.ndarray    # (n,) int64, device sample counter
    device_time: np.ndarray     # (n,) float64, SDK timestamp (= index / nominal rate, not a clock)
    lost: np.ndarray            # (n,) bool, SDK reported the sample lost in transit
    host_time: float            # time.time() when the batch reached the host (for alignment)
    host_mono: float            # time.monotonic() at the same moment (for durations/latency)
    sample_rate: float
    impedance: Optional[np.ndarray] = None   # (8,) Ohm, latest per channel, NaN if unknown
    lost_packages: int = 0
    stream_id: int = 0          # changes on every (re)connection / stream start
    segment: int = 0            # assigned by the hub: continuous run of sample indices


@dataclass
class ImuBlock:
    data: np.ndarray            # (6, n) float32: ACC xyz, GYRO xyz, NaN where not valid
    valid: np.ndarray           # (6, n) bool
    sample_index: np.ndarray
    device_time: np.ndarray
    host_time: float
    host_mono: float
    sample_rate: float
    stream_id: int = 0
    segment: int = 0


class _Continuity:
    """Assigns segments and detects index gaps for one stream."""

    def __init__(self):
        self.segment = 0
        self.last_index: Optional[int] = None
        self.stream_id = None
        self.rate = None
        self.anomalies = 0          # non-consecutive indices inside one block

    def update(self, block) -> tuple:
        idx = block.sample_index
        new = (self.last_index is None or block.stream_id != self.stream_id
               or block.sample_rate != self.rate)
        gap = 0
        if not new:
            d = int(idx[0]) - self.last_index
            max_gap = int(MAX_GAP_S * (block.sample_rate or 1))
            if d <= 0 or d - 1 > max_gap:
                new = True
            else:
                gap = d - 1
        if new:
            self.segment += 1
            self.stream_id = block.stream_id
            self.rate = block.sample_rate
        if len(idx) > 1 and np.any(np.diff(idx) != 1):
            self.anomalies += 1
        self.last_index = int(idx[-1])
        block.segment = self.segment
        return new, gap


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
        self.missing_samples = 0
        self._emg_cont = _Continuity()
        self._imu_cont = _Continuity()
        self._emg_meter = _RateMeter()
        self._imu_meter = _RateMeter()
        self._emg_subscribers: List[Callable[[EmgBlock], None]] = []
        self._imu_subscribers: List[Callable[[ImuBlock], None]] = []

    @property
    def emg_segment(self) -> int:
        return self._emg_cont.segment

    @property
    def imu_segment(self) -> int:
        return self._imu_cont.segment

    @property
    def index_anomalies(self) -> int:
        return self._emg_cont.anomalies + self._imu_cont.anomalies

    # -- subscription ---------------------------------------------------------
    def subscribe_emg(self, fn: Callable[[EmgBlock], None], name: Optional[str] = None):
        self._emg_subscribers.append(fn)

    def unsubscribe_emg(self, fn):
        if fn in self._emg_subscribers:
            self._emg_subscribers.remove(fn)

    def subscribe_imu(self, fn: Callable[[ImuBlock], None], name: Optional[str] = None):
        self._imu_subscribers.append(fn)

    def unsubscribe_imu(self, fn):
        if fn in self._imu_subscribers:
            self._imu_subscribers.remove(fn)

    # -- source side ----------------------------------------------------------
    @staticmethod
    def _append_with_gap(ring: RingBuffer, data: np.ndarray, gap: int):
        if gap > 0:
            ring.append(np.full((ring.channels, min(gap, ring.capacity)), np.nan, np.float32))
        ring.append(data)

    def push_emg(self, block: EmgBlock):
        new, gap = self._emg_cont.update(block)
        if new:
            if block.sample_rate:
                self.emg_rate = float(block.sample_rate)
            self.emg.clear()
            self._emg_meter.clear()
        self.missing_samples += gap
        self._append_with_gap(self.emg, block.data, gap)
        if block.impedance is not None:
            self.impedance = block.impedance
        self.lost_packages += block.lost_packages
        self.lost_samples += int(np.count_nonzero(block.lost))
        self.last_emg_host_time = block.host_time
        self._emg_meter.add(block.host_mono, block.data.shape[1])
        for fn in list(self._emg_subscribers):
            fn(block)

    def push_imu(self, block: ImuBlock):
        new, gap = self._imu_cont.update(block)
        if new:
            if block.sample_rate:
                self.imu_rate = float(block.sample_rate)
            self.imu.clear()
            self._imu_meter.clear()
        self._append_with_gap(self.imu, block.data, gap)
        self._imu_meter.add(block.host_mono, block.data.shape[1])
        for fn in list(self._imu_subscribers):
            fn(block)

    # -- UI side --------------------------------------------------------------
    def reset(self):
        """Clear display state for a new connection (segment numbering continues)."""
        self.emg.clear()
        self.imu.clear()
        self.impedance = np.full(EMG_CHANNELS, np.nan)
        self.lost_packages = 0
        self.lost_samples = 0
        self.missing_samples = 0
        self.last_emg_host_time = 0.0
        self._emg_meter.clear()
        self._imu_meter.clear()

    def emg_actual_rate(self) -> float:
        return self._emg_meter.rate(time.monotonic())

    def imu_actual_rate(self) -> float:
        return self._imu_meter.rate(time.monotonic())
