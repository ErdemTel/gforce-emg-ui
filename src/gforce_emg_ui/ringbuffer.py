"""Thread-safe multi-channel ring buffer.

Written from the data-source thread, read (as a time-ordered copy) from the UI thread.
"""

import threading

import numpy as np


class RingBuffer:
    def __init__(self, channels: int, capacity: int, dtype=np.float32):
        self.channels = channels
        self.capacity = capacity
        self._data = np.zeros((channels, capacity), dtype=dtype)
        self._write = 0           # next write position
        self._count = 0           # valid samples (<= capacity)
        self._total = 0           # samples ever written
        self.version = 0          # bumps on every change (lets views skip redraws)
        self._lock = threading.Lock()

    def clear(self):
        with self._lock:
            self._data.fill(0)
            self._write = 0
            self._count = 0
            self._total = 0
            self.version += 1

    @property
    def total(self) -> int:
        return self._total

    def append(self, block: np.ndarray):
        """Append a (channels, n) block."""
        n = block.shape[1]
        if n == 0:
            return
        with self._lock:
            if n >= self.capacity:
                self._data[:] = block[:, -self.capacity:]
                self._write = 0
            else:
                end = self._write + n
                if end <= self.capacity:
                    self._data[:, self._write:end] = block
                else:
                    first = self.capacity - self._write
                    self._data[:, self._write:] = block[:, :first]
                    self._data[:, :n - first] = block[:, first:]
                self._write = end % self.capacity
            self._count = min(self._count + n, self.capacity)
            self._total += n
            self.version += 1

    def latest(self, n: int) -> np.ndarray:
        """Return the newest min(n, available) samples, oldest first, as a copy."""
        with self._lock:
            n = min(n, self._count)
            start = self._write - n
            if start >= 0:
                return self._data[:, start:self._write].copy()
            return np.concatenate((self._data[:, start:], self._data[:, :self._write]), axis=1)
