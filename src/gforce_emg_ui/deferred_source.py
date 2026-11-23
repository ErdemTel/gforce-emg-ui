"""Initialize the real EMG SDK only after an explicit Scan or Connect action."""

from .config import DEFAULT_EMG_RATE, PACKAGE_SAMPLE_COUNT
from .sources import FAILED, IDLE, Source, SdkSource


class DeferredSdkSource(Source):
    """Keep manual robot control usable when the EMG SDK or dongle is absent."""

    def __init__(self, hub, emg_rate=DEFAULT_EMG_RATE,
                 package_samples=PACKAGE_SAMPLE_COUNT, factory=SdkSource):
        super().__init__(hub, emg_rate)
        self._package_samples = package_samples
        self._factory = factory
        self._inner = None
        self._closed = False

    def _ensure(self):
        if self._closed:
            return None
        if self._inner is not None:
            return self._inner
        try:
            inner = self._factory(self.hub, self.target_emg_rate, self._package_samples)
        except Exception as error:
            self._set_state(FAILED)
            self.error.emit(f"EMG unavailable: {error}. Manual robot control remains available.")
            return None
        self._inner = inner
        inner.lock_settings(self.settings_locked)
        inner.state_changed.connect(self._set_state)
        for name in ("devices_found", "status", "connected", "disconnected",
                     "battery", "params", "error"):
            getattr(inner, name).connect(getattr(self, name).emit)
        return inner

    def backend_name(self):
        return self._inner.backend_name() if self._inner is not None else "not initialized"

    def lock_settings(self, locked):
        super().lock_settings(locked)
        if self._inner is not None:
            self._inner.lock_settings(locked)

    def start_scan(self):
        inner = self._ensure()
        if inner is not None:
            try:
                inner.start_scan()
            except Exception as error:
                self._set_state(FAILED)
                self.error.emit(f"EMG scan failed: {error}")

    def stop_scan(self):
        if self._inner is not None:
            self._inner.stop_scan()

    def connect_device(self, address):
        inner = self._ensure()
        if inner is not None:
            try:
                inner.connect_device(address)
            except Exception as error:
                self._set_state(FAILED)
                self.error.emit(f"EMG connection failed: {error}")

    def disconnect_device(self):
        if self._inner is not None:
            self._inner.disconnect_device()
        else:
            self._set_state(IDLE)

    def set_emg_rate(self, rate):
        if not self._check_unlocked("Sample rate"):
            return False
        self.target_emg_rate = rate
        if self._inner is not None:
            return self._inner.set_emg_rate(rate)
        return True

    def set_filter(self, key, on):
        return self._inner.set_filter(key, on) if self._inner is not None else False

    def shutdown(self):
        self._closed = True
        if self._inner is not None:
            self._inner.shutdown()
