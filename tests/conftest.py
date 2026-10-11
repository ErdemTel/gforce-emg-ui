import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from gforce_emg_ui.hub import EmgBlock, ImuBlock


@pytest.fixture(scope="session")
def qapp():
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def wait_until(cond, timeout=5.0, app=None):
    """Poll cond() (processing Qt events if app is given); True if it became true."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if app is not None:
            app.processEvents()
        if cond():
            return True
        time.sleep(0.005)
    if app is not None:
        app.processEvents()
    return cond()


def make_emg(start, n=16, rate=1000.0, value=None, lost=None, valid=None,
             imp=None, stream_id=1, channels=8):
    """EMG block with sample indices start..start+n-1; data = sample index unless value given."""
    idx = np.arange(start, start + n, dtype=np.int64)
    data = np.tile(idx.astype(np.float32), (channels, 1)) if value is None \
        else np.full((channels, n), value, np.float32)
    lost = np.zeros(n, bool) if lost is None else np.asarray(lost, bool)
    valid = np.ones((channels, n), bool) if valid is None else np.asarray(valid, bool)
    valid = valid & ~lost[None, :]
    data = np.where(valid, data, np.nan).astype(np.float32)
    return EmgBlock(
        data=data, valid=valid, sample_index=idx, device_time=idx / rate,
        lost=lost, host_time=time.time(), host_mono=time.monotonic(),
        sample_rate=rate,
        impedance=np.full(channels, 300_000.0) if imp is None else np.asarray(imp, float),
        stream_id=stream_id)


def make_imu(start, n=2, rate=50.0, stream_id=1):
    idx = np.arange(start, start + n, dtype=np.int64)
    return ImuBlock(
        data=np.ones((6, n), np.float32), valid=np.ones((6, n), bool),
        sample_index=idx, device_time=idx / rate, host_time=time.time(),
        host_mono=time.monotonic(), sample_rate=rate, stream_id=stream_id)
