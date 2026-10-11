import csv
import json
import time

import numpy as np

from gforce_emg_ui.contact import GOOD, MARGINAL, POOR, UNKNOWN, contact_level
from gforce_emg_ui.hub import DataHub, EmgBlock, ImuBlock
from gforce_emg_ui.recorder import Recorder
from gforce_emg_ui.ringbuffer import RingBuffer


def test_ringbuffer_wraps_in_order():
    rb = RingBuffer(2, 5)
    for start in range(0, 12, 3):
        block = np.vstack([np.arange(start, start + 3), -np.arange(start, start + 3)])
        rb.append(block)
    out = rb.latest(5)
    assert out.shape == (2, 5)
    assert out[0].tolist() == [7, 8, 9, 10, 11]
    assert out[1].tolist() == [-7, -8, -9, -10, -11]
    assert rb.latest(2)[0].tolist() == [10, 11]
    assert rb.total == 12


def test_ringbuffer_partial_and_oversized():
    rb = RingBuffer(1, 4)
    rb.append(np.array([[1, 2]]))
    assert rb.latest(10)[0].tolist() == [1, 2]
    rb.append(np.arange(10).reshape(1, 10))
    assert rb.latest(4)[0].tolist() == [6, 7, 8, 9]


def test_contact_levels():
    assert contact_level(None) == UNKNOWN
    assert contact_level(float("nan")) == UNKNOWN
    assert contact_level(200_000) == GOOD
    assert contact_level(500_000) == GOOD
    assert contact_level(750_000) == MARGINAL
    assert contact_level(1_500_000) == POOR


def _emg_block(start, n=16, rate=1000.0, imp=None):
    return EmgBlock(
        data=np.tile(np.arange(start, start + n, dtype=np.float32), (8, 1)),
        sample_index=np.arange(start, start + n),
        device_time=np.arange(start, start + n) / rate,
        lost=np.zeros(n, bool),
        host_time=time.time(),
        sample_rate=rate,
        impedance=np.full(8, 300_000.0) if imp is None else imp,
    )


def test_hub_clears_on_rate_change_and_counts_lost():
    hub = DataHub()
    hub.push_emg(_emg_block(0))
    assert hub.emg.total == 16
    b = _emg_block(16, rate=500.0)
    b.lost[:3] = True
    b.lost_packages = 1
    hub.push_emg(b)
    assert hub.emg_rate == 500.0
    assert hub.emg.total == 16          # cleared on rate change, then refilled
    assert hub.lost_samples == 3 and hub.lost_packages == 1


def test_recorder_writes_session(tmp_path):
    hub = DataHub()
    rec = Recorder(hub, tmp_path)
    folder = rec.start("subj01 fist", {"emg_rate": 1000})
    assert folder.name.endswith("subj01_fist")
    for i in range(10):
        imp = np.full(8, 300_000.0) if i < 5 else np.full(8, 1_200_000.0)
        hub.push_emg(_emg_block(i * 16, imp=imp))
        if i == 3:
            rec.add_marker("fist_on")
    hub.push_imu(ImuBlock(data=np.ones((6, 2), np.float32), sample_index=np.arange(2),
                          device_time=np.zeros(2), host_time=time.time(), sample_rate=50.0))
    rec.stop()
    hub.push_emg(_emg_block(999))       # after stop: must not be written

    rows = list(csv.reader(open(folder / "emg.csv")))
    assert rows[0][:4] == ["sample_index", "device_time", "host_time", "lost"]
    assert len(rows) == 1 + 160
    assert int(rows[-1][0]) == 159 and float(rows[-1][4]) == 159.0
    assert len(list(csv.reader(open(folder / "imu.csv")))) == 3
    contact = list(csv.reader(open(folder / "contact.csv")))
    assert len(contact) == 3            # header + two distinct impedance states
    markers = list(csv.reader(open(folder / "markers.csv")))
    assert markers[1][1:] == ["63", "fist_on"]
    meta = json.load(open(folder / "session.json"))
    assert meta["emg_samples"] == 160 and meta["markers"] == 1 and meta["emg_rate"] == 1000
