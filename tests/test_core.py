import numpy as np

from conftest import make_emg

from gforce_emg_ui.contact import GOOD, MARGINAL, POOR, UNKNOWN, contact_level
from gforce_emg_ui.hub import DataHub
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


def test_hub_clears_on_rate_change_and_counts_lost():
    hub = DataHub()
    hub.push_emg(make_emg(0))
    assert hub.emg.total == 16
    lost = np.zeros(16, bool)
    lost[:3] = True
    b = make_emg(16, rate=500.0, lost=lost)
    b.lost_packages = 1
    hub.push_emg(b)
    assert hub.emg_rate == 500.0
    assert hub.emg.total == 16          # cleared on rate change, then refilled
    assert hub.lost_samples == 3 and hub.lost_packages == 1
