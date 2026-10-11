"""Data-integrity regression tests."""

import csv

import numpy as np
import pytest

from conftest import make_emg, wait_until
from gforce_emg_ui.features import ac_rms
from gforce_emg_ui.hub import DataHub
from gforce_emg_ui.recorder import SAVED, Recorder


def _rows(path):
    return list(csv.DictReader(open(path)))


def _record(tmp_path, blocks, hub=None):
    hub = hub or DataHub()
    rec = Recorder(hub, tmp_path)
    s = rec.start("t")
    for b in blocks:
        hub.push_emg(b)
    s.request_stop()
    assert s.wait(5) and s.state == SAVED
    return hub, s


def test_segments_and_sample_indices_preserved(tmp_path):
    blocks = [make_emg(0), make_emg(16), make_emg(0, stream_id=2)]   # reconnect: new stream
    hub, s = _record(tmp_path, blocks)
    rows = _rows(s.folder / "emg.csv")
    assert [int(r["sample_index"]) for r in rows] == list(range(32)) + list(range(16))
    assert [int(r["segment"]) for r in rows] == [1] * 32 + [2] * 16
    assert hub.emg_segment == 2


def test_index_reset_and_rate_change_start_new_segment():
    hub = DataHub()
    hub.push_emg(make_emg(100))
    hub.push_emg(make_emg(5))                      # index went backwards
    assert hub.emg_segment == 2
    hub.push_emg(make_emg(21, rate=500.0))         # rate change
    assert hub.emg_segment == 3


def test_index_gap_is_explicit_in_display_and_not_fabricated_in_recording(tmp_path):
    blocks = [make_emg(0), make_emg(26)]           # 10 samples missing (16..25)
    hub, s = _record(tmp_path, blocks)
    assert hub.missing_samples == 10
    assert hub.emg_segment == 1                    # small gap: same segment
    shown = hub.emg.latest(1000)
    assert shown.shape[1] == 42
    assert np.isnan(shown[:, 16:26]).all()
    assert np.isfinite(shown[:, :16]).all() and np.isfinite(shown[:, 26:]).all()
    rows = _rows(s.folder / "emg.csv")
    assert len(rows) == 32                         # only real samples written
    assert [int(r["sample_index"]) for r in rows][15:17] == [15, 26]


def test_lost_samples_masked_not_zero(tmp_path):
    lost = np.zeros(16, bool)
    lost[3:6] = True
    hub, s = _record(tmp_path, [make_emg(0, lost=lost)])
    assert hub.lost_samples == 3
    assert np.isnan(hub.emg.latest(16)[:, 3:6]).all()
    rows = _rows(s.folder / "emg.csv")
    for r in rows[3:6]:
        assert r["lost"] == "1" and r["valid_mask"] == "0"
        assert all(r[f"ch{c}"] == "" for c in range(1, 9))
    assert rows[0]["lost"] == "0" and rows[0]["valid_mask"] == "255"


class _FakeSensorData:
    """Mimics sensor.SensorData.as_numpy() for a batch with `ch` channels."""

    def __init__(self, ch, n, start=0, rate=1000.0, lost_at=()):
        dt = np.dtype([("absTimeStampInSec", "<f8"), ("channelIndex", "<i4"),
                       ("sampleIndex", "<i4"), ("rawData", "<i4"), ("data", "<f4"),
                       ("impedance", "<f4"), ("saturation", "<f4"), ("isLost", "u1")])
        a = np.zeros((ch, n), dt)
        a["sampleIndex"] = np.arange(start, start + n)
        a["data"] = 7.0
        a["impedance"] = 250_000.0
        for i in lost_at:
            a["isLost"][:, i] = 1
        self._a = a
        self.sampleRate = rate
        self.lostPackageCount = 0
        self.channelCount = ch
        self.sampleCount = n

    def as_numpy(self):
        return self._a


def test_sdk_missing_channels_are_invalid_not_zero():
    from gforce_emg_ui.sources import SdkSource
    b = SdkSource._emg_block(_FakeSensorData(6, 4, lost_at=(1,)), 0.0, 0.0, 1)
    assert b.data.shape == (8, 4)
    assert np.isnan(b.data[6:]).all() and not b.valid[6:].any()
    assert np.isnan(b.impedance[6:]).all()
    assert not b.valid[:, 1].any() and np.isnan(b.data[:, 1]).all()
    assert (b.data[:6, 0] == 7.0).all() and b.valid[:6, 0].all()


def test_simulator_missing_channels_fault_is_invalid():
    from gforce_emg_ui.sources import SimFaults, SimGenerator
    g = SimGenerator(seed=3, rate=1000, package=16, faults=SimFaults(missing_channels=(2, 7)))
    b = g.next_emg()
    assert np.isnan(b.data[[2, 7]]).all() and not b.valid[[2, 7]].any()
    assert np.isfinite(b.data[[0, 1, 3, 4, 5, 6]]).all()


def test_features_exclude_invalid_samples():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 200)).astype(np.float32) * 10
    clean = ac_rms(x)
    corrupted = x.copy()
    corrupted[:, 50:60] = np.nan                    # invalid stretch
    expect = ac_rms(np.delete(x, np.s_[50:60], axis=1))
    np.testing.assert_allclose(ac_rms(corrupted), expect, rtol=1e-6)
    assert not np.allclose(clean, ac_rms(np.where(np.isnan(corrupted), 0, corrupted)))
    corrupted[3, :190] = np.nan                     # too few valid samples
    assert np.isnan(ac_rms(corrupted)[3])


def test_plots_show_explicit_gaps(qapp):
    from gforce_emg_ui.ui.emg_detail import DetailDashboard
    from gforce_emg_ui.ui.emg_equal import EqualScaleDashboard
    hub = DataHub()
    hub.push_emg(make_emg(0, n=400))
    hub.push_emg(make_emg(450, n=400))             # 50-sample gap
    for dash in (EqualScaleDashboard(hub), DetailDashboard(hub)):
        dash.refresh()
        curves = dash.curves if hasattr(dash, "curves") else dash.wave_curves
        y = curves[0].yData
        assert np.isnan(y).sum() == 50
        assert curves[0].opts["connect"] == "finite"


def test_settings_locked_while_recording(qapp, tmp_path):
    from gforce_emg_ui.sources import SimSource
    from gforce_emg_ui.ui.main_window import MainWindow
    hub = DataHub()
    src = SimSource(hub, seed=1)
    win = MainWindow(hub, src, tmp_path)
    try:
        src.connect_device(SimSource.ADDRESS)
        assert wait_until(lambda: hub.emg.total > 0, 5, qapp)
        qapp.processEvents()
        win._toggle_recording()
        assert src.settings_locked
        assert not any(cb.isEnabled() for cb in win.filter_checks.values())
        assert not any(rb.isEnabled() for rb in win.rate_buttons.values())
        assert src.set_filter("FILTER_HPF", True) is False
        assert src.set_emg_rate(500) is False
        assert src.target_emg_rate == 1000 and not src._filters["FILTER_HPF"]
        win._toggle_recording()
        assert wait_until(lambda: (win._refresh_status() or True)
                          and not src.settings_locked, 10, qapp)
        assert src.set_filter("FILTER_HPF", True) is True
    finally:
        src.disconnect_device()
        win.close()


def test_csv_float32_values_round_trip_exactly(tmp_path):
    vals = np.array([343750.03, -376953.16, 1.2345678e-3, -0.0001234567, 98765.4321,
                     3.4028235e38, 1e-30, 123456.79], np.float32)
    b = make_emg(0, n=1)
    b.data[:, 0] = vals
    _, s = _record(tmp_path, [b])
    row = _rows(s.folder / "emg.csv")[0]
    got = np.array([float(row[f"ch{c}"]) for c in range(1, 9)], np.float32)
    assert (got == vals).all()
    assert abs(float(row["host_time"]) - b.host_time) < 1e-6


def test_contact_missing_values_logged_once_as_empty(tmp_path):
    imp_partial = np.array([2e5, np.nan, 3e5, 3e5, 3e5, 3e5, 3e5, np.nan])
    blocks = [make_emg(i * 16, imp=imp_partial) for i in range(5)]
    blocks.append(make_emg(80, imp=np.full(8, np.nan)))
    _, s = _record(tmp_path, blocks)
    rows = _rows(s.folder / "contact.csv")
    assert len(rows) == 2                          # one per distinct state, NaN == NaN
    assert rows[0]["imp2"] == "" and rows[0]["imp8"] == "" and rows[0]["imp1"] == "200000"
    assert all(rows[1][f"imp{c}"] == "" for c in range(1, 9))
    assert rows[0]["segment"] == "1"
