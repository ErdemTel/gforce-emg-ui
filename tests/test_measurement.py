"""Measurement-consistency regression tests."""

import json
import time

import numpy as np

from conftest import make_emg, wait_until
from gforce_emg_ui.features import ac_rms, rms_shares
from gforce_emg_ui.hub import DataHub
from gforce_emg_ui.recorder import Recorder


def _noisy_hub(offset=1000.0, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    hub = DataHub()
    for start in range(0, n, 16):
        b = make_emg(start)
        b.data[:] = offset + rng.standard_normal((8, 16)).astype(np.float32) * np.arange(1, 9)[:, None]
        hub.push_emg(b)
    return hub


def test_rms_independent_of_display_settings(qapp):
    from gforce_emg_ui.ui.emg_equal import EqualScaleDashboard
    hub = _noisy_hub()
    dash = EqualScaleDashboard(hub)
    results = []
    for dc in (True, False):
        for w in range(dash.window_combo.count()):
            dash.dc_check.setChecked(dc)
            dash.window_combo.setCurrentIndex(w)
            dash.refresh()
            results.append(dash.last_rms.copy())
    for r in results[1:]:
        np.testing.assert_allclose(r, results[0], rtol=1e-9)
    m = int(dash.rms_combo.currentData() * hub.emg_rate / 1000)
    np.testing.assert_allclose(results[0], ac_rms(hub.emg.latest(m)), rtol=1e-9)
    # AC RMS: the 1000-unit DC offset must not dominate
    assert results[0][0] < 3 and results[0][7] > 5


def test_relative_rms_labels_are_accurate(qapp):
    from gforce_emg_ui.ui.emg_equal import EqualScaleDashboard
    rms = np.array([1, 1, 2, 4, np.nan, 0, 1, 1], float)
    shares = rms_shares(rms)
    assert np.isnan(shares[4])
    assert abs(np.nansum(shares) - 100) < 1e-9
    assert shares[3] == 40.0
    hub = _noisy_hub()
    dash = EqualScaleDashboard(hub)
    dash.refresh()
    title = dash.rms_plot.getPlotItem().titleLabel.text
    assert "share of summed RMS" in title
    assert "AC" in title and "ms" in title
    text = dash.bar_texts[0].textItem.toPlainText()
    assert text.endswith("%")


def test_rate_meter_uses_monotonic_clock():
    hub = DataHub()
    t0 = time.monotonic()
    for k in range(60):
        b = make_emg(k * 16)
        b.host_mono = t0 + k * 0.016
        b.host_time = 1e9 - k * 3600.0          # wall clock jumping backwards
        hub.push_emg(b)
    rate = hub.emg_actual_rate(now=t0 + 59 * 0.016)
    assert 950 < rate < 1050


def test_recording_duration_uses_monotonic_clock(tmp_path, monkeypatch):
    hub = DataHub()
    rec = Recorder(hub, tmp_path)
    s = rec.start("mono")
    hub.push_emg(make_emg(0))
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() - 7200.0)   # wall clock steps back 2 h
    assert 0 <= s.elapsed < 5
    s.request_stop()
    assert s.wait(5)
    monkeypatch.setattr(time, "time", real_time)
    meta = json.load(open(s.folder / "session.json"))
    assert 0 <= meta["duration_s"] < 5


def test_status_distinguishes_callback_age_from_sensor_latency(qapp, tmp_path):
    from gforce_emg_ui.sources import SimSource
    from gforce_emg_ui.ui.main_window import MainWindow
    hub = DataHub()
    src = SimSource(hub, seed=1)
    win = MainWindow(hub, src, tmp_path)
    try:
        src.connect_device(SimSource.ADDRESS)
        assert wait_until(lambda: hub.emg.total > 500, 5, qapp)
        win._refresh_plots()
        win._refresh_status()
        text = win.status_text.text()
        assert "callback→draw" in text
        assert "sensor→host latency: not measured" in text
        assert "data age" not in text
    finally:
        src.disconnect_device()
        win.close()
