"""Live Classifier tab in the offscreen MainWindow. The simulator makes these SOFTWARE
tests only: synthetic EMG, predictions are meaningless and no accuracy is claimed."""

import csv
import json
import sys
import threading
import time

import pytest

from conftest import wait_until
from gforce_emg_ui.hub import DataHub
from gforce_emg_ui.live.engine import LiveEngine, RUNNING, STOPPED
from gforce_emg_ui.recorder import SAVED
from gforce_emg_ui.sources import SimSource
from test_live_model import MODEL, needs_model

FILTERS = ("FILTER_50HZ", "FILTER_60HZ", "FILTER_HPF", "FILTER_LPF")
LIVE_THREADS = ("live-worker", "live-watchdog", "live-log", "live-model-load")


def live_threads():
    return [t.name for t in threading.enumerate() if t.name in LIVE_THREADS]


def make(qapp, tmp_path, show_live=True):
    from gforce_emg_ui.ui.main_window import MainWindow
    hub = DataHub()
    src = SimSource(hub, seed=3, reconnect_delay=0.3)
    win = MainWindow(hub, src, tmp_path / "data")
    win.live.set_log_root(tmp_path / "live_logs")
    win.show()
    if show_live:
        win.tabs.setCurrentWidget(win.live)
    src.connect_device(src.ADDRESS)
    assert wait_until(lambda: hub.emg.total > 0, 5, qapp)
    return win, src, hub


def load(qapp, live):
    live.load_model_file(MODEL)
    assert wait_until(lambda: not live.loading, 30, qapp)
    assert live.model is not None, live.load_error


def make_compatible(qapp, live, src):
    for f in FILTERS:                       # the operator's choice; never done by the tab
        src.set_filter(f, True)
    assert wait_until(lambda: live.report is not None and not live.report.blocking, 5, qapp)
    live.ack_check.setChecked(True)
    assert wait_until(lambda: live.btn_start.isEnabled(), 5, qapp), live.start_blockers()


def running(qapp, live, timeout=10):
    return wait_until(lambda: live.engine.snapshot().state == RUNNING
                      and live.s_label.text() in ("s = 0", "s = 0.5", "s = 1"), timeout, qapp)


def shut(qapp, win, src):
    src.disconnect_device()
    win.close()
    assert wait_until(lambda: not live_threads(), 10, qapp), live_threads()


def test_tab_is_added_and_app_records_without_the_classifier(qapp, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "emg_classifier", None)
    win, src, hub = make(qapp, tmp_path)
    try:
        names = [win.tabs.tabText(i) for i in range(win.tabs.count())]
        assert names == ["EMG – Equal scale", "EMG – Detail", "IMU", "Live Classifier"]
        live = win.live
        assert "not installed" in live.model_label.text()
        assert not live.btn_start.isEnabled() and live.engine.model is None
        live.load_model_file(tmp_path / "whatever.joblib")
        assert wait_until(lambda: not live.loading, 10, qapp)
        assert "cannot be imported" in live.model_label.text()
        win._toggle_recording()                           # recording is unaffected
        assert wait_until(lambda: win.recorder.session.counts["emg"]["received"] > 0, 5, qapp)
        win._toggle_recording()
        s = win.recorder.session
        assert s.wait(10) and s.state == SAVED
        assert s.counts["emg"]["written"] == s.counts["emg"]["received"]
    finally:
        shut(qapp, win, src)


@needs_model
def test_full_cycle_compatibility_start_hidden_tab_config_change_and_logs(qapp, tmp_path):
    win, src, hub = make(qapp, tmp_path)
    live = win.live
    try:
        assert live.engine.snapshot().state == "NO_MODEL" and live.s_label.text() == "s = --"
        load(qapp, live)
        assert not live.engine.running                    # never auto-starts
        assert live.report.blocking                       # simulator filters off by default
        assert any("incompatible" in b for b in live.start_blockers())
        assert not live.btn_start.isEnabled() and not live.ack_check.isEnabled()
        for f in FILTERS:
            src.set_filter(f, True)
        assert wait_until(lambda: not live.report.blocking, 5, qapp)
        assert {c.field for c in live.report.needs_ack} >= {"source", "device_model"}
        assert not live.btn_start.isEnabled()             # warnings need acknowledgement
        live.ack_check.setChecked(True)
        assert wait_until(lambda: live.btn_start.isEnabled(), 5, qapp)
        live.log_check.setChecked(True)
        live.btn_start.click()
        assert live.engine.running and running(qapp, live)
        assert live.bars.values() is not None and live.class_label.text() in ("LOW", "MEDIUM",
                                                                              "HIGH")
        assert "soft class score" in live.soft_label.text()
        assert wait_until(lambda: len(live._t_gui) > 0, 5, qapp)

        win.tabs.setCurrentIndex(0)                       # hidden: inference continues
        assert not live.refresh()
        n0 = live.engine.snapshot().counters["predictions"]
        assert wait_until(lambda: live.engine.snapshot().counters["predictions"] > n0 + 5, 5,
                          qapp)
        win.tabs.setCurrentWidget(live)
        live.btn_reset.click()
        assert running(qapp, live)
        assert live.engine.snapshot().counters["resets"].get("user_reset") == 1

        src.set_filter("FILTER_HPF", False)               # material settings change
        assert wait_until(lambda: not live.engine.running, 5, qapp)
        snap = live.engine.snapshot()
        assert snap.state == STOPPED and "configuration changed" in snap.reason
        assert wait_until(lambda: live.s_label.text() == "s = --", 5, qapp)
        assert not live.ack_check.isChecked() and not live.btn_start.isEnabled()
        log = live.log
        assert log.wait(5) and not log.error
        meta = json.load(open(log.folder / "run.json"))
        assert meta["source"]["type"] == "simulation" and "sim_" in log.folder.name
        assert meta["model"]["sha256"] == live.model.sha256 and meta["log"]["complete"]
        assert meta["labels"].startswith("none")
        rows = list(csv.DictReader(open(log.folder / "predictions.csv")))
        assert rows and all(r["window_end_index"] for r in rows)
        events = [r["event"] for r in csv.DictReader(open(log.folder / "events.csv"))]
        assert events[0] == "start" and "reset" in events and events[-1] == "stop"
        assert len(list(csv.DictReader(open(log.folder / "display.csv")))) > 0
        assert not (tmp_path / "data").exists() or not any((tmp_path / "data").rglob("*.csv"))

        make_compatible(qapp, live, src)
        live.log_check.setChecked(False)
        live.btn_start.click()
        assert running(qapp, live)
        src.disconnect_device()                           # disconnect stops classification
        assert wait_until(lambda: not live.engine.running, 5, qapp)
        assert "disconnected" in live.engine.snapshot().reason
        assert wait_until(lambda: live.s_label.text() == "s = --", 5, qapp)
    finally:
        shut(qapp, win, src)
    assert LiveEngine.SUB_NAME not in [n for n, _ in hub._emg_subscribers]


@needs_model
def test_link_loss_resets_then_recovers_on_the_new_stream(qapp, tmp_path):
    win, src, hub = make(qapp, tmp_path)
    live = win.live
    try:
        load(qapp, live)
        make_compatible(qapp, live, src)
        live.btn_start.click()
        assert running(qapp, live)
        old_stream = live.engine.snapshot().result.stream_id
        src.inject_disconnect()
        assert wait_until(lambda: live.s_label.text() == "s = --", 5, qapp)
        assert running(qapp, live, timeout=10)
        snap = live.engine.snapshot()
        assert snap.result.stream_id != old_stream
        assert snap.counters["resets"].get("link_lost", 0) >= 1
    finally:
        shut(qapp, win, src)


@needs_model
def test_lifecycle_repeats_without_leaks_or_blocking(qapp, tmp_path):
    win, src, hub = make(qapp, tmp_path)
    live = win.live
    try:
        load(qapp, live)
        make_compatible(qapp, live, src)
        worst = 0.0
        for k in range(5):
            t0 = time.perf_counter()
            live.btn_start.click()
            live.btn_reset.click()
            win.tabs.setCurrentIndex(k % 3)
            win.tabs.setCurrentWidget(live)
            live.btn_stop.click()
            worst = max(worst, time.perf_counter() - t0)
            live.ack_check.setChecked(True)
            assert wait_until(lambda: live.btn_start.isEnabled(), 5, qapp)
            names = [n for n, _ in hub._emg_subscribers]
            assert names.count(LiveEngine.SUB_NAME) == 0
        assert worst < 0.2
        live.btn_start.click()
        assert running(qapp, live)
        live.load_model_file(MODEL)                       # replacement stops the run first
        assert not live.engine.running and live.engine.model is None
        assert wait_until(lambda: live.s_label.text() == "s = --", 5, qapp)
        assert wait_until(lambda: not live.loading and live.model is not None, 30, qapp)
        assert not live.engine.running
        assert [n for n, _ in hub._emg_subscribers].count(LiveEngine.SUB_NAME) == 0
        assert len([t for t in live_threads() if t == "live-worker"]) <= 1
        live.ack_check.setChecked(True)
        live.btn_start.click()
        assert running(qapp, live)
        live.load_model_file(MODEL)                       # close while a load is in flight
    finally:
        shut(qapp, win, src)


@needs_model
def test_no_relayout_and_widgets_change_only_on_the_gui_thread(qapp, tmp_path, monkeypatch):
    from PyQt5 import QtCore
    from gforce_emg_ui.ui import live_view
    off_thread = []
    for name in ("_show_prediction", "_draw_history", "_show_diagnostics", "_show_model",
                 "_show_compat", "_update_controls"):
        orig = getattr(live_view.LiveClassifierDashboard, name)

        def guarded(self, *a, _orig=orig, _name=name, **k):
            if threading.current_thread() is not threading.main_thread():
                off_thread.append(_name)
            return _orig(self, *a, **k)
        monkeypatch.setattr(live_view.LiveClassifierDashboard, name, guarded)
    win, src, hub = make(qapp, tmp_path)
    live = win.live
    layouts = []

    class Spy(QtCore.QObject):
        def eventFilter(self, obj, ev):
            if obj is live and ev.type() == QtCore.QEvent.LayoutRequest:
                layouts.append(1)
            return False
    spy = Spy()
    try:
        load(qapp, live)
        make_compatible(qapp, live, src)
        live.btn_start.click()
        assert running(qapp, live)
        qapp.installEventFilter(spy)
        texts = set()
        layouts.clear()
        wait_until(lambda: texts.add(live.diag_label.text()) or False, 1.0, qapp)
        assert len(texts) >= 2                            # diagnostics keep updating ...
        assert len(layouts) <= 3, len(layouts)            # ... without re-layouts
    finally:
        qapp.removeEventFilter(spy)
        shut(qapp, win, src)
    assert off_thread == []


@needs_model
def test_recorder_reconciles_and_controls_respond_while_classifying(qapp, tmp_path):
    from PyQt5 import QtCore
    win, src, hub = make(qapp, tmp_path)
    live = win.live
    try:
        load(qapp, live)
        make_compatible(qapp, live, src)
        live.btn_start.click()
        assert running(qapp, live)
        win._toggle_recording()
        assert win.recorder.active
        lag = []
        for _ in range(10):                               # GUI event round trips
            t0 = time.perf_counter()
            done = []
            QtCore.QTimer.singleShot(0, lambda: done.append(1))
            assert wait_until(lambda: done, 1, qapp)
            lag.append(time.perf_counter() - t0)
            wait_until(lambda: False, 0.1, qapp)
        win._toggle_recording()
        s = win.recorder.session
        assert s.wait(10) and s.state == SAVED
        e = s.counts["emg"]
        assert e["written"] == e["received"] == e["enqueued"] and e["dropped"] == 0
        assert live.engine.snapshot().counters["valid_results"] > 10
        assert max(lag) < 0.2, lag
        assert hub.subscriber_errors == {}
    finally:
        shut(qapp, win, src)
