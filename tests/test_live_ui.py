"""Live Classifier tab in the offscreen MainWindow. The simulator makes these SOFTWARE
tests only: synthetic EMG, predictions are meaningless and no accuracy is claimed. Robot
output tests send to a loopback socket of the test, never to a robot."""

import csv
import json
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from conftest import wait_until
from gforce_emg_ui.hub import DataHub
from gforce_emg_ui.live import robot_out
from gforce_emg_ui.live.engine import HUB_CHANNELS, LiveEngine, RUNNING, STOPPED
from gforce_emg_ui.recorder import SAVED
from gforce_emg_ui.sources import SimSource
from test_live_model import MODEL, needs_model

FILTERS = ("FILTER_50HZ", "FILTER_60HZ", "FILTER_HPF", "FILTER_LPF")
LIVE_THREADS = ("live-worker", "live-watchdog", "live-log", "live-model-load", "live-robot-out")


def live_threads():
    return [t.name for t in threading.enumerate() if t.name in LIVE_THREADS]


def make(qapp, tmp_path, show_live=True, source_cls=SimSource, **win_kw):
    from gforce_emg_ui.ui.main_window import MainWindow
    hub = DataHub()
    src = source_cls(hub, seed=3, reconnect_delay=0.3)
    win = MainWindow(hub, src, tmp_path / "data", **win_kw)     # positional, as before
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
                 "_show_compat", "_update_controls", "_update_robot_controls", "_show_robot"):
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


# -- robot output (loopback receiver of the test, no robot) ---------------------------

class BandLikeSim(SimSource):
    """Test double: the simulator presenting itself as a real band, so the arming rules see
    a clean compatibility report. Its EMG stays synthetic."""
    is_simulation = False


def fake_model(tmp_path, live):
    """A LoadedModel whose saved acquisition equals the live one (all checks ok) and a
    test-double classifier: no emg_classifier, no model file."""
    from gforce_emg_ui.live import compat
    from gforce_emg_ui.live.model import LoadedModel
    from test_live_engine import FakeClf
    acq = compat.live_acquisition(live._device, live._params, live.source, live.hub)
    return LoadedModel(
        clf=FakeClf(), path=tmp_path / "runs" / "fake" / "model.joblib", sha256="0" * 64,
        sidecar_sha256=None, name="short_window_mlp", window_samples=100, hop_samples=25,
        window_units="samples", channel_names=HUB_CHANNELS,
        classes=((0, "low", 0.0), (1, "medium", 0.5), (2, "high", 1.0)),
        feature={"name": "log_ac_rms", "version": 1},
        acquisition={"keys": ["live"], "configurations": {"live": {"summary": acq}}},
        training={"n_trials": 77, "n_windows": 2310}, cv_reference={}, hold_rule={},
        created="", code_sha256=None, installed_code_sha256="", trained_versions={},
        installed_versions={}, limitations=(), warnings=())


def make_robot(qapp, tmp_path, source_cls=BandLikeSim):
    """MainWindow targeting a loopback receiver (as --robot-udp would), test model installed
    like a finished background load; nothing started or armed."""
    from test_live_robot_out import Receiver                     # (import cycle at load time)
    rx = Receiver()
    win, src, hub = make(qapp, tmp_path, source_cls=source_cls, robot_target=(rx.host, rx.port))
    live = win.live
    live._load_token += 1
    live._on_model_loaded(fake_model(tmp_path, live), live._load_token)
    assert live.model is not None and live.report is not None
    return win, src, hub, rx


def start_and_arm(qapp, live, rx):
    if live.report.needs_ack:
        live.ack_check.setChecked(True)
    live.btn_start.click()
    assert live.engine.running
    assert wait_until(lambda: live.robot_check.isEnabled(), 5, qapp), live.robot_blockers()
    live.robot_check.click()
    assert live.robot.armed, live.robot_blockers()
    assert rx.collect(lambda m: len(m) >= 3)


@pytest.fixture
def dialogs(monkeypatch):
    """Answers QMessageBox.question (Yes unless queued otherwise) and records the texts."""
    from PyQt5 import QtWidgets
    rec = SimpleNamespace(texts=[], answers=[])

    def question(parent, title, text, *a, **k):
        rec.texts.append(text)
        return rec.answers.pop(0) if rec.answers else QtWidgets.QMessageBox.Yes
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(question))
    return rec


def golden_state(**kw) -> dict:
    """The protocol's example emg-stiffness-state/1 datagram (shared golden file)."""
    from test_live_robot_out import GOLDEN
    return {**GOLDEN["state_example"], **kw}


def test_robot_output_arms_only_when_running_clean_and_confirmed(qapp, tmp_path, dialogs):
    from PyQt5 import QtWidgets
    win, src, hub, rx = make_robot(qapp, tmp_path)
    live = win.live
    target = f"{rx.host}:{rx.port}"
    try:
        assert (live.robot_host.text(), live.robot_port.value()) == (rx.host, rx.port)
        assert not live.robot.armed and not live.robot_check.isChecked()   # CLI never arms
        assert not live.robot_check.isEnabled() and win.robot_indicator.isHidden()
        assert "start the classifier" in live.robot_hint.text()
        assert not live.report.needs_ack and not live.report.blocking
        assert not live.robot_ack.isEnabled()                 # nothing to acknowledge
        assert "robot will ignore: stabilization OFF" in live.robot_hint.text()
        assert "robot state unknown" in live.robot_status.text()
        live.btn_start.click()
        assert live.engine.running and live.robot_check.isEnabled(), live.robot_blockers()
        dialogs.answers.append(QtWidgets.QMessageBox.No)
        live.robot_check.click()                              # declined: nothing armed
        assert len(dialogs.texts) == 1 and target in dialogs.texts[0]
        assert not live.robot.armed and not live.robot_check.isChecked()
        assert not rx.collect(lambda m: m, timeout=0.3) and live.store.value(
            robot_out.STORE_KEY) is None
        live.robot_check.click()                              # confirmed
        assert live.robot.armed and live.robot_check.isChecked() and len(dialogs.texts) == 2
        assert win.robot_indicator.text() == f"ROBOT OUTPUT ARMED → {target}"
        assert not win.robot_indicator.isHidden()
        win.tabs.setCurrentIndex(0)                           # any tab: still shown, sending
        assert not win.robot_indicator.isHidden()
        assert rx.collect(lambda m: len(m) >= 5)
        assert all(not x["simulated"] and not x["stabilized"] and x["run_id"] for x in rx.msgs)
        assert not live.robot_host.isEnabled() and not live.robot_port.isEnabled()
        win.tabs.setCurrentWidget(live)
        assert wait_until(lambda: "SENDING" in live.robot_status.text(), 2, qapp)
        rx.reply(golden_state(session=live.robot.session))   # the bridge's back-channel
        assert wait_until(lambda: "robot: applied s 0.5 · phase idle · armed · dry run"
                          in live.robot_status.text(), 3, qapp), live.robot_status.text()
        rx.reply(golden_state(session=live.robot.session, seq=13, detail="<b>x</b> & y"))
        assert wait_until(lambda: "&lt;b&gt;x&lt;/b&gt; &amp; y" in live.robot_status.text(),
                          3, qapp), live.robot_status.text()          # network text: escaped
        live.robot_check.click()                              # operator disarm
        assert not live.robot.armed and win.robot_indicator.isHidden()
        assert rx.collect(lambda m: m[-1]["kind"] == "stop")
        assert rx.msgs[-1]["reason"] == "operator" and live.robot_host.isEnabled()
        assert json.loads(live.store.value(robot_out.STORE_KEY)) == {
            "host": rx.host, "port": rx.port, "rate_hz": 20}  # target only, never armed
    finally:
        shut(qapp, win, src)
        rx.close()


@pytest.mark.parametrize("event", ["model_load", "apply_off", "apply_on", "stop", "disconnect",
                                   "link_lost", "engine_stopped_itself", "app_close"])
def test_robot_output_disarms_itself(qapp, tmp_path, dialogs, event):
    win, src, hub, rx = make_robot(qapp, tmp_path)
    live = win.live
    closed = False
    try:
        if event == "apply_off":                              # stabilization ON, then off
            live.stab_enable.setChecked(True)
            assert live.apply_stabilization()
        start_and_arm(qapp, live, rx)
        n = len(rx.msgs)
        if event == "model_load":
            live.load_model_file(tmp_path / "other.joblib")
        elif event == "apply_off":
            live.stab_enable.setChecked(False)
            assert live.apply_stabilization()
        elif event == "apply_on":
            live.stab_enable.setChecked(True)
            assert live.apply_stabilization() and live.engine.running
        elif event == "stop":
            live.btn_stop.click()
        elif event == "disconnect":
            src.disconnect_device()
        elif event == "link_lost":
            src.inject_disconnect()                           # reconnects by itself
        elif event == "engine_stopped_itself":
            live.engine.stop("worker error")                  # not through the tab
        elif event == "app_close":
            win.close()
            closed = True
        assert wait_until(lambda: not live.robot.armed, 3, qapp)
        assert rx.collect(lambda m: m[-1]["kind"] == "stop")
        stop = rx.msgs[-1]
        assert stop["s"] is None and not stop["valid"] and stop["reason"]
        assert [x["kind"] for x in rx.msgs].count("stop") == 1
        if event.startswith("apply"):                         # nothing sent with the new meaning
            assert all(x["stabilized"] == (event == "apply_off") for x in rx.msgs[n:-1])
        assert not live.robot_check.isChecked() and win.robot_indicator.isHidden()
        assert not rx.collect(lambda m: m[-1]["kind"] != "stop", timeout=0.2)
    finally:
        if closed:
            src.disconnect_device()
            assert wait_until(lambda: not live_threads(), 10, qapp), live_threads()
        else:
            shut(qapp, win, src)
        rx.close()
    reasons = {"model_load": "model load", "apply_off": "stabilization settings applied",
               "apply_on": "stabilization settings applied", "stop": "classifier stopped",
               "disconnect": "band", "link_lost": "band reconnecting",
               "engine_stopped_itself": "classifier stopped: worker error",
               "app_close": "app closing"}
    assert stop["reason"].startswith(reasons[event]), stop["reason"]


def test_robot_output_with_the_simulator_needs_its_own_acknowledgement(qapp, tmp_path,
                                                                        dialogs):
    win, src, hub, rx = make_robot(qapp, tmp_path, source_cls=SimSource)
    live = win.live
    try:
        assert [c.field for c in live.report.needs_ack] == ["source"]    # simulator WARN
        assert "robot will ignore: simulated" in live.robot_hint.text()
        live.ack_check.setChecked(True)                       # Start's acknowledgement ...
        live.btn_start.click()
        assert live.engine.running and not live.robot_check.isEnabled()  # ... is not enough
        assert live.robot_ack.isEnabled() and "Send despite" in live.robot_hint.text()
        assert not live.arm_robot() and not dialogs.texts
        live.robot_ack.setChecked(True)
        live.robot_check.click()
        assert live.robot.armed and not live.robot_ack.isEnabled()
        assert "simulated source" in dialogs.texts[-1]
        assert rx.collect(lambda m: len(m) >= 3) and all(x["simulated"] for x in rx.msgs)
        src.set_filter("FILTER_HPF", True)                    # a compatibility change ...
        assert wait_until(lambda: not live.robot.armed, 3, qapp)
        assert not live.robot_ack.isChecked() and not live.ack_check.isChecked()   # ... clears
        assert rx.collect(lambda m: m[-1]["kind"] == "stop")
        assert rx.msgs[-1]["reason"].startswith("classifier stopped: configuration changed")
        live.robot_ack.setChecked(True)                       # not running: tick, then recheck
        src.set_filter("FILTER_HPF", False)
        assert wait_until(lambda: not live.robot_ack.isChecked(), 3, qapp)
    finally:
        shut(qapp, win, src)
        rx.close()


def test_robot_target_persists_cli_prefills_and_bad_values_block(qapp, tmp_path):
    from PyQt5 import QtCore
    from gforce_emg_ui.ui.live_view import LiveClassifierDashboard
    from test_live_stabilization import make_dash
    store = QtCore.QSettings(str(tmp_path / "robot.ini"), QtCore.QSettings.IniFormat)
    dash, _ = make_dash(tmp_path, store)
    try:
        assert (dash.robot_host.text(), dash.robot_port.value()) == ("127.0.0.1", 6731)
        dash.robot_host.setText(" 192.168.1.20 ")
        dash.robot_port.setValue(7000)
        dash.robot_port.editingFinished.emit()
        assert json.loads(store.value(robot_out.STORE_KEY)) == {
            "host": "192.168.1.20", "port": 7000, "rate_hz": 20}
        dash.robot_host.setText("robot-pc.local")             # never resolved: refused
        dash.robot_host.editingFinished.emit()
        assert "not an IP address" in dash.robot_hint.text()
        assert "a valid robot IP and port" in dash.robot_hint.text()
        assert json.loads(store.value(robot_out.STORE_KEY))["host"] == "192.168.1.20"
    finally:
        dash.shutdown()
    dash, _ = make_dash(tmp_path, store)                      # restored, never armed
    try:
        assert (dash.robot_host.text(), dash.robot_port.value()) == ("192.168.1.20", 7000)
        assert not dash.robot.armed and not dash.robot_check.isChecked()
    finally:
        dash.shutdown()
    hub = DataHub()
    dash = LiveClassifierDashboard(hub, SimSource(hub), store=store,
                                   robot_target=("10.0.0.5", 6800))   # --robot-udp
    try:
        assert (dash.robot_host.text(), dash.robot_port.value()) == ("10.0.0.5", 6800)
        assert not dash.robot.armed and dash.robot.settings.target == "10.0.0.5:6800"
        assert json.loads(store.value(robot_out.STORE_KEY))["host"] == "192.168.1.20"
    finally:
        dash.shutdown()
    store.setValue(robot_out.STORE_KEY, "{not json")
    dash, _ = make_dash(tmp_path, store)
    try:
        assert dash.robot_host.text() == "127.0.0.1" and "ignored" in dash.robot_hint.text()
    finally:
        dash.shutdown()


def test_robot_widgets_change_only_on_the_gui_thread_without_relayouts(qapp, tmp_path, dialogs,
                                                                        monkeypatch):
    from PyQt5 import QtCore
    from gforce_emg_ui.ui import live_view
    off_thread = []
    for name in ("_update_controls", "_update_robot_controls", "_show_robot", "_robot_changed"):
        orig = getattr(live_view.LiveClassifierDashboard, name)

        def guarded(self, *a, _orig=orig, _name=name, **k):
            if threading.current_thread() is not threading.main_thread():
                off_thread.append(_name)
            return _orig(self, *a, **k)
        monkeypatch.setattr(live_view.LiveClassifierDashboard, name, guarded)
    win, src, hub, rx = make_robot(qapp, tmp_path)
    live = win.live
    layouts = []

    class Spy(QtCore.QObject):
        def eventFilter(self, obj, ev):
            if obj is live and ev.type() == QtCore.QEvent.LayoutRequest:
                layouts.append(1)
            return False
    spy = Spy()
    try:
        start_and_arm(qapp, live, rx)
        qapp.installEventFilter(spy)
        texts = set()
        layouts.clear()
        wait_until(lambda: texts.add(live.robot_status.text()) or False, 1.0, qapp)
        assert len(texts) >= 2                                # sent count keeps updating ...
        assert len(layouts) <= 3, len(layouts)                # ... without re-layouts
        t0 = time.perf_counter()
        live.robot_check.click()                              # disarm never waits
        assert time.perf_counter() - t0 < 0.1 and not live.robot.armed
    finally:
        qapp.removeEventFilter(spy)
        shut(qapp, win, src)
        rx.close()
    assert off_thread == []


def test_cli_robot_target_is_validated_and_only_prefills(capsys):
    from gforce_emg_ui.__main__ import parse_args
    assert parse_args([]).robot_target is None
    assert parse_args(["--robot-udp", "127.0.0.1:6731"]).robot_target == ("127.0.0.1", 6731)
    assert parse_args(["--robot-udp", "[::1]:7000"]).robot_target == ("::1", 7000)
    for bad in ("localhost:6731", "127.0.0.1", "224.0.0.1:6731", "127.0.0.1:70000",
                "0.0.0.0:6731"):
        with pytest.raises(SystemExit) as e:
            parse_args(["--robot-udp", bad])
        assert e.value.code == 2 and "--robot-udp" in capsys.readouterr().err, bad
