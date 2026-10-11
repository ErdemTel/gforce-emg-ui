"""Recording lifecycle regression tests."""

import csv
import json
import threading
import time

import numpy as np

from conftest import make_emg, make_imu, wait_until
from gforce_emg_ui import recorder as recorder_mod
from gforce_emg_ui.hub import DataHub
from gforce_emg_ui.recorder import ERROR, FINALIZING, RECORDING, SAVED, Recorder


def _rows(path):
    return list(csv.DictReader(open(path)))


def test_closed_session_rejects_callbacks(tmp_path):
    hub = DataHub()
    rec = Recorder(hub, tmp_path)
    s = rec.start("a")
    hub.push_emg(make_emg(0))
    s.request_stop()
    s._on_emg(make_emg(16))            # stale callback still holding the session
    hub.push_emg(make_emg(32))          # hub must no longer deliver to it
    assert s.wait(5)
    assert s.state == SAVED
    assert s.counts["emg"]["received"] == 16
    assert s.counts["emg"]["rejected"] == 16
    assert [int(r["sample_index"]) for r in _rows(s.folder / "emg.csv")] == list(range(16))


def test_no_leak_between_sessions_under_concurrent_push(tmp_path):
    hub = DataHub()
    rec = Recorder(hub, tmp_path)
    stop = threading.Event()
    next_index = [0]

    def pusher():
        while not stop.is_set():
            hub.push_emg(make_emg(next_index[0], n=8))
            next_index[0] += 8
            time.sleep(0.001)

    t = threading.Thread(target=pusher)
    t.start()
    try:
        a = rec.start("a")
        time.sleep(0.2)
        a.request_stop()
        assert a.wait(5)
        b = rec.start("b")
        time.sleep(0.2)
        b.request_stop()
        assert b.wait(5)
    finally:
        stop.set()
        t.join()
    assert a.session_id != b.session_id and a.folder != b.folder
    ia = {int(r["sample_index"]) for r in _rows(a.folder / "emg.csv")}
    ib = {int(r["sample_index"]) for r in _rows(b.folder / "emg.csv")}
    assert ia and ib and not (ia & ib)
    assert max(ia) < min(ib)
    for s in (a, b):
        c = s.counts["emg"]
        assert c["received"] == c["enqueued"] + c["dropped"]
        assert c["written"] == c["enqueued"] == len(_rows(s.folder / "emg.csv"))


def test_marker_index_is_per_session(tmp_path):
    hub = DataHub()
    rec = Recorder(hub, tmp_path)
    a = rec.start("a")
    hub.push_emg(make_emg(100))
    a.add_marker("in_a")
    a.request_stop()
    assert a.wait(5)
    b = rec.start("b")
    b.add_marker("before_any_emg")       # must not inherit session a's last index
    hub.push_emg(make_emg(500))
    b.add_marker('quote "and", comma')
    b.request_stop()
    assert b.wait(5)
    ma = _rows(a.folder / "markers.csv")
    mb = _rows(b.folder / "markers.csv")
    assert ma[0]["emg_sample_index"] == "115"
    assert mb[0]["emg_sample_index"] == ""
    assert mb[1]["emg_sample_index"] == "515"
    assert mb[1]["label"] == 'quote "and", comma'


def test_same_second_sessions_get_distinct_folders(tmp_path):
    hub = DataHub()
    rec = Recorder(hub, tmp_path)
    folders = []
    for _ in range(2):
        s = rec.start("same")
        s.request_stop()
        assert s.wait(5)
        folders.append(s.folder)
    assert folders[0] != folders[1]


def test_writer_exception_is_visible_error(tmp_path, monkeypatch):
    def boom(self, f, block):
        raise OSError("disk full (injected)")
    monkeypatch.setattr(recorder_mod.RecordingSession, "_write_emg", boom)
    hub = DataHub()
    rec = Recorder(hub, tmp_path)
    s = rec.start("err")
    hub.push_emg(make_emg(0))
    assert wait_until(lambda: s.state == ERROR, 5)
    assert "disk full" in s.error
    hub.push_emg(make_emg(16))           # a failed session must stop accepting
    s.request_stop()
    assert s.wait(5)
    assert s.state == ERROR              # never reported as SAVED
    meta = json.load(open(s.folder / "session.json"))
    assert meta["state"] == ERROR and "disk full" in meta["error"]


def test_queue_overflow_is_visible_error_and_counts_reconcile(tmp_path, monkeypatch):
    gate = threading.Event()
    orig = recorder_mod.RecordingSession._write_emg

    def slow(self, f, block):
        gate.wait(5)
        return orig(self, f, block)
    monkeypatch.setattr(recorder_mod.RecordingSession, "_write_emg", slow)
    hub = DataHub()
    rec = Recorder(hub, tmp_path, queue_size=4)
    s = rec.start("overflow")
    for i in range(20):
        hub.push_emg(make_emg(i * 16))
    assert s.state == ERROR and "overflow" in s.error
    gate.set()
    s.request_stop()
    assert s.wait(5)
    c = s.counts["emg"]
    assert c["received"] == 320
    assert c["dropped"] > 0
    assert c["received"] == c["enqueued"] + c["dropped"]
    assert c["written"] == c["enqueued"] == len(_rows(s.folder / "emg.csv"))
    assert s.state == ERROR


def test_saved_only_after_writer_completion_and_stop_does_not_block(tmp_path, monkeypatch):
    gate = threading.Event()
    orig = recorder_mod.RecordingSession._write_emg

    def slow(self, f, block):
        gate.wait(5)
        return orig(self, f, block)
    monkeypatch.setattr(recorder_mod.RecordingSession, "_write_emg", slow)
    hub = DataHub()
    rec = Recorder(hub, tmp_path)
    s = rec.start("slow")
    hub.push_emg(make_emg(0))
    t0 = time.perf_counter()
    s.request_stop()
    assert time.perf_counter() - t0 < 0.05       # non-blocking
    assert s.state == FINALIZING
    assert not s.wait(0.2)
    assert s.state == FINALIZING                 # not SAVED while the writer is busy
    gate.set()
    assert s.wait(5)
    assert s.state == SAVED
    meta = json.load(open(s.folder / "session.json"))
    assert meta["state"] == SAVED
    assert meta["counts"]["emg"]["written"] == 16


def test_counts_reconcile_for_all_streams(tmp_path):
    hub = DataHub()
    rec = Recorder(hub, tmp_path)
    s = rec.start("counts")
    for i in range(10):
        hub.push_emg(make_emg(i * 16))
    hub.push_imu(make_imu(0, n=3))
    s.add_marker("m")
    assert s.state == RECORDING
    s.request_stop()
    assert s.wait(5)
    meta = json.load(open(s.folder / "session.json"))
    assert meta["session_id"] == s.session_id
    for name, n in (("emg", 160), ("imu", 3), ("markers", 1)):
        c = meta["counts"][name]
        assert c["received"] == c["enqueued"] == c["written"] == n, name
        assert c["dropped"] == 0
    assert len(_rows(s.folder / "imu.csv")) == 3


def test_main_window_stop_is_nonblocking_and_reports_saved(qapp, tmp_path, monkeypatch):
    from gforce_emg_ui.sources import SimSource
    from gforce_emg_ui.ui.main_window import MainWindow
    gate = threading.Event()
    orig = recorder_mod.RecordingSession._write_emg

    def slow(self, f, block):
        gate.wait(5)
        return orig(self, f, block)
    monkeypatch.setattr(recorder_mod.RecordingSession, "_write_emg", slow)
    hub = DataHub()
    src = SimSource(hub, seed=1)
    win = MainWindow(hub, src, tmp_path)
    try:
        src.connect_device(SimSource.ADDRESS)
        assert wait_until(lambda: hub.emg.total > 0, 5, qapp)
        win._toggle_recording()
        assert wait_until(lambda: win.recorder.session.counts["emg"]["received"] > 0, 5, qapp)
        t0 = time.perf_counter()
        win._toggle_recording()
        assert time.perf_counter() - t0 < 0.1
        win._refresh_status()
        assert "Finalizing" in win.record_label.text()
        assert "Saved" not in win.record_label.text()
        gate.set()
        assert wait_until(lambda: (win._refresh_status() or True)
                          and "Saved" in win.record_label.text(), 10, qapp)
    finally:
        gate.set()
        src.disconnect_device()
        win.close()
