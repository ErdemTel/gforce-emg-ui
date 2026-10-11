"""Main window: control panel (device, battery, contact, settings, recording)
plus dashboards in tabs. Only the visible dashboard is redrawn."""

from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .. import __version__
from ..config import (EMG_CHANNELS, EMG_RATES, FILTERS, PLOT_INTERVAL_MS,
                      STATUS_INTERVAL_MS)
from ..contact import LEVEL_COLORS, contact_color, format_kohm
from ..hub import DataHub
from ..recorder import ERROR, FINALIZING, RECORDING, SAVED, Recorder
from ..sources import (CONNECTING, DISCONNECTING, FAILED, IDLE, RECONNECTING, SCANNING,
                       STREAMING, Source)
from .emg_detail import DetailDashboard
from .emg_equal import EqualScaleDashboard
from .imu_view import ImuDashboard


PANEL_WIDTH = 380


class ContactStrip(QtWidgets.QWidget):
    """8 chips, green / orange / red by electrode impedance (vendor demo thresholds)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QtWidgets.QGridLayout(self)
        grid.setSpacing(4)
        self.chips = []
        for ch in range(EMG_CHANNELS):
            chip = QtWidgets.QLabel()
            chip.setAlignment(QtCore.Qt.AlignCenter)
            chip.setMinimumSize(70, 38)
            grid.addWidget(chip, ch // 4, ch % 4)
            self.chips.append(chip)
        legend = QtWidgets.QLabel(
            f"<span style='color:{LEVEL_COLORS['good']}'>■</span> ≤500 kΩ  "
            f"<span style='color:{LEVEL_COLORS['marginal']}'>■</span> ≤999 kΩ  "
            f"<span style='color:{LEVEL_COLORS['poor']}'>■</span> loose")
        grid.addWidget(legend, 2, 0, 1, 4)
        self.update_contact(np.full(EMG_CHANNELS, np.nan))

    def update_contact(self, impedance):
        for ch, chip in enumerate(self.chips):
            chip.setText(f"CH{ch + 1}\n{format_kohm(impedance[ch])}")
            chip.setStyleSheet(f"background:{contact_color(impedance[ch])}; color:white;"
                               "border-radius:4px; font-weight:bold;")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, hub: DataHub, source: Source, data_dir: Path):
        super().__init__()
        self.hub = hub
        self.source = source
        self.recorder = Recorder(hub, data_dir)
        self.device_summary = {}
        self.current_params = {}
        self._devices = {}
        self._age_ms = 0.0
        self._close_pending = False
        self._finalizing_seen = False
        self.setWindowTitle(f"gForce EMG UI {__version__}")
        self.resize(1500, 950)

        self.equal = EqualScaleDashboard(hub)
        self.detail = DetailDashboard(hub)
        self.imu = ImuDashboard(hub)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.equal, "EMG – Equal scale")
        self.tabs.addTab(self.detail, "EMG – Detail")
        self.tabs.addTab(self.imu, "IMU")

        panel = self._build_panel()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(PANEL_WIDTH)
        split = QtWidgets.QSplitter()
        split.addWidget(scroll)
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        split.setSizes([PANEL_WIDTH, 1200])
        self.setCentralWidget(split)

        self.status_text = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.status_text, 1)

        source.state_changed.connect(self._on_state_changed)
        source.devices_found.connect(self._on_devices)
        source.status.connect(self._set_status)
        source.connected.connect(self._on_connected)
        source.disconnected.connect(self._on_disconnected)
        source.battery.connect(self._on_battery)
        source.params.connect(self._on_params)
        source.error.connect(self._on_error)

        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(self._refresh_plots)
        self.plot_timer.start(PLOT_INTERVAL_MS)
        self.status_timer = QtCore.QTimer(self)
        self.status_timer.timeout.connect(self._refresh_status)
        self.status_timer.start(STATUS_INTERVAL_MS)

        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+M"), self, activated=self._add_marker)
        self._on_state_changed(source.state)

    # -- panel ------------------------------------------------------------------
    def _build_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setMaximumWidth(PANEL_WIDTH - 20)
        v = QtWidgets.QVBoxLayout(panel)

        dev = QtWidgets.QGroupBox("Device")
        dl = QtWidgets.QVBoxLayout(dev)
        row = QtWidgets.QHBoxLayout()
        self.btn_scan = QtWidgets.QPushButton("Scan")
        self.btn_stop_scan = QtWidgets.QPushButton("Stop scan")
        row.addWidget(self.btn_scan)
        row.addWidget(self.btn_stop_scan)
        dl.addLayout(row)
        self.device_list = QtWidgets.QListWidget()
        self.device_list.setMaximumHeight(110)
        dl.addWidget(self.device_list)
        row = QtWidgets.QHBoxLayout()
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect")
        row.addWidget(self.btn_connect)
        row.addWidget(self.btn_disconnect)
        dl.addLayout(row)
        self.state_label = QtWidgets.QLabel()
        dl.addWidget(self.state_label)
        self.device_label = QtWidgets.QLabel("Not connected")
        self.device_label.setWordWrap(True)
        dl.addWidget(self.device_label)
        v.addWidget(dev)

        bat = QtWidgets.QGroupBox("Battery")
        bl = QtWidgets.QVBoxLayout(bat)
        self.battery_bar = QtWidgets.QProgressBar()
        self.battery_bar.setRange(0, 100)
        self.battery_bar.setValue(0)
        self.battery_bar.setFormat("--")
        bl.addWidget(self.battery_bar)
        v.addWidget(bat)

        con = QtWidgets.QGroupBox("Electrode contact")
        cl = QtWidgets.QVBoxLayout(con)
        self.contact_strip = ContactStrip()
        cl.addWidget(self.contact_strip)
        v.addWidget(con)

        st = QtWidgets.QGroupBox("Settings")
        sl = QtWidgets.QVBoxLayout(st)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("EMG rate"))
        self.rate_buttons = {}
        self.rate_group = QtWidgets.QButtonGroup(self)
        for r in EMG_RATES:
            rb = QtWidgets.QRadioButton(f"{r} Hz")
            self.rate_group.addButton(rb, r)
            self.rate_buttons[r] = rb
            row.addWidget(rb)
        self.rate_buttons[self.source.target_emg_rate].setChecked(True)
        sl.addLayout(row)
        sl.addWidget(QtWidgets.QLabel("On-board filters"))
        self.filter_checks = {}
        for key, label in FILTERS.items():
            cb = QtWidgets.QCheckBox(label)
            cb.toggled.connect(lambda on, k=key: self._on_filter_toggled(k, on))
            self.filter_checks[key] = cb
            sl.addWidget(cb)
        v.addWidget(st)

        rec = QtWidgets.QGroupBox("Recording")
        rl = QtWidgets.QFormLayout(rec)
        self.session_name = QtWidgets.QLineEdit()
        self.session_name.setPlaceholderText("e.g. subj01_fist_trial1")
        rl.addRow("Name", self.session_name)
        self.folder_label = QtWidgets.QLabel(str(self.recorder.root))
        self.folder_label.setWordWrap(True)
        btn_folder = QtWidgets.QPushButton("Change folder…")
        btn_folder.clicked.connect(self._choose_folder)
        rl.addRow(self.folder_label)
        rl.addRow(btn_folder)
        self.btn_record = QtWidgets.QPushButton("● Start recording")
        self.btn_record.setStyleSheet("font-weight:bold;")
        rl.addRow(self.btn_record)
        self.record_label = QtWidgets.QLabel("Idle")
        rl.addRow(self.record_label)
        self.marker_edit = QtWidgets.QLineEdit()
        self.marker_edit.setPlaceholderText("label, e.g. fist_on")
        self.btn_marker = QtWidgets.QPushButton("Add marker (Ctrl+M)")
        rl.addRow("Marker", self.marker_edit)
        rl.addRow(self.btn_marker)
        v.addWidget(rec)
        v.addStretch(1)

        self.btn_scan.clicked.connect(self._scan)
        self.btn_stop_scan.clicked.connect(self.source.stop_scan)
        self.btn_connect.clicked.connect(self._connect)
        self.device_list.itemDoubleClicked.connect(lambda _: self._connect())
        self.btn_disconnect.clicked.connect(self._disconnect)
        self.rate_group.idToggled.connect(self._on_rate_toggled)
        self.btn_record.clicked.connect(self._toggle_recording)
        self.btn_marker.clicked.connect(self._add_marker)
        self.marker_edit.returnPressed.connect(self._add_marker)
        return panel

    STATE_STYLE = {
        IDLE: ("Idle", "#757575"), SCANNING: ("Scanning…", "#1565c0"),
        CONNECTING: ("Connecting…", "#1565c0"), STREAMING: ("Streaming", "#2e7d32"),
        RECONNECTING: ("Link lost – reconnecting…", "#ef6c00"),
        DISCONNECTING: ("Disconnecting…", "#757575"), FAILED: ("Connection failed", "#c62828"),
    }

    def _on_state_changed(self, state: str):
        text, color = self.STATE_STYLE.get(state, (state, "#000"))
        self.state_label.setText(f"<b style='color:{color}'>● {text}</b>")
        if state in (IDLE, FAILED) and self.recorder.active:
            self._toggle_recording()       # link is gone for good: finish the file
        self._update_buttons()

    def _update_buttons(self):
        state = self.source.state
        connected = state == STREAMING
        self.btn_connect.setEnabled(state in (IDLE, FAILED, SCANNING)
                                    and self.device_list.currentItem() is not None)
        self.btn_disconnect.setEnabled(state in (CONNECTING, STREAMING, RECONNECTING))
        finalizing = self.recorder.busy and not self.recorder.active
        self.btn_record.setEnabled((connected or self.recorder.active) and not finalizing)
        self.btn_record.setText("■ Stop recording" if self.recorder.active else "● Start recording")
        self.btn_marker.setEnabled(self.recorder.active)
        # Firmware filters and sample rate are locked while a recording is open
        locked = self.recorder.busy
        if self.source.settings_locked != locked:
            self.source.lock_settings(locked)
        for rb in self.rate_buttons.values():
            rb.setEnabled(not locked)
        available = self.current_params.get("filters", {})
        for key, cb in self.filter_checks.items():
            cb.setEnabled(key in available and not locked)

    # -- device -----------------------------------------------------------------
    def _scan(self):
        self.device_list.clear()
        self._devices.clear()
        self.source.start_scan()

    def _on_devices(self, devices):
        for name, addr, rssi in devices:
            text = f"{name}  [{addr}]  RSSI {rssi}"
            item = self._devices.get(addr)
            if item is None:
                item = QtWidgets.QListWidgetItem(text)
                item.setData(QtCore.Qt.UserRole, addr)
                self.device_list.addItem(item)
                self._devices[addr] = item
                if self.device_list.currentItem() is None:
                    self.device_list.setCurrentItem(item)
            else:
                item.setText(text)
        self._update_buttons()

    def _connect(self):
        item = self.device_list.currentItem()
        if item is None:
            return
        self.btn_connect.setEnabled(False)
        self.source.connect_device(item.data(QtCore.Qt.UserRole))

    def _disconnect(self):
        if self.recorder.active:
            self._toggle_recording()
        self.source.disconnect_device()

    def _on_connected(self, summary: dict):
        self.device_summary = summary
        self.device_label.setText(
            f"<b>{summary.get('name')}</b><br>{summary.get('address')}<br>"
            f"{summary.get('model')} · FW {summary.get('firmware')}<br>"
            f"backend {summary.get('backend')} · MTU {summary.get('mtu')} · "
            f"{summary.get('package_samples')} samples/packet")
        self._set_status("Streaming")
        self._update_buttons()

    def _on_disconnected(self):
        self.device_label.setText("Disconnected")
        self.battery_bar.setValue(0)
        self.battery_bar.setFormat("--")
        self._update_buttons()

    def _on_battery(self, pct: int):
        self.battery_bar.setValue(pct)
        self.battery_bar.setFormat(f"{pct}%")
        color = "#2e7d32" if pct > 40 else "#ef6c00" if pct > 15 else "#c62828"
        self.battery_bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; }}")

    def _on_params(self, params: dict):
        self.current_params = params
        rate = params.get("emg_rate")
        if rate in self.rate_buttons:
            with QtCore.QSignalBlocker(self.rate_group):
                self.rate_buttons[rate].setChecked(True)
        for key, cb in self.filter_checks.items():
            with QtCore.QSignalBlocker(cb):
                cb.setChecked(bool(params.get("filters", {}).get(key, False)))
        self._update_buttons()

    def _on_rate_toggled(self, rate: int, checked: bool):
        if checked and not self.source.set_emg_rate(rate):
            self._on_params(self.current_params)          # revert the radio button

    def _on_filter_toggled(self, key: str, on: bool):
        if not self.source.set_filter(key, on):
            self._on_params(self.current_params)          # revert the checkbox

    def _on_error(self, msg: str):
        self._set_status(f"⚠ {msg}")
        print(f"[error] {msg}")
        self._update_buttons()

    def _set_status(self, msg: str):
        self.statusBar().showMessage(msg, 10000)

    # -- recording --------------------------------------------------------------
    def _choose_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Recording folder", str(self.recorder.root))
        if d:
            self.recorder.root = Path(d)
            self.folder_label.setText(d)

    def _toggle_recording(self):
        """Start, or request stop (non-blocking; the status timer reports the outcome)."""
        if self.recorder.active:
            self.recorder.stop()
            self._set_status("Finalizing recording…")
        elif not self.recorder.busy:
            meta = {"device": self.device_summary,
                    "emg_rate": self.hub.emg_rate,
                    "imu_rate": self.hub.imu_rate,
                    "onboard_filters": self.current_params.get("filters", {})}
            try:
                session = self.recorder.start(self.session_name.text(), meta)
            except Exception as e:
                self._on_error(f"Could not start recording: {e}")
                return
            self._set_status(f"Recording to {session.folder}")
        self._update_record_label()
        self._update_buttons()

    def _update_record_label(self):
        s = self.recorder.session
        if s is None:
            self.record_label.setText("Idle")
            return
        emg = s.counts["emg"]
        if s.state == RECORDING:
            self.record_label.setText(
                f"<b style='color:#c62828'>● REC</b> {s.elapsed:6.1f} s · "
                f"{emg['written']}/{emg['received']} EMG samples written · "
                f"{s.counts['markers']['received']} markers")
        elif s.state == FINALIZING:
            self.record_label.setText(
                f"Finalizing… {emg['written']}/{emg['enqueued']} EMG samples written")
        elif s.state == SAVED:
            self.record_label.setText(
                f"<b style='color:#2e7d32'>Saved</b> {s.folder}<br>"
                f"{emg['written']} EMG / {s.counts['imu']['written']} IMU samples, "
                f"{s.counts['markers']['written']} markers, {s.elapsed:.1f} s")
        elif s.state == ERROR:
            suffix = " (still recording)" if s.accepting else ""
            self.record_label.setText(
                f"<b style='color:#c62828'>ERROR{suffix}:</b> {s.error}<br>"
                f"written {emg['written']} / received {emg['received']} EMG samples, "
                f"dropped {emg['dropped']} · {s.folder}")
        self.record_label.setWordWrap(True)

    def _add_marker(self):
        if not self.recorder.active:
            return
        n = self.recorder.session.counts["markers"]["received"]
        label = self.marker_edit.text().strip() or f"marker{n + 1}"
        self.recorder.add_marker(label)
        self._set_status(f"Marker '{label}' added")

    # -- timers -----------------------------------------------------------------
    def _refresh_plots(self):
        # Dashboards skip the redraw when neither data nor view settings changed
        if not self.tabs.currentWidget().refresh():
            return
        age = self.hub.emg_callback_age_ms()
        if age == age:      # not NaN
            self._age_ms = age if not self._age_ms else 0.9 * self._age_ms + 0.1 * age

    def _refresh_status(self):
        imp = self.hub.impedance
        self.contact_strip.update_contact(imp)
        self.equal.update_contact(imp)
        self.detail.update_contact(imp)
        self._update_record_label()
        if self.source.settings_locked != self.recorder.busy:
            self._update_buttons()          # e.g. session ended by a writer error
        if self._finalizing_seen and not self.recorder.busy:
            self._finalizing_seen = False
            self._update_buttons()
            if self._close_pending:
                self.close()
                return
        if self.recorder.busy and not self.recorder.active:
            self._finalizing_seen = True
        if self.source.state != STREAMING:
            self.status_text.setStyleSheet("")
            self.status_text.setText(f"backend: {self.source.backend_name() or '--'} · "
                                     f"{self.source.state}")
            return
        rate = self.hub.emg_rate
        actual = self.hub.emg_actual_rate()
        pkt = self.device_summary.get("package_samples", 0)
        # The band can deliver far fewer samples than its nominal rate (seen: ~505
        # of 1000 Hz on FW V1.0.4); time/frequency axes assume the nominal rate.
        mismatch = actual > 0 and abs(actual - rate) / rate > 0.05
        self.status_text.setStyleSheet("color:#c62828; font-weight:bold;" if mismatch else "")
        warn = "⚠ actual rate ≠ nominal, time/FFT axes off · " if mismatch else ""
        self.status_text.setText(
            f"{warn}EMG {actual:.1f}/{rate:g} Hz · "
            f"IMU {self.hub.imu_actual_rate():.1f}/{self.hub.imu_rate:g} Hz · "
            f"lost {self.hub.lost_packages} pkts / {self.hub.lost_samples} samples · "
            f"index gaps {self.hub.missing_samples} · segment {self.hub.emg_segment} · "
            f"{self._subscriber_error_text()}"
            f"packet {pkt / rate * 1000 if rate else 0:.0f} ms · "
            f"callback gap p95 {self.hub.emg_callback_gap_ms():.0f} ms · "
            f"callback→draw {self._age_ms:.0f} ms (host only) · "
            f"sensor→host latency: not measured · "
            f"backend {self.device_summary.get('backend', '')}")

    def _subscriber_error_text(self) -> str:
        errs = sum(self.hub.subscriber_errors.values())
        return f"⚠ {errs} subscriber errors · " if errs else ""

    def closeEvent(self, event):
        if self.recorder.busy:
            # Never block the event loop on the writer: stop, keep the window
            # until the status timer sees the session finish, then close.
            self.recorder.stop()
            self._close_pending = True
            self._finalizing_seen = True
            self._set_status("Finishing the recording before closing…")
            event.ignore()
            return
        self.plot_timer.stop()
        self.status_timer.stop()
        self.source.shutdown()
        event.accept()
