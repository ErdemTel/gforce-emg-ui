"""Main window: control panel (device, battery, contact, settings, recording)
plus dashboards in tabs. Only the visible dashboard is redrawn; the status bar shows on
every tab when the Live Classifier's robot output is armed."""

import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .. import __version__
from ..collection.controller import REVIEW
from ..config import (EMG_CHANNELS, EMG_RATES, FILTERS, PLOT_INTERVAL_MS,
                      STATUS_INTERVAL_MS)
from ..contact import LEVEL_COLORS, contact_color, format_kohm
from ..hub import DataHub
from ..recorder import ERROR, FINALIZING, RECORDING, SAVED, Recorder
from ..sources import (CONNECTING, DISCONNECTING, FAILED, IDLE, RECONNECTING, SCANNING,
                       STREAMING, Source)
from .collection_panel import StiffnessCollectionPanel
from .emg_detail import DetailDashboard
from .emg_equal import EqualScaleDashboard
from .imu_view import ImuDashboard
from .live_view import LiveClassifierDashboard


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
        # Only on change: a restyle re-lays out and repaints the whole side panel
        for ch, chip in enumerate(self.chips):
            text = f"CH{ch + 1}\n{format_kohm(impedance[ch])}"
            css = (f"background:{contact_color(impedance[ch])}; color:white;"
                   "border-radius:4px; font-weight:bold;")
            if chip.text() != text:
                chip.setText(text)
            if chip.styleSheet() != css:
                chip.setStyleSheet(css)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, hub: DataHub, source: Source, data_dir: Path,
                 robot_target: Optional[Tuple[str, int]] = None,
                 emg_producer: bool = False,
                 robot_control_target: Optional[Tuple[str, int]] = None,
                 hand: str = "right", robot_client=None):
        """robot_target: (ip, port) prefilled in the Live Classifier's robot output (CLI
        --robot-udp); never arms it."""
        super().__init__()
        self.unified_control = robot_control_target is not None
        self.emg_producer = bool(emg_producer or self.unified_control)
        self.robot_control = None
        self._emg_requested = False
        self._control_instance = None
        self._manual_handover = None
        self._prediction_notice = ""
        self._control_clock = time.monotonic
        self.hub = hub
        self.source = source
        self._data_dir = Path(data_dir)
        self.recorder = Recorder(hub, data_dir)
        self.device_summary = {}
        self.current_params = {}
        self._devices = {}
        self._age_ms = 0.0
        self._close_pending = False
        self._finalizing_seen = False
        title = ("gForce EMG + Sharpa Control" if self.unified_control else
                 "gForce EMG Producer" if self.emg_producer else "gForce EMG UI")
        self.setWindowTitle(f"{title} {__version__}")
        self.resize(1500, 950)

        self.equal = EqualScaleDashboard(hub)
        self.detail = DetailDashboard(hub)
        self.imu = ImuDashboard(hub)
        self.live = LiveClassifierDashboard(
            hub, source, robot_target=robot_control_target or robot_target,
            emg_producer=self.emg_producer, managed_output=self.unified_control)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.equal, "EMG – Equal scale")
        self.tabs.addTab(self.detail, "EMG – Detail")
        self.tabs.addTab(self.imu, "IMU")
        self.tabs.addTab(self.live, "Live Classifier")
        if self.unified_control:
            from .robot_control_panel import RobotControlPanel
            self.robot_control = RobotControlPanel(hand, *robot_control_target, client=robot_client)
            self.tabs.insertTab(0, self.robot_control, "Robot control")
            self.tabs.setCurrentWidget(self.robot_control)
            self.robot_control.mode_requested.connect(self._robot_mode_requested)
            self.robot_control.start_requested.connect(self._robot_start_requested)
            self.robot_control.pause_requested.connect(self._robot_pause_requested)
            self.robot_control.updated.connect(self._sync_prediction_stream)
            self.robot_control.close_ready.connect(self._robot_close_ready)

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
        self.robot_indicator = QtWidgets.QLabel()       # visible on every tab while armed
        self.robot_indicator.setStyleSheet("background:#c62828; color:white; "
                                           "font-weight:bold; padding:1px 8px;")
        self.robot_indicator.setVisible(False)
        self.statusBar().addPermanentWidget(self.robot_indicator)
        self.live.robot_armed.connect(self._on_robot_armed)

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
        rv = QtWidgets.QVBoxLayout(rec)
        # Both subtabs share self.recorder; at most one recording at a time
        self.collection_panel = StiffnessCollectionPanel(
            self.hub, self.recorder, self.source, self._data_dir, context=self._collection_context)
        self.collection = self.collection_panel.controller
        rv.addWidget(self.collection_panel.summary_label)
        manual = QtWidgets.QWidget()
        rl = QtWidgets.QFormLayout(manual)
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
        self.record_label.setWordWrap(True)
        rl.addRow(self.record_label)
        self.marker_edit = QtWidgets.QLineEdit()
        self.marker_edit.setPlaceholderText("label, e.g. fist_on")
        self.btn_marker = QtWidgets.QPushButton("Add marker (Ctrl+M)")
        rl.addRow("Marker", self.marker_edit)
        rl.addRow(self.btn_marker)
        # Tab bar + framed pages, only the current one shown: unlike QTabWidget the
        # section is as tall as the visible page. Switching never touches acquisition,
        # the recorder or the collection timer.
        self.record_tabs = QtWidgets.QTabBar()
        self.record_tabs.setExpanding(False)
        self.record_tabs.setDrawBase(False)
        self.record_tabs.addTab("Manual")
        self.record_tabs.addTab("Stiffness Collection")
        pages = QtWidgets.QFrame()
        pages.setFrameShape(QtWidgets.QFrame.StyledPanel)
        pl = QtWidgets.QVBoxLayout(pages)
        pl.setContentsMargins(2, 2, 2, 2)
        self._record_pages = (manual, self.collection_panel)
        for page in self._record_pages:
            pl.addWidget(page)
        self.collection_panel.wheel_guard.guard(self.record_tabs)
        self.record_tabs.currentChanged.connect(self._show_record_page)
        self._show_record_page(0)
        rv.setSpacing(0)
        rv.addWidget(self.record_tabs)
        rv.addWidget(pages)
        v.addWidget(rec)
        v.addStretch(1)
        self.collection_panel.recorder_changed.connect(self._update_buttons)

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

    def _show_record_page(self, index: int):
        for i, page in enumerate(self._record_pages):
            page.setVisible(i == index)

    STATE_STYLE = {
        IDLE: ("Idle", "#757575"), SCANNING: ("Scanning…", "#1565c0"),
        CONNECTING: ("Connecting…", "#1565c0"), STREAMING: ("Streaming", "#2e7d32"),
        RECONNECTING: ("Link lost – reconnecting…", "#ef6c00"),
        DISCONNECTING: ("Disconnecting…", "#757575"), FAILED: ("Connection failed", "#c62828"),
    }

    def _on_state_changed(self, _signalled_state: str = ""):
        # State signals from SDK threads are queued and may arrive out of order
        # (observed on hardware), so always act on the source's current state.
        state = self.source.state
        text, color = self.STATE_STYLE.get(state, (state, "#000"))
        self.state_label.setText(f"<b style='color:{color}'>● {text}</b>")
        if self.collection.owns_recorder:
            # A stiffness trial never continues across a link loss or reconnect
            if state in (IDLE, FAILED):
                self.collection_panel.abort("disconnected")
            elif state == RECONNECTING:
                self.collection_panel.abort("link_lost")
        elif state in (IDLE, FAILED) and self.recorder.active:
            self._toggle_recording()       # link is gone for good: finish the file
        self._update_buttons()

    def _update_buttons(self):
        state = self.source.state
        connected = state == STREAMING
        self.btn_connect.setEnabled(state in (IDLE, FAILED, SCANNING)
                                    and self.device_list.currentItem() is not None)
        self.btn_disconnect.setEnabled(state in (CONNECTING, STREAMING, RECONNECTING))
        finalizing = self.recorder.busy and not self.recorder.active
        trial = self.collection.owns_recorder
        self.btn_record.setEnabled((connected or self.recorder.active) and not finalizing)
        self.btn_record.setText("■ Stop (aborts stiffness trial)" if trial and self.recorder.active
                                else "■ Stop recording" if self.recorder.active
                                else "● Start recording")
        self.btn_marker.setEnabled(self.recorder.active and not trial)
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
        if self.collection.owns_recorder:
            self.collection_panel.abort("user_disconnect")
        elif self.recorder.active:
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
        css = f"QProgressBar::chunk {{ background: {color}; }}"
        if self.battery_bar.styleSheet() != css:
            self.battery_bar.setStyleSheet(css)

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

    def _on_robot_armed(self, armed: bool, target: str):
        label = ("EMG PREDICTION STREAM ON" if self.unified_control else
                 "PREDICTION STREAM ON" if self.emg_producer else "ROBOT OUTPUT ARMED")
        text = f"{label} → {target}" if armed else ""
        if self.robot_indicator.text() != text:
            self.robot_indicator.setText(text)
        self.robot_indicator.setVisible(armed)

    # -- unified robot control ---------------------------------------------------
    def _robot_mode_requested(self, mode):
        self._prediction_notice = ""
        if mode == "manual" and self.live.robot.armed:
            pending = self.robot_control.client.pending
            request_id = pending[1]["request_id"] if pending is not None else None
            self._manual_handover = {"request_id": request_id, "started": self._control_clock()}
            # A prediction Stop arriving before set_mode would pause the EMG
            # controller and turn a running handover into a stopped selection.
            self._emg_requested = True
        else:
            self._manual_handover = None
            self._emg_requested = mode == "emg"
            if not self._emg_requested:
                self.live.disarm_robot("Manual selected")
        self._sync_prediction_stream()

    def _robot_start_requested(self, mode):
        self._manual_handover = None
        self._prediction_notice = ""
        self._emg_requested = mode == "emg"
        self._sync_prediction_stream()

    def _robot_pause_requested(self):
        self._manual_handover = None
        self._prediction_notice = ""
        self._emg_requested = False
        self.live.disarm_robot("Robot updates paused")

    def _sync_prediction_stream(self):
        panel = self.robot_control
        if panel is None:
            return
        model, client = panel.client.model, panel.client
        state = model.state
        if state is not None:
            instance = state["instance_id"]
            if self._control_instance is not None and instance != self._control_instance:
                self._manual_handover = None
                self._emg_requested = False
                self.live.disarm_robot("Controller restarted")
            self._control_instance = instance
        handover = self._manual_handover
        if handover is not None:
            acknowledgment = getattr(client, "last_ack", None)
            rejected = bool(acknowledgment and handover["request_id"]
                            and acknowledgment["request_id"] == handover["request_id"]
                            and not acknowledgment["accepted"])
            confirmed = bool(model.fresh and state and
                             (state["mode"] == "manual" or state.get("pending_mode") == "manual"))
            if confirmed:
                self._manual_handover = None
                self._emg_requested = False
                self.live.disarm_robot("Manual handover confirmed")
            elif rejected or not model.fresh or self._control_clock() - handover["started"] >= 3.5:
                reason = (acknowledgment["message"] if rejected else
                          "controller state unavailable" if not model.fresh else "acknowledgment timed out")
                self._manual_handover = None
                self._emg_requested = False
                # Fence delayed mode requests with a newer Pause operation and
                # stop renewing the old EMG-control lease before closing its stream.
                client.submit("pause")
                self._prediction_notice = (
                    f"Manual switch unconfirmed ({reason}); Pause requested, EMG stream stopped.")
                self.live.disarm_robot("Manual handover unconfirmed")
        ready_mode = bool(self._emg_requested and model.fresh and state
                          and state["mode"] == "emg"
                          and state.get("owner_id") in (None, model.client_id)
                          and not client.closing)
        if not ready_mode:
            self.live.disarm_robot("EMG input is not selected for this operator")
            message = ("EMG stream off — closing robot control." if client.closing else
                       "EMG waiting for fresh controller state." if self._emg_requested and not model.fresh else
                       "EMG waiting for the confirmed EMG control source." if self._emg_requested else
                       "EMG stream off — Manual does not require EMG.")
            panel.set_prediction_status(self._prediction_notice or message)
            return
        blockers = [b for b in self.live.robot_blockers() if b != "armed"]
        if blockers:
            self.live.disarm_robot("EMG classifier is not ready")
            panel.set_prediction_status("EMG waiting: " + " · ".join(blockers))
            return
        if not self.live.robot.armed:
            try:
                self.live.robot.configure(self.live.robot_draft())
                self.live.robot.arm()
                self.live._robot_changed()
            except (OSError, RuntimeError, ValueError) as error:
                panel.set_prediction_status(f"EMG stream unavailable: {error}")
                return
        panel.set_prediction_status(
            "Switching to Manual — keeping EMG stream until the controller confirms the handover."
            if self._manual_handover is not None else
            "EMG prediction stream on. Requested and verified gains are shown below.")

    def _robot_close_ready(self):
        if self._close_pending and not self.recorder.busy and not self.collection.busy:
            self.close()

    # -- recording --------------------------------------------------------------
    def _choose_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Recording folder", str(self.recorder.root))
        if d:
            self.recorder.root = Path(d)
            self.folder_label.setText(d)

    def _collection_context(self) -> dict:
        """Device / settings info saved with every stiffness-collection attempt."""
        return {"device": dict(self.device_summary), "params": dict(self.current_params)}

    def _toggle_recording(self):
        """Start, or request stop (non-blocking; the status timer reports the outcome)."""
        if self.collection.owns_recorder:
            # Manual Stop while a stiffness trial records: abort that trial properly
            if self.collection_panel.abort("manual_stop"):
                self._set_status("Stiffness trial aborted (manual Stop); finalizing…")
            self._update_buttons()
            return
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

    def _add_marker(self):
        if not self.recorder.active or self.collection.owns_recorder:
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
        self.live.robot_tick()              # any tab: auto-disarm checks, indicator
        imp = self.hub.impedance
        self.contact_strip.update_contact(imp)
        self.equal.update_contact(imp)
        self.detail.update_contact(imp)
        self._update_record_label()
        if self.source.settings_locked != self.recorder.busy:
            self._update_buttons()          # e.g. session ended by a writer error
        if self._finalizing_seen and not self.recorder.busy and not self.collection.busy:
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
        self._emg_requested = False
        self.live.disarm_robot("app closing")   # at once, even if a recording delays the close
        if self.robot_control is not None:
            self.robot_control.begin_close()
        if self.collection.state == REVIEW:   # not accepted: deleted like a rejected attempt
            self.collection.submit_review("reject", comment="app closed before review")
        if self.recorder.busy or self.collection.busy:
            # Never block the event loop on the writer: stop, keep the window
            # until the status timer sees the session (and trial.json) finish, then close.
            if self.collection.owns_recorder:
                self.collection_panel.abort("app_close")
            elif self.recorder.busy:
                self.recorder.stop()
            self._close_pending = True
            self._finalizing_seen = True
            self._set_status("Finishing the recording before closing…")
            event.ignore()
            return
        if self.robot_control is not None and not self.robot_control.client.close_complete:
            self._close_pending = True
            self._set_status("Pausing robot updates before closing…")
            event.ignore()
            return
        self.plot_timer.stop()
        self.status_timer.stop()
        self.collection_panel.shutdown()
        self.live.shutdown()
        if self.robot_control is not None:
            self.robot_control.shutdown()
        self.source.shutdown()
        event.accept()
