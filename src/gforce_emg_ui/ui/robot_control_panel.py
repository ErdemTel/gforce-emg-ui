"""Robot operator controls embedded in the existing EMG application."""

import sys

from PyQt5 import QtCore, QtWidgets

from ..live.robot_control import RobotControlClient


class RobotControlPanel(QtWidgets.QWidget):
    mode_requested = QtCore.pyqtSignal(str)
    start_requested = QtCore.pyqtSignal(str)
    pause_requested = QtCore.pyqtSignal()
    updated = QtCore.pyqtSignal()
    close_ready = QtCore.pyqtSignal()

    def __init__(self, side, host, port, parent=None, client=None):
        super().__init__(parent)
        self.client = client if client is not None else RobotControlClient(
            side=side, host=host, port=port)
        self._close_reported = False
        self._draft_dirty = False
        self._selection_seen = None
        self._build()
        self.render()
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    @staticmethod
    def _label(text=""):
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setTextFormat(QtCore.Qt.PlainText)
        label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        return label

    def _build(self):
        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        self.labels = {}
        for key in ("identity", "execution", "phase"):
            label = self.labels[key] = self._label()
            label.setStyleSheet("font-size:16px; font-weight:bold;")
            layout.addWidget(label)
        layout.addWidget(self._label(
            "Manual control needs no armband or classifier. EMG acquisition, recording and "
            "the classifier remain in this same window."))
        form = QtWidgets.QFormLayout()
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Manual", "EMG"])
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        form.addRow("Control source", self.mode_combo)
        self.kp_edit, self.kd_edit = QtWidgets.QLineEdit("1.0"), QtWidgets.QLineEdit("1.0")
        for name, widget in (("Kp multiplier", self.kp_edit), ("Kd multiplier", self.kd_edit)):
            widget.textEdited.connect(self._draft_edited)
            form.addRow(name, widget)
        layout.addLayout(form)
        self.draft_label = self._label()
        layout.addWidget(self.draft_label)
        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start control")
        self.apply_button = QtWidgets.QPushButton("Apply gains")
        self.pause_button = QtWidgets.QPushButton("Pause updates")
        self.reset_button = QtWidgets.QPushButton("Reset gains to 1 / 1")
        self.start_button.clicked.connect(self._start)
        self.apply_button.clicked.connect(self._apply)
        self.pause_button.clicked.connect(self._pause)
        self.reset_button.clicked.connect(self._reset)
        for widget in (self.start_button, self.apply_button, self.pause_button, self.reset_button):
            buttons.addWidget(widget)
        layout.addLayout(buttons)
        poses = QtWidgets.QHBoxLayout()
        self.hold_button = QtWidgets.QPushButton("Hold current pose")
        self.release_button = QtWidgets.QPushButton("Release pose for teleop")
        self.home_button = QtWidgets.QPushButton("Return full robot home")
        self._home_tooltip = (
            "Move both arms to the lab home pose, then both hands to zero using current gains. "
            "Exit teleop and clear the whole path first. Pauses gains and releases this GUI's hold. "
            "Cancel stops new targets; it is not an emergency stop.")
        self.home_button.setToolTip(self._home_tooltip)
        self.hold_button.clicked.connect(lambda: self.client.submit("hold_pose", enabled=True))
        self.release_button.clicked.connect(lambda: self.client.submit("hold_pose", enabled=False))
        self.home_button.clicked.connect(self._home)
        poses.addWidget(self.hold_button)
        poses.addWidget(self.release_button)
        layout.addLayout(poses)
        layout.addWidget(self.home_button)
        layout.addWidget(self._label(
            "Full home moves BOTH ARMS to the lab home pose, then BOTH HANDS to zero. "
            "Exit teleop and clear the whole path first; no collision planning. "
            "Gains stay unchanged and updates stay paused afterward. "
            "Cancel stops new targets; it is not an emergency stop."))
        layout.addWidget(self._label(
            "Edit multipliers, then Apply. Start uses Requested gains below. "
            "Pause stops gain updates; pose hold and motor enable are separate."))
        self.prediction_label = self._label("EMG stream off — Manual does not require EMG.")
        layout.addWidget(self.prediction_label)
        state_form = QtWidgets.QFormLayout()
        for key, title in (
                ("mode", "Confirmed control source"), ("requested", "Requested gains"),
                ("verified", "Last verified device gains"), ("readback_age", "Readback age"),
                ("feedback", "Hand feedback"), ("emg", "Latest EMG level"),
                ("mapping", "EMG development presets (not calibrated)"),
                ("hold", "Pose control"), ("reason", "Pending / blocked reason")):
            self.labels[key] = self._label()
            state_form.addRow(title, self.labels[key])
        layout.addLayout(state_form)
        self.notice_label = self._label()
        layout.addWidget(self.notice_label)
        layout.addStretch()
        scroll.setWidget(body)
        outer.addWidget(scroll)

    def _draft(self):
        try:
            from ..live.direct_gains import validate_pair
            return validate_pair(float(self.kp_edit.text()), float(self.kd_edit.text()))
        except (TypeError, ValueError):
            return None

    def _set_draft(self, kp, kd):
        self.kp_edit.setText(f"{kp:g}")
        self.kd_edit.setText(f"{kd:g}")
        self._draft_dirty = False

    def _draft_edited(self, *_):
        self._draft_dirty = True
        self.render()

    def _mode_changed(self, index):
        mode = "emg" if index else "manual"
        if self.client.submit("set_mode", mode=mode):
            if mode == "manual":
                self._set_draft(1., 1.)
            self.mode_requested.emit(mode)
        self.render()

    def _start(self):
        state = self.client.model.state
        if state and self.client.submit("start"):
            self.start_requested.emit(state["mode"])
        self.render()

    def _apply(self):
        pair = self._draft()
        if pair is not None and self.client.submit("set_manual", kp_ratio=pair[0], kd_ratio=pair[1]):
            self._draft_dirty = False
        self.render()

    def _pause(self):
        if self.client.submit("pause"):
            self.pause_requested.emit()
        self.render()

    def _home(self):
        if self.client.submit("return_full_home"):
            self.pause_requested.emit()
        self.render()

    def _reset(self):
        if self.client.submit("reset"):
            self._set_draft(1., 1.)
            self.mode_requested.emit("manual")
        self.render()

    def set_prediction_status(self, text):
        self.prediction_label.setText(text)

    def render(self):
        model = self.client.model
        display = model.display()
        for key, label in self.labels.items():
            label.setText(display.get(key, "Unknown"))
        notice = model.notice
        if notice.startswith("Request accepted: "):
            notice = "Last request acknowledgment: " + notice[len("Request accepted: "):]
        self.notice_label.setText(notice)
        state = model.state
        if state is not None:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(1 if state["mode"] == "emg" else 0)
            self.mode_combo.blockSignals(False)
            selected = state["selected"]
            key = selected["kp_ratio"], selected["kd_ratio"]
            if state["mode"] == "manual" and self._selection_seen != key and not self._draft_dirty:
                self._set_draft(*key)
            self._selection_seen = key
        pair = self._draft()
        matches = bool(pair and state and pair == (
            state["selected"]["kp_ratio"], state["selected"]["kd_ratio"]))
        self.draft_label.setText(
            "Manual draft is inactive in EMG mode; Requested gains below come from EMG."
            if state and state["mode"] == "emg" else
            "Enter both multipliers from 0 to 1." if pair is None else
            "Fields match Requested gains; device readback is shown separately." if matches else
            "Draft changes — Apply gains selects this pair before Start.")
        busy = self.client.busy or self.client.closing
        self.pause_button.setText("Cancel return home" if state and
                                  state.get("home", {}).get("active") else "Pause updates")
        for action, button in (
                ("start", self.start_button), ("set_manual", self.apply_button),
                ("pause", self.pause_button), ("reset", self.reset_button),
                ("hold_pose", self.hold_button), ("release_pose", self.release_button),
                ("return_full_home", self.home_button)):
            allowed, reason = model.allowed(action)
            if action == 'return_full_home':
                button.setToolTip(self._home_tooltip if allowed else reason)
            blocked = self.client.closing if action == "pause" else busy
            if action == "set_manual":
                blocked = blocked or pair is None
            if action == "start" and state and state["mode"] == "manual":
                blocked = blocked or not matches
            button.setEnabled(allowed and not blocked)
        allowed, _ = model.allowed("set_mode")
        self.mode_combo.setEnabled(allowed and not busy)
        for widget in (self.kp_edit, self.kd_edit):
            widget.setEnabled(bool(not self.client.closing and state and state["mode"] == "manual"))

    def tick(self):
        try:
            self.client.tick()
        except Exception as error:
            self.client.model.notice = f"Robot communication failed: {error}"
        self.render()
        self.updated.emit()
        if self.client.close_complete and not self._close_reported:
            self._close_reported = True
            if not self.client.close_confirmed:
                print(self.client.model.notice, file=sys.stderr)
            self.close_ready.emit()

    def refresh(self):
        return False

    def begin_close(self):
        self.client.begin_close()
        self.pause_requested.emit()
        self.render()

    def shutdown(self):
        self.timer.stop()
        self.client.close()
