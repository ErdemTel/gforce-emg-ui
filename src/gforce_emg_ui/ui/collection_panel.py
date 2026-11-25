"""'Stiffness Collection' tab of the Recording section: a thin Qt layer over
CollectionController (timed, labelled trials on the shared Recorder).

Plan mode: Object / Target / Movement show the controller's pending trial. Only
user activation (`activated`) edits it, one field at a time; programmatic display
uses QSignalBlocker and never counts as an edit. Surface Press and Open hand are
task conditions listed after the objects; each is ticked separately (opt-in).
Surface Press is always stationary; Open hand has the plan's movements.

Folder mode (default): only a folder and stiffness 0 / 1; the Plan / Manual form is
hidden. Same protocol, recorder, trial.json and review."""

import re
from pathlib import Path
from typing import Callable, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from ..collection.controller import (AVAILABILITY_BYPASS_TEXT, FINALIZING, FOLDER_OBJECT,
                                     HAND_SIDES, IDLE, MANUAL_OVERRIDE, REVIEW, REVIEW_REASONS,
                                     ROLES, RUNNING, SAVING_REVIEW, CollectionController,
                                     TrialSpec, folder_ids)
from ..collection.plan import (DEFAULT_PLAN_COUNTS, EMPTY_HAND, EXTENDABLE, EXTENDED,
                               FIELD_NAMES, TASK_CONDITIONS, TASKS, attempt_category,
                               default_posture, describe, merge_objects, task_motion_counts,
                               trial_values)
from ..collection.protocol import (ABORTED, HOLD, OPEN_HAND, OPEN_HAND_POSTURE, PREPARE, REST,
                                   SETTLE, MOTION_ORDER, MOTIONS, STATIONARY, SURFACE_PRESS,
                                   SURFACE_PRESS_POSTURE, TARGET_ORDER, TARGETS,
                                   ProtocolSettings, operator_note, phase_instruction,
                                   trial_type)

TICK_MS = 50
PHASE_COLORS = {PREPARE: "#1565c0", SETTLE: "#ef6c00", HOLD: "#2e7d32", REST: "#757575",
                ABORTED: "#c62828"}
AUDIO_EVENTS = ("PREPARE_START", "SETTLE_START", "HOLD_START", "REST_START", "ABORT")
PLAN_MODE, MANUAL_MODE, FOLDER_MODE = "plan", "manual", "folder"
STIFFNESS_TARGETS = {0: "low", 1: "high"}        # folder mode: stiffness value -> target
MOTION_SHORT = {STATIONARY: "stationary", "wrist_flex_ext": "wrist flex/ext",
                "forearm_rotation": "forearm rotation", "reach_sweep": "reach/sweep"}
POSTURE_SHORT = {SURFACE_PRESS_POSTURE: "press", OPEN_HAND_POSTURE: "open"}   # narrow panel
EXTEND_LABELS = {SURFACE_PRESS: "Surface Press", OPEN_HAND: "Open-Hand"}     # "Add N … Trials"


class WheelGuard(QtCore.QObject):
    """Wheel over a tab bar, or an unfocused combo / spin box, scrolls the side panel
    instead of switching tabs or silently changing a setting."""

    def guard(self, *widgets):
        for w in widgets:
            if not isinstance(w, QtWidgets.QTabBar):
                w.setFocusPolicy(QtCore.Qt.StrongFocus)     # wheel no longer focuses it
            w.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() != QtCore.QEvent.Wheel or not (isinstance(obj, QtWidgets.QTabBar)
                                                    or not obj.hasFocus()):
            return False
        area = obj.parentWidget()
        while area is not None and not isinstance(area, QtWidgets.QAbstractScrollArea):
            area = area.parentWidget()
        if area is not None:             # scroll the enclosing panel exactly once
            vp = area.viewport()
            gpos = ev.globalPosition()
            QtWidgets.QApplication.sendEvent(vp, QtGui.QWheelEvent(
                QtCore.QPointF(vp.mapFromGlobal(gpos.toPoint())), gpos, ev.pixelDelta(),
                ev.angleDelta(), ev.buttons(), ev.modifiers(), ev.phase(), ev.inverted()))
        return True


class LiveText(QtWidgets.QWidget):
    """Rich text that changes every tick (countdowns). Unlike QLabel, setText() only
    repaints this widget and never re-runs the layout, so the side panel is not
    re-laid out and repainted 20 times a second (visible flicker)."""

    def __init__(self, lines: int = 1, px: int = 0, bold: bool = False, pad: int = 8,
                 parent=None):
        super().__init__(parent)
        font = self.font()
        if px:
            font.setPixelSize(px)
        font.setBold(bold)
        self.setFont(font)
        self._html, self._bg, self._fg = "", None, None
        self._doc = QtGui.QTextDocument(self)
        self._doc.setDefaultFont(font)
        self._doc.setDocumentMargin(0)
        self.setFixedHeight(QtGui.QFontMetrics(font).lineSpacing() * lines + pad)

    def sizeHint(self):
        return QtCore.QSize(0, self.height())

    def minimumSizeHint(self):
        return self.sizeHint()

    def text(self) -> str:
        return self._html

    def setText(self, html: str):
        if html != self._html:
            self._html = html
            self.update()

    def setColors(self, bg, fg):
        if (bg, fg) != (self._bg, self._fg):
            self._bg, self._fg = bg, fg
            self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        if self._bg:
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(self._bg))
            p.drawRoundedRect(QtCore.QRectF(self.rect()), 4, 4)
        if self._html:
            color = f"color:{self._fg};" if self._fg else ""
            self._doc.setTextWidth(max(10, self.width() - 8))
            self._doc.setHtml(f"<div align='center' style='{color}'>{self._html}</div>")
            p.translate(4, max(0.0, (self.height() - self._doc.size().height()) / 2))
            self._doc.drawContents(p)


class StiffnessCollectionPanel(QtWidgets.QWidget):
    """Emits recorder_changed when the collection takes or releases the Recorder."""

    recorder_changed = QtCore.pyqtSignal()

    def __init__(self, hub, recorder, source, data_dir: Path,
                 context: Optional[Callable[[], dict]] = None, parent=None,
                 settings: Optional[ProtocolSettings] = None):
        super().__init__(parent)
        self.controller = CollectionController(hub, recorder, source, settings=settings,
                                               context=context)
        self.controller.on_cue = self._on_cue
        self.output_root = Path(data_dir) / "stiffness_collection"
        self.folder: Optional[Path] = None           # folder mode: chosen by the operator
        self._last_state = None
        self._last_owns = False
        self._reviews_seen = 0
        self._shown_version = None
        self._planned_key = None
        self._posture_obj = None
        self._session_seen = ""
        self._objects_shown = None
        self._plans_shown = None
        self._build()
        self.wheel_guard = WheelGuard(self)
        self.wheel_guard.guard(*self.findChildren(QtWidgets.QComboBox),
                               *self.findChildren(QtWidgets.QAbstractSpinBox))
        # Status line shown above the Manual / Stiffness tabs (visible from either tab)
        self.summary_label = LiveText(lines=2, pad=4)
        self.summary_label.setVisible(False)
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self.tick)
        self.timer.start(TICK_MS)
        self._apply_settings()
        self._on_mode_changed()
        self.tick()

    # -- layout -------------------------------------------------------------------
    def _build(self):
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(4, 6, 4, 4)
        self.sim_banner = QtWidgets.QLabel(
            "<b>SIMULATION</b> – synthetic data, saved under <i>simulated/</i>, "
            "never counted as real data")
        self.sim_banner.setWordWrap(True)
        self.sim_banner.setStyleSheet("background:#6a1b9a; color:white; padding:3px;"
                                      "border-radius:3px;")
        self.sim_banner.setVisible(self.controller.is_simulation)
        v.addWidget(self.sim_banner)

        self.simple_form = QtWidgets.QWidget()          # folder mode: folder, stiffness
        sf = QtWidgets.QFormLayout(self.simple_form)
        sf.setContentsMargins(0, 0, 0, 0)
        sf.setVerticalSpacing(6)
        self.folder_label = QtWidgets.QLabel("<i>not chosen</i>")
        self.btn_folder = QtWidgets.QPushButton("Choose…")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.folder_label, 1)
        row.addWidget(self.btn_folder)
        sf.addRow("1. Folder", row)
        self.stiff_group = QtWidgets.QButtonGroup(self)
        row = QtWidgets.QHBoxLayout()
        for value in STIFFNESS_TARGETS:
            rb = QtWidgets.QRadioButton(str(value))
            rb.setStyleSheet("font-size:16px; font-weight:bold;")
            self.stiff_group.addButton(rb, value)
            row.addWidget(rb)
        row.addStretch(1)
        sf.addRow("2. Stiffness", row)
        v.addWidget(self.simple_form)

        self.full_form = QtWidgets.QWidget()            # Plan / Manual (hidden in folder mode)
        form = self._form = QtWidgets.QFormLayout(self.full_form)
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(4)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("Plan (dev. trials)", PLAN_MODE)
        self.mode_combo.addItem("Manual (practice / test)", MANUAL_MODE)
        self.mode_combo.addItem("Folder (stiffness 0 / 1)", FOLDER_MODE)
        self.mode_combo.setCurrentIndex(self.mode_combo.findData(FOLDER_MODE))
        self.mode_combo.setToolTip("Plan: the persistent development plan (200 trials; +30 "
                                   "Surface Press, +35 Open hand).\n"
                                   "Manual: choose condition/target yourself for practice or "
                                   "held-out tests (never counted toward the plan).")
        form.addRow("Mode", self.mode_combo)
        self.participant_edit = QtWidgets.QLineEdit()
        self.participant_edit.setPlaceholderText("e.g. P01")
        self.session_edit = QtWidgets.QLineEdit()
        self.session_edit.setPlaceholderText("e.g. S01")
        self.hand_combo = QtWidgets.QComboBox()
        self.hand_combo.addItems(HAND_SIDES)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.participant_edit)
        row.addWidget(QtWidgets.QLabel("Session"))
        row.addWidget(self.session_edit)
        form.addRow("Participant", row)
        form.addRow("Hand", self.hand_combo)

        self.plan_combo = QtWidgets.QComboBox()
        self.btn_new_plan = QtWidgets.QPushButton("New…")
        self.btn_new_plan.setToolTip("Create and save a new 200-trial plan for this participant")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.plan_combo, 1)
        row.addWidget(self.btn_new_plan)
        self.plan_row = QtWidgets.QWidget()
        self.plan_row.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        form.addRow("Plan", self.plan_row)
        self.avail_row = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(self.avail_row)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(2)
        grid.addWidget(QtWidgets.QLabel("Available objects:"), 0, 0, 1, 3)
        self.avail_checks = {}
        for i, oid in enumerate(o for o in DEFAULT_PLAN_COUNTS if o != EMPTY_HAND):
            cb = QtWidgets.QCheckBox(oid)
            cb.setChecked(True)
            cb.toggled.connect(self._on_available_changed)
            self.avail_checks[oid] = cb
            grid.addWidget(cb, 1 + i // 3, i % 3)
        empty = QtWidgets.QCheckBox(EMPTY_HAND)          # always eligible until done
        empty.setChecked(True)
        empty.setEnabled(False)
        empty.setToolTip("Always included until its planned trials are complete")
        grid.addWidget(empty, 1 + len(self.avail_checks) // 3, len(self.avail_checks) % 3)
        self.task_checks = {}                           # task conditions: opt-in, unticked
        for i, (cond, name) in enumerate(TASK_CONDITIONS.items()):
            cb = QtWidgets.QCheckBox(name)
            cb.toggled.connect(self._on_available_changed)
            self.task_checks[cond] = cb
            grid.addWidget(cb, 2 + len(self.avail_checks) // 3 + i, 0, 1, 3)
        self.surface_check = self.task_checks[SURFACE_PRESS]
        self.open_check = self.task_checks[OPEN_HAND]
        form.addRow(self.avail_row)                    # full width: fits the narrow panel
        self.next_label = QtWidgets.QLabel("--")        # explicit short lines, no wrapping
        form.addRow(self.next_label)
        self.sel_row = QtWidgets.QWidget()               # Automatic / Edited + reset
        row = QtWidgets.QHBoxLayout(self.sel_row)
        row.setContentsMargins(0, 0, 0, 0)
        self.sel_state = QtWidgets.QLabel()
        self.btn_reset_sel = QtWidgets.QPushButton("Reset edits")
        self.btn_reset_sel.setToolTip("Clear this trial's edits and use the random selection")
        row.addWidget(self.sel_state, 1)
        row.addWidget(self.btn_reset_sel)
        form.addRow(self.sel_row)
        self.sel_note = QtWidgets.QLabel()               # plain text, wrapped
        self.sel_note.setTextFormat(QtCore.Qt.PlainText)
        self.sel_note.setWordWrap(True)
        self.sel_note.setVisible(False)
        form.addRow(self.sel_note)

        self.object_combo = QtWidgets.QComboBox()
        form.addRow("Object", self.object_combo)
        self.posture_combo = QtWidgets.QComboBox()
        self.posture_combo.setEditable(True)
        self.posture_combo.addItems(["fist", "grasp", SURFACE_PRESS_POSTURE, OPEN_HAND_POSTURE])
        self.posture_desc = QtWidgets.QLineEdit()
        self.posture_desc.setPlaceholderText("posture description")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.posture_combo)
        row.addWidget(self.posture_desc, 1)
        form.addRow("Posture", row)
        self.target_combo = QtWidgets.QComboBox()
        for t in TARGET_ORDER:
            self.target_combo.addItem(f"{TARGETS[t]['display'].title()} (s={TARGETS[t]['s']:g})", t)
        form.addRow("Target", self.target_combo)
        self.motion_combo = QtWidgets.QComboBox()
        for i, m in enumerate(MOTION_ORDER):
            self.motion_combo.addItem(MOTIONS[m]["display"], m)
            self.motion_combo.setItemData(i, MOTIONS[m]["movement"] or "No movement",
                                          QtCore.Qt.ToolTipRole)
        form.addRow("Movement", self.motion_combo)
        self.role_combo = QtWidgets.QComboBox()
        for r in ROLES:
            self.role_combo.addItem(r, r)
        form.addRow("Role", self.role_combo)
        self.trial_spin = QtWidgets.QSpinBox()
        self.trial_spin.setRange(1, 999)
        form.addRow("Trial #", self.trial_spin)
        self.out_label = QtWidgets.QLabel()
        self.btn_out = QtWidgets.QPushButton("Change…")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.out_label, 1)
        row.addWidget(self.btn_out)
        form.addRow("Output", row)
        self._show_output_root()
        self.notes_edit = QtWidgets.QLineEdit()
        self.notes_edit.setPlaceholderText("optional")
        form.addRow("Notes", self.notes_edit)
        v.addWidget(self.full_form)
        for combo in (self.mode_combo, self.hand_combo, self.plan_combo, self.object_combo,
                      self.target_combo, self.motion_combo, self.role_combo, self.posture_combo):
            # Narrow side panel: let combos shrink (full text stays in the popup)
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
        self.plan_combo.setMinimumContentsLength(5)

        self.btn_advanced = QtWidgets.QToolButton()
        self.btn_advanced.setText("▸ Protocol && plan settings")
        self.btn_advanced.setCheckable(True)
        self.btn_advanced.setAutoRaise(True)
        v.addWidget(self.btn_advanced)
        self.advanced = self._build_advanced()
        self.advanced.setVisible(False)
        v.addWidget(self.advanced)

        self.phase_label = LiveText(lines=1, px=26, bold=True, pad=20)
        self.target_label = QtWidgets.QLabel()
        self.target_label.setAlignment(QtCore.Qt.AlignCenter)
        self.target_label.setStyleSheet("font-size:15px; font-weight:bold;")
        self.instruction_label = QtWidgets.QLabel()
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setStyleSheet("font-size:13px;")
        self.time_label = LiveText(lines=2, pad=4)
        for w in (self.phase_label, self.target_label, self.instruction_label, self.time_label):
            v.addWidget(w)
        self.ready_label = QtWidgets.QLabel()
        self.ready_label.setWordWrap(True)
        v.addWidget(self.ready_label)

        self.btn_start = QtWidgets.QPushButton("▶ Start trial")
        self.btn_start.setStyleSheet("font-weight:bold;")
        v.addWidget(self.btn_start)
        row = QtWidgets.QHBoxLayout()
        self.btn_abort = QtWidgets.QPushButton("Abort")
        self.btn_extend = QtWidgets.QPushButton()
        for b in (self.btn_abort, self.btn_extend):
            row.addWidget(b)
        v.addLayout(row)

        self.review_box = QtWidgets.QGroupBox("Review attempt")
        rv = QtWidgets.QVBoxLayout(self.review_box)
        # Short explicit lines, no wrapping: wrapped rich text with <br> under-reports
        # its height in this narrow panel and squeezes the whole page
        self.review_summary = QtWidgets.QLabel()
        rv.addWidget(self.review_summary)
        self.review_flags = QtWidgets.QLabel()           # plain text, wrapped
        self.review_flags.setTextFormat(QtCore.Qt.PlainText)
        self.review_flags.setWordWrap(True)
        self.review_flags.setStyleSheet("color:#c62828;")
        rv.addWidget(self.review_flags)
        grid = QtWidgets.QGridLayout()
        self.reason_checks = {}
        short = {"wrong_effort": "Wrong effort", "effort_not_maintained": "Effort lost",
                 "incorrect_movement": "Wrong move", "hand_movement": "Hand moved",
                 "grasp_change": "Grasp changed", "incorrect_posture": "Wrong posture",
                 "fatigue": "Fatigue", "contact_problem": "Contact", "other": "Other"}
        for i, (key, text) in enumerate(REVIEW_REASONS.items()):
            cb = QtWidgets.QCheckBox(short.get(key, text))
            cb.setToolTip(text)
            self.reason_checks[key] = cb
            grid.addWidget(cb, i // 2, i % 2)
        rv.addLayout(grid)
        self.review_comment = QtWidgets.QLineEdit()
        self.review_comment.setPlaceholderText("comment (optional)")
        rv.addWidget(self.review_comment)
        row = QtWidgets.QHBoxLayout()
        self.btn_accept = QtWidgets.QPushButton("Accept")
        self.btn_reject = QtWidgets.QPushButton("Reject")
        self.btn_repeat = QtWidgets.QPushButton("Repeat")
        self.btn_reject.setToolTip("Delete this attempt; the trial comes back later")
        self.btn_repeat.setToolTip("Delete this attempt and retry the same trial now")
        for b in (self.btn_accept, self.btn_reject, self.btn_repeat):
            row.addWidget(b)
        rv.addLayout(row)
        v.addWidget(self.review_box)

        self.progress_label = QtWidgets.QLabel()
        self.progress_label.setWordWrap(True)
        v.addWidget(self.progress_label)
        self.message_label = QtWidgets.QLabel()
        self.message_label.setWordWrap(True)
        v.addWidget(self.message_label)

        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.participant_edit.editingFinished.connect(self._reopen_participant)
        self.plan_combo.activated.connect(self._on_plan_selected)
        self.btn_new_plan.clicked.connect(self._new_plan)
        self.object_combo.currentIndexChanged.connect(self._on_object_changed)
        self._sel_combos = {"object_id": self.object_combo, "target_label": self.target_combo,
                            "motion_condition": self.motion_combo}
        for name, combo in self._sel_combos.items():
            combo.activated.connect(lambda i, n=name, cb=combo: self._on_field_activated(
                n, cb.itemData(i)))
        self.btn_reset_sel.clicked.connect(self._reset_selection)
        self.session_edit.editingFinished.connect(self._on_session_changed)
        self.btn_out.clicked.connect(self._choose_output)
        self.btn_folder.clicked.connect(self._choose_folder)
        self.btn_advanced.toggled.connect(self._toggle_advanced)
        self.btn_start.clicked.connect(self.start_trial)
        self.btn_abort.clicked.connect(lambda: self.abort("user_abort"))
        self.btn_extend.clicked.connect(self.controller.extend_rest)
        self.btn_accept.clicked.connect(lambda: self.review("accept"))
        self.btn_reject.clicked.connect(lambda: self.review("reject"))
        self.btn_repeat.clicked.connect(lambda: self.review("repeat"))

    def _build_advanced(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        f.setContentsMargins(8, 0, 0, 0)
        s = self.controller.settings
        self.dur_spins = {}

        def spin(value, lo, hi, step=0.5, decimals=1):
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setDecimals(decimals)
            sb.setSingleStep(step)
            sb.setSuffix(" s")
            sb.setValue(value)
            sb.valueChanged.connect(self._apply_settings)
            return sb
        for key, label in (("prepare_s", "PREPARE"), ("settle_s", "SETTLE"),
                           ("hold_s", "HOLD"), ("rest_s", "REST")):
            self.dur_spins[key] = spin(getattr(s, key), 0.1, 600)
            f.addRow(label, self.dur_spins[key])
        self.dur_spins["rest_extension_s"] = spin(s.rest_extension_s, 1, 600, 5)
        f.addRow("Extend rest by", self.dur_spins["rest_extension_s"])
        self.dur_spins["readiness_s"] = spin(s.readiness_s, 0, 60)
        f.addRow("Readiness period", self.dur_spins["readiness_s"])
        self.dur_spins["emg_freshness_s"] = spin(s.emg_freshness_s, 0.05, 10, 0.1, 2)
        f.addRow("EMG freshness", self.dur_spins["emg_freshness_s"])
        self.audio_check = QtWidgets.QCheckBox("Beep on phase changes")
        self.audio_check.setToolTip("Optional short system beep; the visual cue is primary. "
                                    "Audio failures never stop a trial.")
        self.audio_check.setChecked(s.audio_cues)
        self.audio_check.toggled.connect(self._apply_settings)
        f.addRow(self.audio_check)
        self._adv_form = f
        self.seed_edit = QtWidgets.QLineEdit()
        self.seed_edit.setPlaceholderText("random")
        f.addRow("New-plan seed", self.seed_edit)
        self.extend_buttons = {}
        for cond, task in TASKS.items():
            b = QtWidgets.QPushButton(f"Add {task.total} {EXTEND_LABELS[cond]} Trials")
            b.setToolTip("Append {} low / {} medium / {} high ".format(*task.counts)
                         + ("stationary surface-press" if task.stationary_only else
                            "open-hand (split over the movements)")
                         + " trials to the selected plan (once; same plan ID)")
            b.clicked.connect(lambda _, cond=cond: self._extend_plan(cond))
            self.extend_buttons[cond] = b
            f.addRow(b)
        self.btn_extend_plan = self.extend_buttons[SURFACE_PRESS]
        self.btn_extend_open = self.extend_buttons[OPEN_HAND]
        self.objects_edit = QtWidgets.QPlainTextEdit()
        self.objects_edit.setFixedHeight(96)
        self.objects_edit.setToolTip("One object per line: id = name. IDs of the default "
                                     "objects are fixed; extra IDs (e.g. obj06) are for "
                                     "held-out tests and practice.")
        f.addRow("Objects", self.objects_edit)
        self.btn_save_objects = QtWidgets.QPushButton("Save object names")
        self.btn_save_objects.clicked.connect(self._save_objects)
        f.addRow(self.btn_save_objects)
        self._plan_settings = (self.seed_edit, *self.extend_buttons.values(),
                               self.objects_edit, self.btn_save_objects)
        return w

    # -- user actions ------------------------------------------------------------------
    @property
    def mode(self) -> str:
        return self.mode_combo.currentData()

    def _toggle_advanced(self, on: bool):
        self.advanced.setVisible(on)
        title = "Protocol settings" if self.mode == FOLDER_MODE else "Protocol && plan settings"
        self.btn_advanced.setText(("▾" if on else "▸") + " " + title)

    def _apply_settings(self, *_):
        if self.controller.state != IDLE and self.controller.state != REVIEW:
            return
        vals = {k: sb.value() for k, sb in self.dur_spins.items()}
        self.controller.settings = ProtocolSettings(**{**self.controller.settings.to_dict(),
                                                       **vals,
                                                       "audio_cues": self.audio_check.isChecked()})
        self.btn_extend.setText(f"Extend rest +{self.controller.settings.rest_extension_s:g} s")

    def _reopen_participant(self):
        c = self.controller
        pid = self.participant_edit.text().strip()
        if c.state != IDLE or (pid == c.participant_id and self.output_root == c.output_root
                               and c.folder is None):
            return
        c.open_participant(self.output_root, pid)
        self._plans_shown = None
        self._objects_shown = None

    def _choose_output(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Collection output folder",
                                                       str(self.output_root))
        if d:
            self.set_output_root(Path(d))

    def _choose_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Recording folder",
                                                       str(self.folder or self.output_root))
        if d:
            self.set_folder(Path(d))

    def set_folder(self, path: Path) -> bool:
        """Folder mode: save the next trials directly in `path`."""
        if self.controller.state != IDLE or self.controller.extending:
            return False
        self.folder = Path(path)
        pid, sid = folder_ids(self.folder)
        text = str(self.folder)
        fm = self.folder_label.fontMetrics()
        self.folder_label.setText(fm.elidedText(text, QtCore.Qt.ElideLeft, 105))
        self.folder_label.setToolTip(f"{text}\nSaved as participant {pid}, session {sid}")
        if self.mode == FOLDER_MODE:
            self.controller.open_folder(self.folder)
        return True

    def _show_output_root(self):
        text = str(self.output_root)
        fm = self.out_label.fontMetrics()
        self.out_label.setText(fm.elidedText(text, QtCore.Qt.ElideMiddle, 110))
        self.out_label.setToolTip(text)

    def set_output_root(self, path: Path):
        self.output_root = Path(path)
        self._show_output_root()
        self.controller.output_root = None          # force a reload
        self._reopen_participant()

    def _on_mode_changed(self, *_):
        folder = self.mode == FOLDER_MODE
        self.simple_form.setVisible(folder)
        self.full_form.setVisible(not folder)
        for w in self._plan_settings:
            w.setVisible(not folder)
            lab = self._adv_form.labelForField(w)
            if lab is not None:
                lab.setVisible(not folder)
        self._toggle_advanced(self.btn_advanced.isChecked())
        c = self.controller
        if folder and self.folder is not None and c.folder != self.folder:
            c.open_folder(self.folder)
        elif not folder and c.folder is not None:
            self._reopen_participant()
        plan = self.mode == PLAN_MODE
        for w, show in ((self.plan_row, plan), (self.avail_row, plan), (self.next_label, plan),
                        (self.sel_row, plan), (self.trial_spin, not plan)):
            w.setVisible(show)
            lab = self._form.labelForField(w)
            if lab is not None:
                lab.setVisible(show)
        if plan:
            self.role_combo.setCurrentIndex(ROLES.index("development"))
        else:
            self.controller.clear_edits()        # one-trial edits belong to Plan mode
            self.sel_note.setVisible(False)
            for combo in self._sel_combos.values():
                if combo.currentIndex() < 0:     # Manual mode always has a value
                    combo.setCurrentIndex(0)
            if self.role_combo.currentData() == "development":
                self.role_combo.setCurrentIndex(ROLES.index("practice"))
        self._shown_version = None
        self._planned_key = None
        self._posture_obj = None
        self._show_planned()

    def _on_available_changed(self, *_):
        self.controller.set_available_objects(
            [oid for oid, cb in {**self.avail_checks, **self.task_checks}.items()
             if cb.isChecked()])
        self._planned_key = None

    def _on_field_activated(self, name: str, value):
        """The operator picked a value: edit only that field of the next planned trial."""
        if self.mode != PLAN_MODE:
            return
        self.controller.edit_selection(name, value)   # refused (busy): the display reverts
        self._refresh_selection()

    def _reset_selection(self):
        self.controller.clear_edits()
        self._refresh_selection()

    def _refresh_selection(self):
        self._planned_key = None
        self._show_planned()
        self._show_phase()
        self._show_controls()

    def _on_session_changed(self):
        sid = self.session_edit.text().strip()
        if sid != self._session_seen:
            self._session_seen = sid
            self.controller.clear_edits()        # no draft carried into another session

    def _on_plan_selected(self, idx: int):
        self.controller.select_plan(self.plan_combo.itemData(idx))

    def _new_plan(self, confirm: bool = True):
        c = self.controller
        self._reopen_participant()
        if c.participant_id is None or c.loading:
            self._message("Enter a valid participant ID first (and wait for it to load).",
                          error=True)
            return
        if confirm and c.plans:
            ans = QtWidgets.QMessageBox.question(
                self, "New plan", f"{c.participant_id} already has {len(c.plans)} plan(s). "
                "Create another plan? Existing plans and their attempts are kept.")
            if ans != QtWidgets.QMessageBox.Yes:
                return
        text = self.seed_edit.text().strip()
        try:
            seed = int(text) if text else None
        except ValueError:
            self._message("Seed must be an integer.", error=True)
            return
        plan = c.create_plan(seed)
        if plan is not None:
            self._message(f"Saving plan {plan.plan_id} (seed {plan.seed})…")

    def _extension_text(self, cond: str) -> str:
        """Confirmation text: plan ID, actual before / after totals, what is added."""
        task, plan = TASKS[cond], self.controller.plan
        n = task.total
        counts = ", ".join(f"{k} {TARGETS[t]['display']}"
                           for t, k in zip(TARGET_ORDER, task.counts))
        lines = [f"Plan {plan.plan_id} ({plan.participant_id}): {plan.total} → {plan.total + n} "
                 "planned trials.", ""]
        if task.stationary_only:
            lines.append(f"Adds {n} stationary index-finger presses on a desk: {counts}.")
        else:
            split = task_motion_counts(task, plan.motions)
            lines.append(f"Adds {n} open-hand trials (hand comfortably open, no fist, no "
                         f"object): {counts} in total, split over the plan's movements:")
            lines += [f"   {MOTION_SHORT[m]}: " + " · ".join(
                f"{TARGETS[t]['display']} {split[(cond, t, m)]}" for t in TARGET_ORDER)
                for m in task.plan_motions(plan.motions)]
            lines.append(f"Stationary relaxed LOW: {split[(cond, 'low', STATIONARY)]} trials "
                         "(moving LOW = no added hand or grip tension while moving).")
        lines += ["", "Existing trials, trial IDs, accepted recordings, progress and earlier "
                  "extensions are kept; the current plan file is backed up to plan_backups/ "
                  "first. This can be done only once per plan."]
        return "\n".join(lines)

    def _extend_plan(self, cond: str = SURFACE_PRESS, confirm: bool = True) -> bool:
        """Add a task condition's trials to the selected plan, after confirmation."""
        c, task = self.controller, TASKS[cond]
        title = f"Add {task.total} {EXTEND_LABELS[cond]} Trials"
        status, why = c.extension_status(cond)
        if status != EXTENDABLE:
            if confirm:
                box = (QtWidgets.QMessageBox.information if status == EXTENDED
                       else QtWidgets.QMessageBox.warning)
                box(self, title, why)
            self._message(why, error=status != EXTENDED)
            return False
        if confirm:
            ans = QtWidgets.QMessageBox.question(
                self, title, self._extension_text(cond),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if ans != QtWidgets.QMessageBox.Yes:
                return False
        ok = c.extend_plan(cond)
        if not ok:
            self._message(c.error, error=True)
        self.tick()
        return ok

    def _save_objects(self):
        items = []
        for line in self.objects_edit.toPlainText().splitlines():
            if "=" in line:
                oid, name = line.split("=", 1)
                items.append((oid.strip(), name.strip()))
        if self.controller.save_objects(items):
            self._message("Saving object names…")

    def _on_object_changed(self, *_):
        if self.mode != MANUAL_MODE:
            return
        oid = self.object_combo.currentData() or EMPTY_HAND
        posture, desc = default_posture(oid)
        self.posture_combo.setCurrentText(posture)
        self.posture_desc.setText(desc)
        if oid in TASKS and TASKS[oid].stationary_only:  # surface press: stationary only
            self.motion_combo.setCurrentIndex(self.motion_combo.findData(STATIONARY))

    def _next_folder_trial(self) -> int:
        """Folder mode: trial m### after the highest one saved in the folder."""
        nums = [int(r["trial_id"][1:]) for r in self.controller.records.values()
                if r.get("plan_id") is None and re.fullmatch(r"m\d{3}", str(r.get("trial_id")))]
        return min(max(nums, default=0) + 1, 999)

    def _folder_spec(self) -> TrialSpec:
        pid, sid = folder_ids(self.folder) if self.folder is not None else ("", "")
        return TrialSpec(
            participant_id=pid, session_id=sid, object_id=FOLDER_OBJECT,
            object_name="Unspecified", posture_id=FOLDER_OBJECT,
            target_label=STIFFNESS_TARGETS.get(self.stiff_group.checkedId(), ""),
            motion_condition=STATIONARY, dataset_role="development",
            output_root=self.folder or "", trial_number=self._next_folder_trial(),
            folder=self.folder)

    def spec(self) -> TrialSpec:
        c = self.controller
        if self.mode == FOLDER_MODE:
            return self._folder_spec()
        planned = c.planned_trial() if self.mode == PLAN_MODE else None
        if planned:                              # Plan mode: from the controller, not the combos
            oid, target, posture = (planned["object_id"], planned["target_label"],
                                    planned["posture_id"])
            motion = c.plan.motion_of(planned)
        elif self.mode == PLAN_MODE:             # no valid trial: the requested values (blocked)
            req = c.pending.requested if c.pending is not None else {}
            oid, target, motion = (req.get(f) or "" for f in
                                   ("object_id", "target_label", "motion_condition"))
            posture = default_posture(oid)[0] if oid else ""
        else:
            oid = self.object_combo.currentData() or ""
            target, motion = self.target_combo.currentData(), self.motion_combo.currentData()
            posture = self.posture_combo.currentText().strip()
        return TrialSpec(
            participant_id=self.participant_edit.text().strip(),
            session_id=self.session_edit.text().strip(),
            hand_side=self.hand_combo.currentText(),
            object_id=oid, object_name=c.object_name(oid),
            posture_id=posture,
            posture_description=self.posture_desc.text().strip(),
            target_label=target, motion_condition=motion,
            dataset_role="development" if self.mode == PLAN_MODE else self.role_combo.currentData(),
            output_root=self.output_root, trial_number=self.trial_spin.value(),
            notes=self.notes_edit.text().strip(), planned_trial=planned,
            plan_id=c.plan.plan_id if planned and c.plan else None)

    def start_blockers(self) -> list:
        c = self.controller
        blockers = c.start_blockers(self.spec())
        if self.mode == FOLDER_MODE:                 # say what to choose, not why the IDs fail
            first, noise = [], []
            if self.folder is None:
                first.append("Choose a folder")
                noise += ["Participant ID", "Session ID", "Output directory",
                          "Participant data not loaded", "Development trials"]
            if self.stiff_group.checkedId() not in STIFFNESS_TARGETS:
                first.append("Choose stiffness 0 or 1")
                noise.append("Target:")
            return first + [b for b in blockers if not any(n in b for n in noise)]
        if self.mode == PLAN_MODE and c.planned_trial() is None:
            p = c.pending
            if c.plan is None:
                first = "No plan: create one with New… (or use Manual for practice)"
            elif p is not None and p.edited:
                first = p.problem or "No matching planned trial"
            elif c.progress is not None and c.progress.remaining:
                first = ("No remaining trials for the ticked objects: tick more objects "
                         "or choose object, target and movement")
            else:
                first = "Plan complete"
            noise = ("come from the plan", "match the planned", "Object ID:", "Posture ID:",
                     "Target:", "Movement:") + tuple(t.short for t in TASKS.values())
            blockers = [first] + [b for b in blockers if not any(n in b for n in noise)]
        return blockers

    def start_trial(self) -> bool:
        if self.mode == PLAN_MODE and self.controller.planned_trial() is None:
            p = self.controller.pending
            self._message(p.problem if p is not None and p.problem else
                          "No planned trial (create or select a plan).", error=True)
            return False
        ok = self.controller.start_trial(self.spec())
        if not ok:
            self._message(self.controller.error, error=True)
        self.tick()
        return ok

    def abort(self, reason: str) -> bool:
        ok = self.controller.abort(reason)
        self.tick()
        return ok

    def review(self, decision: str) -> bool:
        reasons = [k for k, cb in self.reason_checks.items() if cb.isChecked()]
        ok = self.controller.submit_review(decision, reasons, self.review_comment.text())
        if not ok and self.controller.error:
            self._message(self.controller.error, error=True)
        self.tick()
        return ok

    def shutdown(self):
        self.timer.stop()
        self.controller.io.shutdown(wait=False)

    # -- cues -----------------------------------------------------------------------
    def _on_cue(self, ev) -> dict:
        """Runs inside the transition: update the visual cue now; optional beep."""
        self._show_phase()
        cue = {"visual": "requested"}
        if self.controller.settings.audio_cues and ev.name in AUDIO_EVENTS:
            try:
                QtWidgets.QApplication.beep()
                cue["audio"] = "requested (system beep)"
            except Exception as e:          # audio problems never stop the trial
                cue["audio"] = f"failed: {e}"
        return cue

    # -- periodic update ----------------------------------------------------------------
    def tick(self):
        c = self.controller
        c.tick()
        owns = c.owns_recorder
        if c.reviews_saved != self._reviews_seen:
            self._reviews_seen = c.reviews_saved
            self._after_review()
        if c.state != self._last_state or owns != self._last_owns:
            self._last_state, self._last_owns = c.state, owns
            self.recorder_changed.emit()
        self._sync_lists()
        self._show_planned()
        self._show_phase()
        self._show_controls()
        if (c.version, self.mode) != self._shown_version:
            self._shown_version = (c.version, self.mode)
            self._show_progress()
        if c.error:
            self._message(c.error, error=True)
        elif c.message:
            self._message(c.message)

    def _after_review(self):
        c = self.controller
        if self.mode == MANUAL_MODE and c.last_decision in ("accept", "reject"):
            self.trial_spin.setValue(self.trial_spin.value() + 1)
        for cb in self.reason_checks.values():
            cb.setChecked(False)
        self.review_comment.clear()

    def _sync_lists(self):
        c = self.controller
        if c.objects != self._objects_shown:
            self._objects_shown = list(c.objects)
            current = self.object_combo.currentData()
            with QtCore.QSignalBlocker(self.object_combo):
                self.object_combo.clear()
                for oid, name in list(c.objects) + list(TASK_CONDITIONS.items()):
                    self.object_combo.addItem(f"{oid} – {name}" if name != oid else oid, oid)
                i = self.object_combo.findData(current)
                self.object_combo.setCurrentIndex(max(i, 0))
            self.objects_edit.setPlainText("\n".join(f"{o} = {n}" for o, n in c.objects))
            for oid, cb in self.avail_checks.items():
                cb.setToolTip(c.object_name(oid))
            self._on_object_changed()
        plans = [(p.plan_id, p.total) for p in c.plans]
        if plans != self._plans_shown:
            self._plans_shown = plans
            with QtCore.QSignalBlocker(self.plan_combo):
                self.plan_combo.clear()
                for i, p in enumerate(c.plans):
                    self.plan_combo.addItem(p.plan_id, p.plan_id)
                    self.plan_combo.setItemData(i, f"created {p.created} · seed {p.seed} · "
                                                   f"{p.total} trials", QtCore.Qt.ToolTipRole)
                if c.plan is not None:
                    self.plan_combo.setCurrentIndex(self.plan_combo.findData(c.plan.plan_id))
        elif c.plan is not None and self.plan_combo.currentData() != c.plan.plan_id:
            with QtCore.QSignalBlocker(self.plan_combo):
                self.plan_combo.setCurrentIndex(self.plan_combo.findData(c.plan.plan_id))

    def _show_planned(self):
        """Display the controller's next trial. Cheap per tick: text only on change,
        combos only when they differ, nothing while one of their lists is open."""
        c = self.controller
        if self.mode != PLAN_MODE:
            return
        idle = c.state == IDLE
        if idle and any(cb.view().isVisible() for cb in self._sel_combos.values()):
            return
        key = (c.selection_version, c.version, c.state, c.loading, c.plan is None)
        if key != self._planned_key:
            self._planned_key = key
            self._show_selection_text()
        if idle and c.pending is not None:
            self._sync_selection_combos()

    def _sync_selection_combos(self):
        """Programmatic display of the pending values (never an edit)."""
        c = self.controller
        req = c.pending.requested
        for name, combo in self._sel_combos.items():
            i = combo.findData(req[name]) if req.get(name) else -1
            if combo.currentIndex() != i:
                with QtCore.QSignalBlocker(combo):
                    combo.setCurrentIndex(i)
        obj, t = req.get("object_id"), c.planned_trial()
        posture = t["posture_id"] if t else (default_posture(obj)[0] if obj else "")
        if self.posture_combo.currentText() != posture:
            with QtCore.QSignalBlocker(self.posture_combo):
                self.posture_combo.setCurrentText(posture)
        if obj != self._posture_obj:             # posture follows the object (fist / grasp)
            self._posture_obj = obj
            self.posture_desc.setText(default_posture(obj)[1] if obj else "")

    def _trial_lines(self, tid, values: dict, posture: str, v1: bool) -> str:
        c = self.controller
        obj, tgt, motion = (values.get(f) for f in ("object_id", "target_label",
                                                     "motion_condition"))
        name = c.object_name(obj) if obj else ""
        name = name if len(name) <= 22 else name[:21] + "…"
        head = f"<b>{tid}</b> of {c.plan.total}" if tid else "<b>no match</b>"
        what = name if obj in TASK_CONDITIONS else f"{obj} ({name})" if obj else "--"
        return (f"Next: {head} · <b>{TARGETS[tgt]['display'] if tgt in TARGETS else '--'}</b>"
                f" · {POSTURE_SHORT.get(posture, posture) or '--'}<br>{what}<br>"
                f"Movement: <b>{MOTION_SHORT.get(motion, '--')}</b>"
                + (" (plan v1: no movement)" if v1 else ""))

    def _show_selection_text(self):
        c = self.controller
        p, idle = c.pending, c.state == IDLE
        state, note, color = "", "", "#546e7a"
        if c.loading:
            text = "Next: loading…"
        elif c.plan is None:
            text = "Next: no plan yet.<br>Enter a participant, then New…"
        elif idle and p is not None and p.edited:
            t = c.planned_trial()
            obj = p.requested.get("object_id")
            posture = t["posture_id"] if t else (default_posture(obj)[0] if obj else "")
            diff = c.differing_fields()
            auto = c.plan.trials[p.proposal] if p.proposal is not None else None
            text = self._trial_lines(t["trial_id"] if t else None, p.requested, posture,
                                     c.plan.motions is None)
            text += "<br><small>" + (
                ("changed: " + ", ".join(FIELD_NAMES[f] for f in diff)) if diff and auto
                else "same values as the proposal" if auto else "no automatic proposal") + (
                f" · auto {auto['trial_id']}" if auto else "") + "</small>"
            state = "<b style='color:#ef6c00'>Edited</b> (this trial only)"
            parts = ([p.problem] if p.problem else []) + list(p.warnings) + (
                [f"Movement set to stationary: {TASKS[obj].short} is stationary only."]
                if p.enforced and obj in TASKS else [])
            if auto:
                parts.append(f"Automatic proposal: {auto['trial_id']} "
                             f"{describe(trial_values(c.plan, auto))}.")
                if obj and default_posture(obj)[0] != auto["posture_id"]:
                    parts.append(f"Posture {default_posture(obj)[0]} follows {obj}.")
            note = "\n".join(parts)
            color = "#c62828" if p.problem else "#ef6c00" if p.warnings else color
        else:
            t = c.planned_trial() if idle else (c.spec.planned_trial if c.spec else None)
            sel = (p.proposal_selection if p is not None else None) if idle else c.attempt_selection
            if t is None and c.progress is not None and c.progress.remaining:
                text = ("Next: <b>none</b> for the ticked objects<br>"
                        f"{c.progress.remaining} trials need other objects")
            elif t is None:
                text = f"Next: <b>plan complete</b> ({c.plan.total} trials)"
            elif sel and sel.get("mode") == MANUAL_OVERRIDE:          # running an edited trial
                text = self._trial_lines(t["trial_id"], trial_values(c.plan, t), t["posture_id"],
                                         "motion_condition" not in t)
                text += "<br><small>edited: " + ", ".join(
                    FIELD_NAMES[f] for f in sel["differing_fields"] or sel["edited_fields"]) + \
                    "</small>"
                state = "<b style='color:#ef6c00'>Edited</b>"
                note = "\n".join(
                    ([AVAILABILITY_BYPASS_TEXT] if sel["availability_bypass"] else []) +
                    (["Target run preference bypassed."] if sel["ordering_preference_bypass"]
                     else []))
            else:
                text = self._trial_lines(t["trial_id"], trial_values(c.plan, t), t["posture_id"],
                                         "motion_condition" not in t)
                if sel:
                    text += f"<br><small>random pick of {sel['eligible']} eligible</small>"
                state = "<b style='color:#2e7d32'>Automatic</b> (random)" if sel else ""
        self.next_label.setText(text)
        self.sel_state.setText(state)
        self.sel_note.setText(note)
        css = f"color:{color};"
        if self.sel_note.styleSheet() != css:
            self.sel_note.setStyleSheet(css)
        if self.sel_note.isVisibleTo(self) != bool(note):
            self.sel_note.setVisible(bool(note))

    def _show_phase(self):
        c = self.controller
        m = c.machine
        spec = c.spec if c.spec and c.state != IDLE else self.spec()
        known = spec.target_label in TARGETS
        target = spec.target_label if known else "low"
        motion = spec.motion_condition if spec.motion_condition in MOTIONS else STATIONARY
        task = trial_type(spec.object_id)
        info = TARGETS[target]
        tgt_text = ((f"Target: {info['display']} (s = {info['s']:g})" if known else "Target: --")
                    + "<br><span style='font-weight:normal'>Movement: "
                    f"{MOTION_SHORT.get(spec.motion_condition, '--')}</span>")
        color, fg = "#eceff1", "#263238"
        if c.state == RUNNING and m is not None:
            phase = m.phase
            big = (f"{phase}  {m.phase_remaining():4.1f} s" if m.active
                   else "STOPPED" if phase == ABORTED else "TRIAL END")
            color, fg = PHASE_COLORS.get(phase, color), "white"
            instr = phase_instruction(phase, target, motion, task)
            times = f"elapsed {m.elapsed():5.1f} s · remaining {m.total_remaining():5.1f} s"
            summary = (f"<b style='color:#c62828'>● Stiffness trial {c.spec.trial_id}</b> "
                       f"{phase} {m.phase_remaining():.1f} s · {info['display']}")
        elif c.state == FINALIZING:
            aborted = m is not None and m.phase == ABORTED
            big = "STOPPED – SAVING…" if aborted else "SAVING…"
            if aborted:
                color, fg = PHASE_COLORS[ABORTED], "white"
            instr = phase_instruction(ABORTED if aborted else REST, target, motion, task)
            times = f"recorded {m.elapsed():.1f} s" if m else ""
            summary = "<b>Stiffness trial: finalizing…</b>"
        elif c.state in (REVIEW, SAVING_REVIEW):
            out = c.sidecar["outcome"] if c.sidecar else {}
            if out.get("recorder") != "saved" or not c.metadata_ok:
                big, color, fg = "ERROR – NOT SAVED", PHASE_COLORS[ABORTED], "white"
            elif out.get("protocol") != "completed":
                big, color, fg = "ABORTED – partial data", PHASE_COLORS[ABORTED], "white"
            else:
                big = "SAVED – review"
            instr = "Release and rest. Accept keeps the attempt; Reject / Repeat delete it."
            times = f"recorded {m.elapsed():.1f} s" if m else ""
            summary = "<b>Stiffness trial awaiting review</b>"
        else:
            ready = c.readiness is not None and c.readiness.ready
            big = "READY" if ready else "NOT READY"
            if ready:
                color, fg = "#c8e6c9", "#1b5e20"
            note = operator_note(target, task)
            instr = ("Rest between trials as long as needed; start when the participant is "
                     "ready." + (f"<br><i>Operator: {note}</i>" if known else ""))
            st = c.settings
            times = (f"{st.total_s:g} s trial: PREPARE {st.prepare_s:g} · SETTLE {st.settle_s:g}"
                     f" · HOLD {st.hold_s:g} · REST {st.rest_s:g} s")
            summary = ""
        self.phase_label.setText(big)
        self.phase_label.setColors(color, fg)
        self.target_label.setText(tgt_text)
        self.instruction_label.setText(instr)
        self.time_label.setText(times)
        if c.is_simulation and summary:
            summary += " <b style='color:#6a1b9a'>[SIMULATION]</b>"
        self.summary_label.setText(summary)
        if self.summary_label.isVisibleTo(self.summary_label.parentWidget()) != bool(summary):
            self.summary_label.setVisible(bool(summary))

    def _show_controls(self):
        c = self.controller
        idle = c.state == IDLE
        plan_mode = self.mode == PLAN_MODE
        free = idle and not c.extending          # no plan / participant switch while writing
        for w in (self.session_edit, self.hand_combo, self.notes_edit, self.posture_desc,
                  self.btn_advanced):
            w.setEnabled(idle)
        for w in (self.mode_combo, self.participant_edit, self.btn_out, self.btn_folder,
                  *self.stiff_group.buttons()):
            w.setEnabled(free)
        self.advanced.setEnabled(idle)
        self.plan_combo.setEnabled(free and plan_mode and len(c.plans) > 0)
        for cb in self.avail_checks.values():
            cb.setEnabled(idle and plan_mode)
        for cond, cb in self.task_checks.items():
            task, has = TASKS[cond], c.plan is not None and cond in c.plan.counts
            cb.setEnabled(idle and plan_mode and has)
            tip = (f"Include {task.short} trials in the random selection" if has else
                   f"This plan has no {task.short} trials (Protocol & plan settings → Add "
                   f"{task.total} {EXTEND_LABELS[cond]} Trials)")
            if cb.toolTip() != tip:
                cb.setToolTip(tip)
        for b in self.extend_buttons.values():
            b.setEnabled(idle and plan_mode and c.plan is not None and not c.busy
                         and not c.loading)
        self.btn_new_plan.setEnabled(free and plan_mode and c.participant_id is not None
                                     and not c.loading)
        # Plan mode: editable for the next trial only (role, trial ID, posture follow the plan)
        editable = not plan_mode or (c.pending is not None and not c.loading)
        for w in self._sel_combos.values():
            w.setEnabled(idle and editable)
        for w in (self.role_combo, self.trial_spin, self.posture_combo):
            w.setEnabled(idle and not plan_mode)
        self.btn_reset_sel.setEnabled(idle and plan_mode and c.pending is not None
                                      and bool(c.pending.edited))
        blockers = self.start_blockers() if idle else []
        self.btn_start.setEnabled(idle and not blockers)
        self.btn_abort.setEnabled(c.state == RUNNING)
        self.btn_extend.setEnabled(c.state == RUNNING and c.machine is not None
                                   and c.machine.phase == REST)
        if idle:
            if not blockers:
                text = "<b style='color:#2e7d32'>✔ Ready</b> – stream stable"
            else:
                text = "<b style='color:#c62828'>Not ready:</b> " + " · ".join(blockers[:5])
            self.ready_label.setText(text)
        else:
            self.ready_label.setText("")
        review = c.state in (REVIEW, SAVING_REVIEW)
        if self.review_box.isVisibleTo(self) != review:
            self.review_box.setVisible(review)
        if review:
            can = c.state == REVIEW
            self.btn_accept.setEnabled(can and c.acceptable)
            tip = "" if c.acceptable else "; ".join(c.accept_blockers)
            if tip != self.btn_accept.toolTip():
                self.btn_accept.setToolTip(tip)
            self.btn_reject.setEnabled(can)
            self.btn_repeat.setEnabled(can)
            summary, flags = self._review_text()
            self.review_summary.setText(summary)
            self.review_flags.setText(flags)
            if self.review_flags.isVisibleTo(self) != bool(flags):
                self.review_flags.setVisible(bool(flags))

    def _review_text(self):
        """(short rich-text lines, plain-text details that may wrap)."""
        c = self.controller
        sc = c.sidecar or {}
        out = sc.get("outcome", {})
        cand = sc.get("training_candidate") or {}
        q = cand.get("quality_status")
        lines = [f"<b>{sc.get('trial_id')}</b> · a{sc.get('attempt_number', 0):02d} · "
                 f"{sc.get('condition', {}).get('object_id')} · "
                 f"{sc.get('label', {}).get('target_label', '').upper()}",
                 f"protocol {out.get('protocol')}",
                 f"recorder {out.get('recorder')}",
                 f"trial.json {'written' if c.metadata_ok else '<b>NOT written</b>'}",
                 "<b style='color:#2e7d32'>HOLD quality clean</b>" if q == "clean" else
                 f"<b style='color:#c62828'>HOLD quality {q} → repeat</b>"]
        details = []
        if out.get("abort_reason"):
            details.append(f"Aborted: {out.get('abort_reason_text') or out['abort_reason']}.")
        if cand.get("quality_flags"):
            details.append("HOLD flags: " + ", ".join(cand["quality_flags"]) + ".")
        other = [b for b in c.accept_blockers if not b.startswith("HOLD quality")]
        if other:                                    # HOLD flags are already listed above
            details.append("Accept disabled: " + "; ".join(other) + ".")
        return "<br>".join(lines), " ".join(details)

    def _show_progress(self):
        c = self.controller
        sim = " <b style='color:#6a1b9a'>(SIMULATED – not real data)</b>" if c.is_simulation else ""
        if self.mode == FOLDER_MODE:
            if self.folder is None:
                self.progress_label.setText("Choose the folder to save the trials in.")
                return
            counts = {value: 0 for value in STIFFNESS_TARGETS}
            for rec in c.records.values():
                if attempt_category(rec, c.live_ids) == "accepted":
                    for value, target in STIFFNESS_TARGETS.items():
                        counts[value] += (rec.get("label") or {}).get("target_label") == target
            pid, sid = folder_ids(self.folder)
            self.progress_label.setText(
                f"<b>Accepted in this folder</b>{sim}<br>"
                + " · ".join(f"stiffness {v}: {n}" for v, n in counts.items())
                + f"<br><small>Saved as participant {pid}, session {sid}.</small>")
        elif self.mode == PLAN_MODE and c.progress is not None:
            p = c.progress
            rows = ["<tr><th align='left'>condition</th><th>Low</th><th>Med</th><th>High</th></tr>"]
            for obj in c.plan.counts:
                cells = "".join(f"<td align='center'>{p.by_cell[(obj, t)][0]}/{p.by_cell[(obj, t)][1]}"
                                f"</td>" for t in TARGET_ORDER)
                rows.append(f"<tr><td>{obj}</td>{cells}</tr>")
            tasks = ""                               # one line per task condition in the plan
            for cond, task in TASKS.items():
                got = [p.by_cell.get((cond, t)) for t in TARGET_ORDER]
                if all(got):
                    tasks += f"{task.short}: " + " · ".join(
                        f"{TARGETS[t]['display']} {a}/{n}"
                        for t, (a, n) in zip(TARGET_ORDER, got)) \
                        + f" · total {sum(a for a, _ in got)}/{sum(n for _, n in got)}<br>"
            self.progress_label.setText(
                f"<b>Accepted development trials: {p.accepted} / {p.total_planned}</b>{sim}"
                f"<table cellspacing='4'>{''.join(rows)}</table>" + tasks
                + (("movement (grasp/fist): " if tasks else "movement: ")
                   + " · ".join(f"{MOTION_SHORT[m]} {a}/{n}" for m, (a, n)
                                in p.by_motion.items()) + "<br>"
                   if p.by_motion else "plan v1: no movement assigned (recorded stationary)<br>")
                + f"attempted {p.attempted} · rejected {p.rejected} · invalid {p.invalid} · "
                f"unreviewed {p.unreviewed} · remaining {p.remaining}")
        else:
            counts = c.role_counts()
            parts = [f"{role}: " + ", ".join(f"{n} {cat}" for cat, n in sorted(cats.items()))
                     for role, cats in sorted(counts.items())]
            self.progress_label.setText(
                "<b>Attempts</b>" + sim + "<br>" + ("<br>".join(parts) if parts else "none yet")
                + "<br><small>Practice and held-out tests never count toward the plan."
                  "</small>")

    def _message(self, text: str, error: bool = False):
        color = "#c62828" if error else "#263238"
        text = text.replace("_", "_\u200b").replace("/", "/\u200b")   # let long names wrap
        self.message_label.setText(f"<span style='color:{color}'>{text}</span>" if text else "")
