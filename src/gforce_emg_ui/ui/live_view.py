"""Live Classifier tab: the frozen model's live prediction s = 0 / 0.5 / 1, or s = --.

Display-only. A thin Qt layer over live.engine.LiveEngine: inference and the freshness
watchdog run on their own threads whether or not this tab is visible; refresh() (GUI
thread, only while visible) reads the latest-result snapshot and redraws what changed.
The model is loaded only on request, after a trust confirmation, on a background thread.
"""

import collections
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from .. import __version__
from ..collection.storage import git_info
from ..live import compat
from ..live.engine import (ERROR, INVALID_INPUT, NO_MODEL, READY, RUNNING, STALE, STOPPED,
                           WARMING_UP, LiveEngine, LiveSettings)
from ..live.log import LogRootError, PredictionLog, check_root
from ..live.model import TRUST_NOTE, ModelError, classifier_status, load_trusted_model
from ..sources import DISCONNECTING, FAILED, IDLE, RECONNECTING, STREAMING
from .collection_panel import LiveText

NOTE = ("Instructed-effort prediction, not measured stiffness.\n"
        "Low was trained as minimal effort while maintaining the grasp/fist;\n"
        "fully relaxed/open-hand behavior is not yet validated.")
HISTORY_S = 20.0
STATE_COLORS = {RUNNING: "#2e7d32", WARMING_UP: "#1565c0", STALE: "#ef6c00",
                INVALID_INPUT: "#c62828", ERROR: "#c62828", STOPPED: "#757575",
                READY: "#757575", NO_MODEL: "#757575"}
LEVEL_TAGS = {compat.OK: ("#2e7d32", "ok"), compat.WARN: ("#ef6c00", "warn"),
              compat.UNKNOWN: ("#ef6c00", "unknown"), compat.BLOCK: ("#c62828", "BLOCK")}
BAR_COLORS = ("#86b6ef", "#2a78d6", "#104281")   # ordinal blue ramp: low -> high
S_COLOR, SOFT_COLOR = "#2a78d6", "#eb6834"


def _set_enabled(w, on: bool):
    if w.isEnabled() != on:
        w.setEnabled(on)


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "on" if v else "off"
    return "not reported" if v is None else str(v)


def _fmt_ms(t: dict) -> str:
    return "--" if t["p50"] is None else f"p50 {t['p50']:.2f} / p95 {t['p95']:.2f} ms (n={t['n']})"


class ProbBars(QtWidgets.QWidget):
    """Three labelled probability bars; paints itself (no relayout on updates)."""

    ROW = 26

    def __init__(self, labels=("low", "medium", "high"), parent=None):
        super().__init__(parent)
        self.labels = tuple(labels)
        self._p = None
        self.setFixedHeight(self.ROW * len(self.labels) + 8)

    def sizeHint(self):
        return QtCore.QSize(360, self.height())

    def values(self):
        return self._p

    def set_probabilities(self, probs: Optional[dict]):
        vals = None if probs is None else tuple(round(float(probs[l]), 4) for l in self.labels)
        if vals != self._p:
            self._p = vals
            self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w = self.width()
        x0, x1 = 110, max(150, w - 80)
        top = int(np.argmax(self._p)) if self._p else None
        for i, label in enumerate(self.labels):
            y = 4 + i * self.ROW
            f = p.font()
            f.setBold(i == top)
            p.setFont(f)
            p.setPen(QtGui.QColor("#0b0b0b"))
            p.drawText(QtCore.QRectF(0, y, x0 - 8, self.ROW - 4),
                       QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, f"p({label})")
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor("#e1e0d9"))
            p.drawRoundedRect(QtCore.QRectF(x0, y + 4, x1 - x0, self.ROW - 12), 3, 3)
            text = "--"
            if self._p is not None:
                v = self._p[i]
                p.setBrush(QtGui.QColor(BAR_COLORS[i % len(BAR_COLORS)]))
                p.drawRoundedRect(QtCore.QRectF(x0, y + 4, max(0.0, (x1 - x0) * v),
                                                self.ROW - 12), 3, 3)
                text = f"{100 * v:.1f} %"
            p.setPen(QtGui.QColor("#0b0b0b"))
            p.drawText(QtCore.QRectF(x1 + 6, y, w - x1 - 6, self.ROW - 4),
                       QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, text)


class LiveClassifierDashboard(QtWidgets.QWidget):
    model_loaded = QtCore.pyqtSignal(object, int)       # (LoadedModel | ModelError, token)

    def __init__(self, hub, source, parent=None, settings: Optional[LiveSettings] = None,
                 log_root: Optional[Path] = None, clock=time.monotonic):
        super().__init__(parent)
        self.hub = hub
        self.source = source
        self._clock = clock
        self.engine = LiveEngine(hub, settings, clock=clock)
        self.model = None
        self.report: Optional[compat.CompatReport] = None
        self.load_error = ""
        self.log_root = Path(log_root) if log_root else Path.cwd() / "live_logs"
        self.log: Optional[PredictionLog] = None
        self._device, self._params = {}, {}
        self._loading: Optional[str] = None
        self._load_token = 0
        self._closed = False
        self._run_fingerprint = None
        self._ack_key = None
        self._shown = None
        self._applied_id = None
        self._hist_version = None
        self._hist_t = self._diag_t = -1e9
        self._t_gui = collections.deque(maxlen=500)
        self.start_error = ""
        self.redraw_count = 0
        self._build()
        self.model_loaded.connect(self._on_model_loaded)
        source.connected.connect(self._on_connected)
        source.params.connect(self._on_params)
        source.disconnected.connect(self._on_disconnected)
        source.state_changed.connect(self._on_state)
        ok, msg = classifier_status()
        self._show_model(msg if not ok else "")
        self._recheck()

    # -- layout -------------------------------------------------------------------
    def _build(self):
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 4, 0)
        scroll = QtWidgets.QScrollArea()             # long texts never squeeze each other
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setFixedWidth(440)

        box = QtWidgets.QGroupBox("Model")
        bl = QtWidgets.QVBoxLayout(box)
        self.btn_load = QtWidgets.QPushButton("Load model…")
        self.btn_load.clicked.connect(self._choose_model)
        bl.addWidget(self.btn_load)
        self.model_label = QtWidgets.QLabel()
        self.model_label.setWordWrap(True)
        self.model_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        bl.addWidget(self.model_label)
        lv.addWidget(box)

        box = QtWidgets.QGroupBox("Run")
        bl = QtWidgets.QVBoxLayout(box)
        row = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_reset = QtWidgets.QPushButton("Reset buffer")
        for b in (self.btn_start, self.btn_stop, self.btn_reset):
            row.addWidget(b)
        bl.addLayout(row)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(lambda: self.stop("user_stop"))
        self.btn_reset.clicked.connect(lambda: self.engine.reset("user_reset"))
        self.blocker_label = LiveText(lines=2, pad=4)
        bl.addWidget(self.blocker_label)
        self.log_check = QtWidgets.QCheckBox("Log predictions (new folder per run)")
        bl.addWidget(self.log_check)
        row = QtWidgets.QHBoxLayout()
        self.btn_log_dir = QtWidgets.QPushButton("Log folder…")
        self.btn_log_dir.clicked.connect(self._choose_log_root)
        row.addWidget(self.btn_log_dir)
        row.addStretch(1)
        bl.addLayout(row)
        self.log_dir_label = QtWidgets.QLabel(str(self.log_root))
        self.log_dir_label.setWordWrap(True)
        bl.addWidget(self.log_dir_label)
        lv.addWidget(box)

        box = QtWidgets.QGroupBox("Compatibility")
        bl = QtWidgets.QVBoxLayout(box)
        self.compat_label = QtWidgets.QLabel()
        self.compat_label.setWordWrap(True)
        bl.addWidget(self.compat_label)
        self.ack_check = QtWidgets.QCheckBox("I reviewed the warnings above (display-only test)")
        self.ack_check.toggled.connect(self._update_controls)
        bl.addWidget(self.ack_check)
        lv.addWidget(box)

        lv.addStretch(1)

        right = QtWidgets.QVBoxLayout()
        self.sim_label = QtWidgets.QLabel(
            "<b>SIMULATOR</b>: synthetic EMG. Software test only; predictions are meaningless.")
        self.sim_label.setStyleSheet("background:#fff3e0; color:#b71c1c; padding:4px;")
        self.sim_label.setVisible(bool(getattr(self.source, "is_simulation", False)))
        right.addWidget(self.sim_label)
        self.s_label = LiveText(lines=1, px=88, bold=True, pad=12)
        self.class_label = LiveText(lines=1, px=28, bold=True, pad=6)
        self.state_label = LiveText(lines=1, px=15, bold=True, pad=10)
        self.bars = ProbBars()
        self.soft_label = LiveText(lines=2, px=13, pad=6)
        head = QtWidgets.QHBoxLayout()
        col = QtWidgets.QVBoxLayout()
        for w in (self.s_label, self.class_label):
            col.addWidget(w)
        head.addLayout(col, 1)
        col = QtWidgets.QVBoxLayout()
        for w in (self.bars, self.soft_label):
            col.addWidget(w)
        col.addStretch(1)
        head.addLayout(col, 1)
        right.addLayout(head)
        right.addWidget(self.state_label)

        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.plot.showGrid(y=True, alpha=0.25)
        self.plot.setYRange(-0.1, 1.1, padding=0)
        self.plot.setXRange(-HISTORY_S, 0, padding=0)
        self.plot.getAxis("left").setTicks([[(0, "0 low"), (0.5, "0.5 medium"), (1, "1 high")]])
        self.plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot.setLabel("bottom", "host receipt time of the newest window sample (s, 0 = now)")
        self.plot.setTitle("History (raw model output; gaps = no valid prediction)")
        self.plot.addLegend(offset=(10, 10))
        self.s_curve = self.plot.plot(pen=pg.mkPen(S_COLOR, width=2), name="s (argmax class)")
        self.s_dots = pg.ScatterPlotItem(size=5, pen=None, brush=pg.mkBrush(S_COLOR))
        self.plot.addItem(self.s_dots)
        self.soft_curve = self.plot.plot(
            pen=pg.mkPen(SOFT_COLOR, width=2, style=QtCore.Qt.DashLine),
            name="soft class score (not stiffness)")
        right.addWidget(self.plot, 1)
        box = QtWidgets.QGroupBox("Diagnostics (host-side)")
        bl = QtWidgets.QVBoxLayout(box)
        self.diag_label = LiveText(lines=11, px=11, pad=6)
        bl.addWidget(self.diag_label)
        right.addWidget(box)
        note = QtWidgets.QLabel(NOTE)
        note.setStyleSheet("color:#52514e; padding:4px; border-top:1px solid #e1e0d9;")
        right.addWidget(note)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(scroll)
        layout.addLayout(right, 1)
        self._show_prediction(None, NO_MODEL, "", None)

    # -- model --------------------------------------------------------------------
    def _choose_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load trusted model", str(Path.cwd()),
            "emg-classifier model (*.joblib);;All files (*)")
        if not path:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Load a trusted model?", f"{TRUST_NOTE}\n\n{path}\n\nLoad this file?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if answer == QtWidgets.QMessageBox.Yes:
            self.load_model_file(path)

    def load_model_file(self, path):
        """Load in the background. The caller has confirmed the file is trusted."""
        self.engine.set_model(None)              # stops and invalidates any current run
        self.model, self.report, self.load_error = None, None, ""
        self.ack_check.setChecked(False)
        self._load_token += 1
        self._loading = str(path)
        self._show_model()
        self._recheck()
        threading.Thread(target=self._load_worker, args=(str(path), self._load_token),
                         name="live-model-load", daemon=True).start()

    def _load_worker(self, path: str, token: int):
        try:
            result = load_trusted_model(path)
        except ModelError as e:
            result = e
        except Exception as e:                   # never crash the app on a bad file
            result = ModelError(f"unexpected error loading {path}: {type(e).__name__}: {e}")
        try:
            self.model_loaded.emit(result, token)
        except RuntimeError:                     # window already destroyed
            pass

    @property
    def loading(self) -> bool:
        return self._loading is not None

    def _on_model_loaded(self, result, token: int):
        if token != self._load_token or self._closed:
            return
        self._loading = None
        if isinstance(result, Exception):
            self.load_error = str(result)
        else:
            try:
                self.engine.set_model(result)
                self.model = result
            except ValueError as e:
                self.load_error = str(e)
        self._show_model()
        self._recheck()

    def _show_model(self, missing: str = ""):
        if self._loading:
            html = f"Loading <code>{self._loading}</code> …"
        elif self.model is not None:
            m = self.model
            warn = "".join(f"<br><span style='color:#ef6c00'>⚠ {w}</span>" for w in m.warnings)
            html = (f"<b>{m.name}</b> · sha256 {m.sha256[:16]}<br><code>{m.path}</code><br>"
                    f"window {m.window_samples} samples / {m.hop_samples}-sample hop · "
                    f"{len(m.channel_names)} × {m.feature.get('name')} "
                    f"v{m.feature.get('version')} ({', '.join(m.channel_names[:1])}…"
                    f"{m.channel_names[-1]})<br>"
                    f"classes {', '.join(f'{c}={l} (s={s:g})' for c, l, s in m.classes)}<br>"
                    f"fitted on {m.training.get('n_trials')} trials / "
                    f"{m.training.get('n_windows')} windows · scikit-learn "
                    f"{m.installed_versions.get('scikit-learn')}<br>"
                    f"<small>Development CV (not a live guarantee): window BA "
                    f"{m.cv_reference.get('window_balanced_accuracy_trial_weighted', float('nan')):.3f}"
                    f", trial BA {m.cv_reference.get('trial_balanced_accuracy', float('nan')):.3f}"
                    f"</small>{warn}")
        elif self.load_error or missing:
            html = f"<span style='color:#c62828'>{self.load_error or missing}</span>"
        else:
            html = ("No model loaded. Load a trusted local <code>model.joblib</code> "
                    "(nothing is loaded automatically).")
        if self.model_label.text() != html:
            self.model_label.setText(html)

    # -- source / compatibility -------------------------------------------------------
    def _on_connected(self, summary: dict):
        self._device = dict(summary or {})
        self._recheck()

    def _on_params(self, params: dict):
        self._params = dict(params or {})
        self._recheck()

    def _on_disconnected(self):
        self._device, self._params = {}, {}
        self._recheck()

    def _on_state(self, _signalled=""):
        state = self.source.state                # queued signals may arrive out of order
        if self.engine.running:
            if state in (IDLE, FAILED, DISCONNECTING):
                self.stop("disconnected")
            elif state == RECONNECTING:
                self.engine.reset("link_lost")
        self._recheck()

    def _recheck(self):
        if self.model is None:
            self.report = None
        else:
            live = compat.live_acquisition(self._device, self._params, self.source, self.hub)
            self.report = compat.check_compatibility(
                self.model.acquisition, live, bool(getattr(self.source, "is_simulation", False)))
            if self.engine.running and self.report.fingerprint != self._run_fingerprint:
                changed = sorted(set(self.report.fingerprint) ^ set(self._run_fingerprint or ()))
                fields = sorted({k for k, _ in changed})
                self.stop(f"configuration changed ({', '.join(fields)}): review compatibility "
                          f"and Start again")
        key = None if self.report is None else (self.report.fingerprint, tuple(
            (c.field, c.level, repr(c.live)) for c in self.report.needs_ack))
        if key != self._ack_key:
            self._ack_key = key
            self.ack_check.setChecked(False)     # any live change needs a new review
        self._show_compat()
        self._update_controls()

    def _show_compat(self):
        r = self.report
        if r is None:
            html = "Load a model to compare its saved acquisition setup with the live one."
        else:
            bad = [c for c in r.checks if c.level != compat.OK]
            n_ok = len(r.checks) - len(bad)
            lines = [f"Model configuration <code>{r.config_key}</code>: {n_ok} checks ok"]
            for c in bad:
                color, tag = LEVEL_TAGS[c.level]
                lines.append(f"<span style='color:{color}'><b>{tag}</b></span> {c.label}: "
                             f"model {_fmt(c.expected)}, live {_fmt(c.live)}"
                             + (f" <small>({c.note})</small>" if c.note else ""))
            lines += [f"<small>{t}</small>" for t in r.info]
            html = "<br>".join(lines)
        if self.compat_label.text() != html:
            self.compat_label.setText(html)

    def start_blockers(self) -> list:
        out = []
        if self.loading:
            out.append("model loading")
        elif self.model is None:
            out.append("load a model")
        if self.source.state != STREAMING:
            out.append("connect the band (streaming)")
        r = self.report
        if r is not None and r.blocking:
            out.append(f"incompatible settings ({len(r.blocking)}, see Compatibility)")
        elif r is not None and r.needs_ack and not self.ack_check.isChecked():
            out.append("acknowledge the warnings")
        if self.engine.running:
            out.append("running")
        return out

    def _update_controls(self, *_):
        running = self.engine.running
        r = self.report
        _set_enabled(self.btn_start, not self.start_blockers())
        _set_enabled(self.btn_stop, running)
        _set_enabled(self.btn_reset, running)
        _set_enabled(self.btn_load, not self.loading)
        _set_enabled(self.ack_check, r is not None and not r.blocking and bool(r.needs_ack)
                     and not running)
        _set_enabled(self.log_check, not running)
        _set_enabled(self.btn_log_dir, not running)
        blockers = [b for b in self.start_blockers() if b != "running"]
        text = (f"<span style='color:#c62828'>{self.start_error}</span>" if self.start_error
                else "" if running or not blockers else "Start needs: " + " · ".join(blockers))
        self.blocker_label.setText(f"<div align='left'>{text}</div>" if text else "")

    # -- run control ------------------------------------------------------------------
    def _choose_log_root(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Prediction log folder",
                                                       str(self.log_root))
        if d:
            self.set_log_root(d)

    def set_log_root(self, path):
        self.log_root = Path(path)
        self.log_dir_label.setText(str(self.log_root))
        self.start_error = ""
        self._update_controls()

    def start(self) -> bool:
        if self.start_blockers():
            return False
        m, r = self.model, self.report
        sim = bool(getattr(self.source, "is_simulation", False))
        run_id = uuid.uuid4().hex[:10]
        meta = {
            "kind": "gforce_emg_ui.live_classifier_run", "run_id": run_id,
            "started": datetime.now().isoformat(timespec="milliseconds"),
            "source": {"type": "simulation" if sim else "real_band",
                       "device": self._device, "params": self._params,
                       "backend": self.source.backend_name()},
            "app": {"version": __version__, "git": git_info()},
            "model": m.identity(), "compatibility": r.to_dict(),
            "acknowledged": [c.field for c in r.needs_ack],
            "settings": asdict(self.engine.settings),
            "window": {"samples": m.window_samples, "hop_samples": m.hop_samples,
                       "units": "samples"},
            "output": "raw model output: no smoothing, voting, rest gate or thresholds",
            "labels": "none: no ground truth is recorded or inferred",
            "note": NOTE.replace("\n", " "),
        }
        self.log = None
        if self.log_check.isChecked():
            try:
                root = check_root(self.log_root, m.path)
            except LogRootError as e:
                self.start_error = f"Log folder refused: {e}"
                self._update_controls()
                return False
            self.log = PredictionLog(root, run_id, meta, sim)
        self.start_error = ""
        self._run_fingerprint = r.fingerprint
        self._t_gui.clear()
        self.engine.start(meta, self.log, run_id=run_id)
        self._update_controls()
        return True

    def stop(self, reason: str = "user_stop"):
        self.engine.stop(reason)                 # also closes the prediction log
        self._update_controls()

    def shutdown(self):
        """App close: stop classifying, let the threads exit, ignore late model loads."""
        self._closed = True
        self.engine.shutdown()

    # -- refresh (GUI thread, visible tab only) ------------------------------------------
    def refresh(self) -> bool:
        """Redraw what changed. Returns True if anything was drawn."""
        if not self.isVisible():
            return False
        now = self._clock()
        snap = self.engine.snapshot()
        cur = snap.current(now)
        state = STALE if snap.state == RUNNING and cur is None else snap.state
        drew = False
        key = (snap.version, cur.pred_id if cur else None, state, self.loading)
        if key != self._shown:
            self._shown = key
            self._show_prediction(cur, state, snap.reason, snap)
            if cur is not None and cur.pred_id != self._applied_id:
                self._applied_id = cur.pred_id
                ms = cur.age(now) * 1000
                self._t_gui.append(ms)
                self.engine.log_display(cur.pred_id, now, ms)
            drew = True
        if (snap.version != self._hist_version or self.engine.running) \
                and now - self._hist_t >= 0.1:
            self._hist_version, self._hist_t = snap.version, now
            self._draw_history(now)
            drew = True
        if now - self._diag_t >= 0.25:
            self._diag_t = now
            self._show_diagnostics(now, snap, cur)
            self._update_controls()
            drew = True
        if drew:
            self.redraw_count += 1
        return drew

    def _show_prediction(self, cur, state: str, reason: str, snap):
        p = cur.prediction if cur is not None else None
        self.s_label.setText(f"s = {p.s_discrete:g}" if p is not None else "s = --")
        self.class_label.setText(p.label.upper() if p is not None else "--")
        label = "LOADING MODEL" if self.loading else state.replace("_", " ")
        if state == WARMING_UP and snap is not None:
            label += f" · {snap.warm[0]}/{snap.warm[1]} samples"
        if reason and state != RUNNING:
            label += f" · {reason}"
        self.state_label.setText(label)
        self.state_label.setColors(STATE_COLORS.get(state, "#757575"), "white")
        self.bars.set_probabilities(p.probabilities if p is not None else None)
        self.soft_label.setText(
            f"soft class score {p.s_expected:.2f}<br><small>0·p_low + 0.5·p_medium + 1·p_high; "
            f"not stiffness, not a confidence</small>" if p is not None
            else "soft class score --")

    def _draw_history(self, now: float):
        h = self.engine.history()
        if len(h):
            h = h[h[:, 0] >= now - HISTORY_S - 1]
        if len(h) == 0:
            for c in (self.s_curve, self.soft_curve):
                c.setData([], [])
            self.s_dots.setData([], [])
            return
        t = h[:, 0] - now
        s, soft = h[:, 1], h[:, 2]
        x2, y2 = np.repeat(t, 2)[1:], np.repeat(s, 2)[:-1]      # right steps, NaN = gap
        self.s_curve.setData(x2, y2, connect="finite")
        fin = np.isfinite(s)
        self.s_dots.setData(t[fin], s[fin])
        self.soft_curve.setData(t, soft, connect="finite")

    def _show_diagnostics(self, now: float, snap, cur):
        m, c = self.model, snap.counters
        t = self.engine.timings()
        gui = list(self._t_gui)
        gui_t = {"p50": float(np.percentile(gui, 50)) if gui else None,
                 "p95": float(np.percentile(gui, 95)) if gui else None, "n": len(gui)}
        thr = self.hub.emg_actual_rate()
        age = (f"{snap.result.age(now):.2f} s" if snap.result is not None else "--")
        resets = ", ".join(f"{k} {v}" for k, v in sorted((c.get("resets") or {}).items())) or "0"
        log = self.log
        if log is None:
            log_line = "Prediction log: off"
        else:
            log_line = (f"Log: {log.folder.name} · rows {sum(log.written.values())} · dropped "
                        f"{log.dropped}" + (f" · <b>error {log.error}</b>" if log.error else ""))
        lines = [
            (f"Model window {m.window_samples} samples / {m.hop_samples}-sample hop"
             if m else "Model window: --"),
            f"Host receipt throughput {thr:.1f} rows/s · nominal {self.hub.emg_rate:g} Hz "
            f"(neither is a verified acquisition rate)",
            f"Inference (features + model), per batch call: {_fmt_ms(t['inference_batch_ms'])}",
            f"Newest-batch receipt → result ready: {_fmt_ms(t['receipt_to_result_ms'])}",
            f"Newest-batch receipt → GUI applied: {_fmt_ms(gui_t)}",
            f"Current result age {age} · stale after {snap.stale_s:.2f} s",
            f"Predictions {c.get('predictions', 0)}: valid {c.get('valid_results', 0)} · "
            f"invalid {c.get('invalid_results', 0)} · expired {c.get('expired_before_publish', 0)}"
            f" · skipped stale windows {c.get('skipped_stale_windows', 0)}",
            f"Dropped input blocks {c.get('dropped_blocks', 0)} ({c.get('dropped_samples', 0)} "
            f"samples) · old-generation blocks {c.get('discarded_blocks', 0)} · rejected samples "
            f"{c.get('rejected_samples', 0)}",
            f"Resets: {resets} · stale events {c.get('stale_events', 0)} · errors "
            f"{c.get('errors', 0)}",
            log_line,
            "<small>Host-side only: not contraction-to-display, sensor→host or "
            "display-onset latency.</small>",
        ]
        self.diag_label.setText("<div align='left'>" + "<br>".join(lines) + "</div>")
