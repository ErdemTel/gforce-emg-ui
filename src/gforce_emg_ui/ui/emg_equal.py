"""EMG dashboard 1: all channels on the SAME y-scale + per-channel RMS bars,
to compare which channel fires most during a gesture."""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from ..config import CHANNEL_COLORS, EMG_CHANNELS
from ..contact import contact_color, format_kohm
from ..hub import DataHub


class EqualScaleDashboard(QtWidgets.QWidget):
    def __init__(self, hub: DataHub, parent=None):
        super().__init__(parent)
        self.hub = hub
        self._x_cache = {}
        self._yrange = 50.0
        self._rms_peak = 10.0

        # -- toolbar --
        self.window_combo = QtWidgets.QComboBox()
        for s in (1, 2, 5):
            self.window_combo.addItem(f"{s} s", s)
        self.window_combo.setCurrentIndex(1)
        self.rms_combo = QtWidgets.QComboBox()
        for ms in (50, 100, 200, 500):
            self.rms_combo.addItem(f"{ms} ms", ms)
        self.rms_combo.setCurrentIndex(2)
        self.dc_check = QtWidgets.QCheckBox("Remove DC offset")
        self.dc_check.setChecked(True)
        self.auto_check = QtWidgets.QCheckBox("Auto shared scale")
        self.auto_check.setChecked(True)
        self.range_spin = QtWidgets.QDoubleSpinBox()
        self.range_spin.setRange(1, 1e6)
        self.range_spin.setDecimals(0)
        self.range_spin.setValue(500)
        self.range_spin.setPrefix("± ")
        self.range_spin.setEnabled(False)
        self.auto_check.toggled.connect(lambda on: self.range_spin.setEnabled(not on))

        bar = QtWidgets.QHBoxLayout()
        for w in (QtWidgets.QLabel("Window"), self.window_combo,
                  QtWidgets.QLabel("RMS window"), self.rms_combo,
                  self.dc_check, self.auto_check, self.range_spin):
            bar.addWidget(w)
        bar.addStretch(1)

        # -- stacked waveforms, identical y-range --
        self.wave = pg.GraphicsLayoutWidget()
        self.plots, self.curves, self.labels = [], [], []
        for ch in range(EMG_CHANNELS):
            self.labels.append(self.wave.addLabel("", row=ch, col=0))
            p = self.wave.addPlot(row=ch, col=1)
            p.setMenuEnabled(False)
            p.setMouseEnabled(x=False, y=False)
            p.hideButtons()
            p.showGrid(x=True, y=True, alpha=0.2)
            p.setClipToView(True)
            p.setDownsampling(auto=True, mode="peak")
            p.getAxis("left").setWidth(55)
            p.getAxis("bottom").enableAutoSIPrefix(False)
            if ch < EMG_CHANNELS - 1:
                p.getAxis("bottom").setStyle(showValues=False)
            else:
                p.setLabel("bottom", "Time (s)")
            if self.plots:
                p.setXLink(self.plots[0])
            self.curves.append(p.plot(pen=pg.mkPen(CHANNEL_COLORS[ch], width=1)))
            self.plots.append(p)
        self.wave.ci.layout.setColumnStretchFactor(1, 1)
        self._set_labels(np.full(EMG_CHANNELS, np.nan))

        # -- RMS bars on one common axis --
        self.rms_plot = pg.PlotWidget(title="RMS per channel (same scale)")
        self.rms_plot.setMenuEnabled(False)
        self.rms_plot.setMouseEnabled(x=False, y=False)
        self.rms_plot.hideButtons()
        self.rms_plot.showGrid(y=True, alpha=0.3)
        ax = self.rms_plot.getAxis("bottom")
        ax.setTicks([[(i, f"CH{i + 1}") for i in range(EMG_CHANNELS)]])
        self.rms_plot.setXRange(-0.6, EMG_CHANNELS - 0.4, padding=0)
        self._brushes = [pg.mkBrush(c) for c in CHANNEL_COLORS]
        self.bars = pg.BarGraphItem(x=np.arange(EMG_CHANNELS), height=np.zeros(EMG_CHANNELS),
                                    width=0.7, brushes=self._brushes)
        self.rms_plot.addItem(self.bars)
        self.bar_texts = []
        for i in range(EMG_CHANNELS):
            t = pg.TextItem("", anchor=(0.5, 1.0), color="k")
            t.setPos(i, 0)
            self.rms_plot.addItem(t)
            self.bar_texts.append(t)
        self.top_label = QtWidgets.QLabel("Strongest: --")
        self.top_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.top_label)
        right.addWidget(self.rms_plot, 1)
        right_w = QtWidgets.QWidget()
        right_w.setLayout(right)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self.wave)
        split.addWidget(right_w)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(bar)
        layout.addWidget(split, 1)

    def _x(self, n, rate):
        key = (n, rate)
        x = self._x_cache.get(key)
        if x is None:
            x = (np.arange(n) - (n - 1)) / rate
            self._x_cache = {key: x}
        return x

    def _set_labels(self, impedance):
        for ch, label in enumerate(self.labels):
            label.setText(f"<b>CH{ch + 1}</b><br>{format_kohm(impedance[ch])}",
                          color=contact_color(impedance[ch]), size="10pt")

    def update_contact(self, impedance):
        self._set_labels(impedance)

    def refresh(self):
        rate = self.hub.emg_rate
        n = int(self.window_combo.currentData() * rate)
        data = self.hub.emg.latest(n)
        if data.shape[1] < 2:
            return
        if self.dc_check.isChecked():
            data = data - data.mean(axis=1, keepdims=True)
        x = self._x(data.shape[1], rate)
        for ch, curve in enumerate(self.curves):
            curve.setData(x, data[ch], skipFiniteCheck=True)

        # Shared y-range: fast attack, slow release so the scale is stable
        if self.auto_check.isChecked():
            target = max(float(np.abs(data).max()) * 1.05, 1e-3)
            if target > self._yrange:
                self._yrange = target
            else:
                self._yrange = 0.97 * self._yrange + 0.03 * target
            yr = self._yrange
        else:
            yr = self.range_spin.value()
        self.plots[0].setXRange(x[0], 0, padding=0)
        for p in self.plots:
            p.setYRange(-yr, yr, padding=0)

        m = max(2, int(self.rms_combo.currentData() * rate / 1000))
        seg = data[:, -m:]
        if not self.dc_check.isChecked():
            seg = seg - seg.mean(axis=1, keepdims=True)
        rms = np.sqrt(np.mean(seg.astype(np.float64) ** 2, axis=1))
        top = int(np.argmax(rms))
        brushes = list(self._brushes)
        brushes[top] = pg.mkBrush("#000000")
        self.bars.setOpts(height=rms, brushes=brushes)
        peak = float(rms.max()) * 1.2
        self._rms_peak = max(peak, 0.98 * self._rms_peak, 1e-3)
        self.rms_plot.setYRange(0, self._rms_peak, padding=0)
        total = float(rms.sum()) or 1.0
        for i, t in enumerate(self.bar_texts):
            t.setText(f"{rms[i]:.0f}\n{100 * rms[i] / total:.0f}%")
            t.setPos(i, rms[i])
        self.top_label.setText(f"Strongest: CH{top + 1}  ({rms[top]:.0f} RMS)")
