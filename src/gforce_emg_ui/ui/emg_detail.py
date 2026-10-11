"""EMG dashboard 2, modelled on the vendor pyqt_demo: per channel an FFT
spectrum (left) and an individually auto-scaled waveform (right), with the
channel name + impedance colored by electrode contact quality."""

import time

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets

from ..config import CHANNEL_COLORS, EMG_CHANNELS, FFT_INTERVAL_S
from ..contact import contact_color, format_kohm
from ..hub import DataHub


class DetailDashboard(QtWidgets.QWidget):
    def __init__(self, hub: DataHub, parent=None):
        super().__init__(parent)
        self.hub = hub
        self._last_fft = 0.0
        self._x_cache = {}

        self.window_combo = QtWidgets.QComboBox()
        for s in (1, 2, 5):
            self.window_combo.addItem(f"{s} s", s)
        self.log_check = QtWidgets.QCheckBox("Log spectrum")
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Window"))
        bar.addWidget(self.window_combo)
        bar.addWidget(self.log_check)
        bar.addStretch(1)
        self.log_check.toggled.connect(lambda on: [p.setLogMode(y=on) for p in self.fft_plots])

        self.view = pg.GraphicsLayoutWidget()
        self.fft_plots, self.fft_curves = [], []
        self.wave_plots, self.wave_curves = [], []
        self.labels = []
        for ch in range(EMG_CHANNELS):
            pen = pg.mkPen(CHANNEL_COLORS[ch], width=1)
            label = self.view.addLabel("", row=ch, col=0)
            self.labels.append(label)
            f = self.view.addPlot(row=ch, col=1)
            w = self.view.addPlot(row=ch, col=2)
            for p in (f, w):
                p.setMenuEnabled(False)
                p.setMouseEnabled(x=False, y=False)
                p.hideButtons()
                p.showGrid(x=True, y=True, alpha=0.2)
                p.getAxis("left").setWidth(55)
                p.getAxis("bottom").enableAutoSIPrefix(False)
            w.setClipToView(True)
            w.setDownsampling(auto=True, mode="peak")
            if self.fft_plots:
                f.setXLink(self.fft_plots[0])
                w.setXLink(self.wave_plots[0])
            if ch < EMG_CHANNELS - 1:
                f.getAxis("bottom").setStyle(showValues=False)
                w.getAxis("bottom").setStyle(showValues=False)
            else:
                f.setLabel("bottom", "Frequency (Hz)")
                w.setLabel("bottom", "Time (s)")
            self.fft_curves.append(f.plot(pen=pen))
            self.wave_curves.append(w.plot(pen=pen))
            self.fft_plots.append(f)
            self.wave_plots.append(w)
        self.view.ci.layout.setColumnStretchFactor(1, 2)
        self.view.ci.layout.setColumnStretchFactor(2, 3)
        self.update_contact(np.full(EMG_CHANNELS, np.nan))

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(bar)
        layout.addWidget(self.view, 1)

    def update_contact(self, impedance):
        for ch, label in enumerate(self.labels):
            label.setText(f"<b>EMG-{ch + 1}</b><br>{format_kohm(impedance[ch])}",
                          color=contact_color(impedance[ch]), size="10pt")

    def refresh(self):
        rate = self.hub.emg_rate
        n = int(self.window_combo.currentData() * rate)
        data = self.hub.emg.latest(n)
        if data.shape[1] < 2:
            return
        key = (data.shape[1], rate)
        x = self._x_cache.get(key)
        if x is None:
            x = (np.arange(data.shape[1]) - (data.shape[1] - 1)) / rate
            self._x_cache = {key: x}
        mn = data.min(axis=1)
        mx = data.max(axis=1)
        for ch in range(EMG_CHANNELS):
            self.wave_curves[ch].setData(x, data[ch], skipFiniteCheck=True)
            lo, hi = float(mn[ch]), float(mx[ch])
            if lo == hi:
                lo, hi = lo - 1, hi + 1
            margin = max((hi - lo) * 0.1, 0.01)
            self.wave_plots[ch].setYRange(lo - margin, hi + margin, padding=0)
        self.wave_plots[0].setXRange(x[0], 0, padding=0)

        now = time.time()
        if now - self._last_fft >= FFT_INTERVAL_S:
            self._last_fft = now
            seg = data[:, -int(rate):].astype(np.float64)       # last 1 s
            seg = seg - seg.mean(axis=1, keepdims=True)
            win = np.hanning(seg.shape[1])
            mags = np.abs(np.fft.rfft(seg * win, axis=1)) / max(win.sum(), 1e-12) * 2
            freqs = np.fft.rfftfreq(seg.shape[1], d=1.0 / rate)
            if self.log_check.isChecked():
                mags = np.maximum(mags, 1e-6)
            for ch in range(EMG_CHANNELS):
                self.fft_curves[ch].setData(freqs, mags[ch], skipFiniteCheck=True)
            self.fft_plots[0].setXRange(0, rate / 2, padding=0)
