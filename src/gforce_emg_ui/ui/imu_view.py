"""IMU dashboard: accelerometer and gyroscope (gForceUltra: 6-axis, 50 Hz)."""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets

from ..hub import DataHub

AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")
WINDOW_S = 5


class ImuDashboard(QtWidgets.QWidget):
    def __init__(self, hub: DataHub, parent=None):
        super().__init__(parent)
        self.hub = hub
        self._view_version = 0
        self._drawn_key = None
        self.redraw_count = 0
        self.view = pg.GraphicsLayoutWidget()
        self.curves = []
        specs = (("Accelerometer", "ACC", 0), ("Gyroscope", "GYRO", 3))
        self.plots = []
        for row, (title, short, offset) in enumerate(specs):
            p = self.view.addPlot(row=row, col=0, title=title)
            p.setMenuEnabled(False)
            p.showGrid(x=True, y=True, alpha=0.2)
            p.addLegend(offset=(10, 10))
            p.setLabel("bottom", "Time (s)")
            p.getAxis("bottom").enableAutoSIPrefix(False)
            for i, axis in enumerate("XYZ"):
                self.curves.append(p.plot(pen=pg.mkPen(AXIS_COLORS[i], width=1.5),
                                          name=f"{short}-{axis}"))
            if self.plots:
                p.setXLink(self.plots[0])
            self.plots.append(p)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.view)

    def invalidate(self, *_):
        """A view setting changed: redraw on the next refresh even without new data."""
        self._view_version += 1

    def refresh(self) -> bool:
        """Redraw if data or view settings changed. Returns True if it drew."""
        key = (self.hub.imu.version, self._view_version)
        if key == self._drawn_key:
            return False
        rate = self.hub.imu_rate or 50
        data = self.hub.imu.latest(int(WINDOW_S * rate))
        if data.shape[1] < 2:
            return False
        self._drawn_key = key
        self.redraw_count += 1
        x = (np.arange(data.shape[1]) - (data.shape[1] - 1)) / rate
        for i, curve in enumerate(self.curves):
            curve.setData(x, data[i], connect="finite")
        self.plots[0].setXRange(-WINDOW_S, 0, padding=0)
        return True
