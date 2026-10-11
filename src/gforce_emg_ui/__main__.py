"""Entry point: python -m gforce_emg_ui [--simulate]"""

import argparse
import signal
import sys
from pathlib import Path

import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from .config import DEFAULT_EMG_RATE, EMG_RATES, PACKAGE_SAMPLE_COUNT
from .hub import DataHub


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gforce_emg_ui", description=__doc__)
    ap.add_argument("--simulate", action="store_true", help="use synthetic data instead of the band")
    ap.add_argument("--emg-rate", type=int, choices=EMG_RATES, default=DEFAULT_EMG_RATE)
    ap.add_argument("--package-samples", type=int, default=PACKAGE_SAMPLE_COUNT,
                    help="samples per BLE data callback (lower = less latency)")
    ap.add_argument("--data-dir", type=Path, default=Path.cwd() / "data",
                    help="where recordings are saved")
    args = ap.parse_args(argv)

    app = QtWidgets.QApplication(sys.argv[:1])
    pg.setConfigOptions(antialias=False, background="w", foreground="k")

    hub = DataHub()
    if args.simulate:
        from .sources import SimSource
        source = SimSource(hub, args.emg_rate, args.package_samples)
    else:
        from .sources import SdkSource
        source = SdkSource(hub, args.emg_rate, args.package_samples)

    from .ui.main_window import MainWindow
    win = MainWindow(hub, source, args.data_dir)
    win.show()

    # Let Ctrl-C in the terminal close the app
    signal.signal(signal.SIGINT, lambda *_: win.close())
    keepalive = QtCore.QTimer()
    keepalive.timeout.connect(lambda: None)
    keepalive.start(250)

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
