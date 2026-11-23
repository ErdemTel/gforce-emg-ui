"""EMG acquisition, recording and Sharpa robot controls in one window.

Normal launch opens the unified Robot control tab; Manual needs no armband or model.
Use --emg-producer or --legacy-robot-output only for compatibility workflows.
"""

import argparse
import signal
import sys
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from .config import DEFAULT_EMG_RATE, EMG_RATES, PACKAGE_SAMPLE_COUNT
from .hub import DataHub
from .live.robot_out import DEFAULT_PORT, parse_target


def parse_args(argv=None) -> argparse.Namespace:
    """Command line; invalid combinations exit via ap.error. Sets args.robot_target."""
    ap = argparse.ArgumentParser(prog="gforce_emg_ui", description=__doc__)
    ap.add_argument("--simulate", action="store_true", help="use synthetic data instead of the band")
    ap.add_argument("--emg-rate", type=int, choices=EMG_RATES, default=DEFAULT_EMG_RATE)
    ap.add_argument("--package-samples", type=int, default=PACKAGE_SAMPLE_COUNT,
                    help="samples per BLE data callback (lower = less latency)")
    ap.add_argument("--data-dir", type=Path, default=Path.cwd() / "data",
                    help="where recordings are saved")
    ap.add_argument("--sim-contact", choices=("default", "good"), default="default",
                    help="simulator only: electrode-contact scenario ('good' = all 8 "
                         "channels good)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--robot-control", metavar="HOST:PORT",
                      help="unified robot controls (default: --robot-udp target or 127.0.0.1:6731)")
    mode.add_argument("--legacy-robot-output", action="store_true",
                      help="use the previous integrated stiffness-bridge controls")
    ap.add_argument("--hand", choices=("left", "right"), default="right",
                    help="hand controlled by the unified Robot control tab (default: right)")
    mode.add_argument("--emg-producer", action="store_true",
                    help="acquire, record and classify EMG; publish predictions only, with "
                         "robot controls in the separate Sharpa Control window")
    ap.add_argument("--robot-udp", "--prediction-udp", dest="robot_udp", metavar="HOST:PORT",
                    help="prefill the prediction/legacy robot-output target (IP literal, "
                         f"e.g. 127.0.0.1:{DEFAULT_PORT} or [::1]:{DEFAULT_PORT}); never starts "
                         "sending automatically")
    args = ap.parse_args(argv)
    if args.sim_contact != "default" and not args.simulate:
        ap.error("--sim-contact requires --simulate")
    args.robot_target = None
    if args.robot_udp is not None:
        try:
            args.robot_target = parse_target(args.robot_udp)
        except ValueError as e:
            ap.error(f"--robot-udp: {e}")
    args.control_target = None
    if not (args.emg_producer or args.legacy_robot_output):
        try:
            args.control_target = (parse_target(args.robot_control) if args.robot_control
                                   else args.robot_target or ("127.0.0.1", DEFAULT_PORT))
        except ValueError as e:
            ap.error(f"--robot-control: {e}")
        if args.robot_target is not None and args.robot_target != args.control_target:
            ap.error("--robot-control and --robot-udp must use the same endpoint")
    return args


def make_source(args, hub):
    if args.simulate:
        from .sources import SimSource
        return SimSource(hub, args.emg_rate, args.package_samples, contact=args.sim_contact)
    if args.control_target is not None:
        from .deferred_source import DeferredSdkSource
        return DeferredSdkSource(hub, args.emg_rate, args.package_samples)
    from .sources import SdkSource
    return SdkSource(hub, args.emg_rate, args.package_samples)


def main(argv=None):
    args = parse_args(argv)
    import pyqtgraph as pg                  # GUI only (parse_args() works without it)

    app = QtWidgets.QApplication(sys.argv[:1])
    pg.setConfigOptions(antialias=False, background="w", foreground="k")

    hub = DataHub()
    source = make_source(args, hub)

    from .ui.main_window import MainWindow
    win = MainWindow(hub, source, args.data_dir,
                     robot_target=args.control_target or args.robot_target,
                     emg_producer=args.emg_producer,
                     robot_control_target=args.control_target, hand=args.hand)
    win.show()

    # Let Ctrl-C in the terminal close the app
    signal.signal(signal.SIGINT, lambda *_: win.close())
    keepalive = QtCore.QTimer()
    keepalive.timeout.connect(lambda: None)
    keepalive.start(250)

    result = app.exec_()
    if win.robot_control is not None and not win.robot_control.client.close_confirmed:
        return 1
    return result


if __name__ == "__main__":
    sys.exit(main())
