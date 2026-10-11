"""Host-side benchmark: python -m gforce_emg_ui.bench [--quick] [--seconds S] [--json PATH]

Measures the software pipeline with simulated data only:
  * DataHub.push_emg cost per 16-sample block with the recorder and an async
    "inference" subscriber attached (the work done on the SDK callback thread)
  * recorder throughput, finalize time and count reconciliation
  * ring-buffer snapshot cost
  * dashboard refresh (+ offscreen paint) cost, and the cost of a skipped refresh
  * a real-time simulated stream: delivered rate, callback gaps, callback->draw age

None of this measures the band, BLE link or end-to-end sensor latency.
"""

import argparse
import json
import os
import platform
import sys
import tempfile
import time

import numpy as np

NOTE = ("NOTE: simulator / host-only measurements. These are not hardware latency, "
        "BLE throughput or loss figures for the physical band.")


def _stats(values, unit_scale=1.0):
    a = np.asarray(values, dtype=float) * unit_scale
    return {"n": int(a.size), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def bench_pipeline(n_blocks: int, tmp: str) -> dict:
    from .features import ac_rms
    from .hub import DataHub
    from .recorder import Recorder
    from .sources import SimGenerator

    gen = SimGenerator(seed=0)
    blocks = [gen.next_emg() for _ in range(n_blocks)]
    hub = DataHub()
    rec = Recorder(hub, tmp, queue_size=n_blocks + 16)
    sub = hub.subscribe_emg_async(lambda b: ac_rms(b.data), maxsize=256, name="bench-inference")
    session = rec.start("bench")
    t_start = time.perf_counter()
    push = []
    for b in blocks:
        t = time.perf_counter()
        hub.push_emg(b)
        push.append(time.perf_counter() - t)
    t_pushed = time.perf_counter()
    session.request_stop()
    stop_call = time.perf_counter() - t_pushed
    session.wait(120)
    t_done = time.perf_counter()
    sub.stop()
    c = session.counts["emg"]
    samples = sum(b.data.shape[1] for b in blocks)
    ring = [0.0] * 200
    for i in range(len(ring)):
        t = time.perf_counter()
        hub.emg.latest(2000)
        ring[i] = time.perf_counter() - t
    return {
        "hub_push_us": _stats(push, 1e6),
        "recorder": {
            "state": session.state,
            "samples": samples,
            "written": c["written"],
            "dropped": c["dropped"],
            "reconciled": (c["received"] == c["enqueued"] + c["dropped"]
                           and c["written"] == c["enqueued"] == samples),
            "request_stop_ms": stop_call * 1e3,
            "finalize_ms": (t_done - t_pushed) * 1e3,
            "write_throughput_samples_per_s": samples / max(t_done - t_start, 1e-9),
            "emg_csv_bytes": os.path.getsize(session.folder / "emg.csv"),
        },
        "async_subscriber": {"processed": sub.processed, "dropped": sub.dropped,
                             "errors": sub.errors},
        "ring_latest_us": _stats(ring, 1e6),
    }


def bench_dashboards(app, iterations: int) -> dict:
    from .hub import DataHub
    from .sources import SimGenerator
    from .ui.emg_detail import DetailDashboard
    from .ui.emg_equal import EqualScaleDashboard
    from .ui.imu_view import ImuDashboard

    gen = SimGenerator(seed=1)
    hub = DataHub()
    for _ in range(int(5 * 1000 / gen.package)):        # 5 s of data
        hub.push_emg(gen.next_emg())
        imu = gen.take_imu()
        if imu is not None:
            hub.push_imu(imu)
    out = {}
    for name, cls in (("equal_scale", EqualScaleDashboard), ("detail", DetailDashboard),
                      ("imu", ImuDashboard)):
        dash = cls(hub)
        dash.resize(1200, 800)
        dash.show()
        app.processEvents()
        drawn, skipped = [], []
        for _ in range(iterations):
            hub.push_emg(gen.next_emg())
            imu = gen.take_imu()
            if imu is not None:
                hub.push_imu(imu)
            t = time.perf_counter()
            dash.refresh()
            app.processEvents()
            drawn.append(time.perf_counter() - t)
            t = time.perf_counter()
            dash.refresh()                  # no new data -> must be skipped
            skipped.append(time.perf_counter() - t)
        dash.close()
        out[name] = {"refresh_paint": _stats(drawn, 1e3), "skipped_refresh": _stats(skipped, 1e3)}
    return out


def bench_stream(app, seconds: float) -> dict:
    from PyQt5 import QtCore
    from .hub import DataHub
    from .sources import STREAMING, SimSource

    hub = DataHub()
    src = SimSource(hub, seed=2)
    ages = []
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: ages.append(hub.emg_callback_age_ms()))
    src.connect_device(SimSource.ADDRESS)
    timer.start(33)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.002)
    rate = hub.emg_actual_rate()
    gap95 = hub.emg_callback_gap_ms(95)
    timer.stop()
    ok = src.state == STREAMING
    src.disconnect_device()
    ages = [a for a in ages if a == a]
    return {"seconds": seconds, "streaming": ok, "emg_rate_hz": rate, "nominal_hz": hub.emg_rate,
            "callback_gap_p95_ms": gap95,
            "callback_to_draw_age_ms": _stats(ages) if ages else None,
            "lost_samples": hub.lost_samples, "missing_samples": hub.missing_samples}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gforce_emg_ui.bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smaller run (CI / smoke)")
    ap.add_argument("--seconds", type=float, default=None, help="real-time stream duration")
    ap.add_argument("--json", type=str, default=None, help="write results to this file")
    args = ap.parse_args(argv)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PyQt5 import QtWidgets
    import pyqtgraph as pg
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    pg.setConfigOptions(antialias=False, background="w", foreground="k")

    n_blocks = 500 if args.quick else 5000
    iterations = 20 if args.quick else 100
    seconds = args.seconds if args.seconds is not None else (2.0 if args.quick else 10.0)

    with tempfile.TemporaryDirectory() as tmp:
        res = bench_pipeline(n_blocks, tmp)
    res["dashboard_refresh_ms"] = bench_dashboards(app, iterations)
    res["simulated_stream"] = bench_stream(app, seconds)
    res["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                          "qt_platform": os.environ.get("QT_QPA_PLATFORM"), "quick": args.quick}
    res["note"] = NOTE

    p, r = res["hub_push_us"], res["recorder"]
    print(f"hub.push_emg (16 samples, recorder + async subscriber): "
          f"p50 {p['p50']:.0f} µs, p99 {p['p99']:.0f} µs, max {p['max']:.0f} µs")
    print(f"recorder: {r['written']}/{r['samples']} samples, state {r['state']}, "
          f"reconciled {r['reconciled']}, request_stop {r['request_stop_ms']:.2f} ms, "
          f"finalize {r['finalize_ms']:.0f} ms, {r['write_throughput_samples_per_s']:.0f} samples/s")
    print(f"ring.latest(2000): p50 {res['ring_latest_us']['p50']:.0f} µs")
    for name, d in res["dashboard_refresh_ms"].items():
        print(f"{name:12s} refresh+paint p50 {d['refresh_paint']['p50']:.1f} ms, "
              f"p95 {d['refresh_paint']['p95']:.1f} ms; skipped refresh p50 "
              f"{d['skipped_refresh']['p50'] * 1000:.1f} µs")
    st = res["simulated_stream"]
    age = st["callback_to_draw_age_ms"]
    print(f"simulated stream {st['seconds']:.0f} s: {st['emg_rate_hz']:.1f}/{st['nominal_hz']:g} Hz, "
          f"callback gap p95 {st['callback_gap_p95_ms']:.1f} ms, callback→draw age "
          + (f"p50 {age['p50']:.1f} / p95 {age['p95']:.1f} ms" if age else "n/a"))
    print(NOTE)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2, default=str)
    return 0 if r["reconciled"] and r["state"] == "saved" else 1


if __name__ == "__main__":
    sys.exit(main())
