"""Live Classifier host benchmark (simulated burst delivery, real selected model):

  python -m gforce_emg_ui.live.bench --model PATH [--seconds S] [--json PATH]

Replays synthetic 16-sample EMG blocks in band-like bursts (~505 rows/s, one burst
every ~150 ms) through DataHub + Recorder, first without and then with the Live
Classifier engine, in real time. Reports the data-thread cost of push_emg, engine
timings and counters, and recorder reconciliation. Synthetic EMG: the predictions
are meaningless; nothing here measures the band, BLE or sensor-to-display latency.
"""

import argparse
import json
import tempfile
import time

import numpy as np

from ..bench import _stats

ROWS_PER_S, BURST_S, BLOCK = 505.0, 0.150, 16


def _run(seconds: float, tmp: str, model=None) -> dict:
    from ..hub import DataHub
    from ..recorder import Recorder
    from ..sources import SimGenerator
    from .engine import LiveEngine

    gen = SimGenerator(seed=0, package=BLOCK)
    hub = DataHub()
    rec = Recorder(hub, tmp)
    session = rec.start("live-bench")
    eng = None
    if model is not None:
        eng = LiveEngine(hub)
        eng.set_model(model)
        eng.start()
    push, owed = [], 0.0
    t0 = next_t = time.monotonic()
    while time.monotonic() - t0 < seconds:
        owed += ROWS_PER_S * BURST_S / BLOCK
        while owed >= 1:                                  # one burst: several blocks at once
            owed -= 1
            now = time.monotonic()
            b = gen.next_emg(host_time=time.time(), host_mono=now)
            t = time.perf_counter()
            hub.push_emg(b)
            push.append(time.perf_counter() - t)
        next_t += BURST_S
        time.sleep(max(0.0, next_t - time.monotonic()))
    out = {"hub_push_us": _stats(push, 1e6), "blocks": len(push)}
    if eng is not None:
        time.sleep(0.2)
        snap = eng.snapshot()
        out["engine"] = {"state": snap.state, "timings_ms": eng.timings(),
                         "counters": snap.counters}
        eng.shutdown()
    session.request_stop()
    session.wait(60)
    c = session.counts["emg"]
    out["recorder"] = {"state": session.state, "received": c["received"],
                       "written": c["written"], "dropped": c["dropped"],
                       "reconciled": c["written"] == c["received"] == c["enqueued"]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gforce_emg_ui.live.bench", description=__doc__)
    ap.add_argument("--model", required=True, help="trusted local model.joblib")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--json", help="also write the results here")
    args = ap.parse_args(argv)
    from .model import load_trusted_model
    model = load_trusted_model(args.model)
    with tempfile.TemporaryDirectory() as tmp:
        res = {"note": "simulated delivery + synthetic EMG; host-side only",
               "model": model.short_id, "seconds": args.seconds,
               "without_live": _run(args.seconds, tmp),
               "with_live": _run(args.seconds, tmp, model)}
    print(json.dumps(res, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
