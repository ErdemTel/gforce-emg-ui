"""Opt-in prediction log for one live run, in a NEW folder of its own.

  run.json         model / configuration hashes, versions, source, compatibility,
                   settings; rewritten at the end with counters and timings
  predictions.csv  one row per classified window (published or expired)
  events.csv       start / stop / reset / stale events
  display.csv      GUI-applied times, keyed by pred_id (not every result is shown)

All file work runs on one writer thread behind a bounded queue: put_*() never
blocks; overflow is counted (dropped) and write errors are kept in .error, never
raised into acquisition or the GUI. No ground-truth labels are written.
"""

import csv
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..collection.storage import atomic_write_text, dumps
from .engine import DISPLAY_FIELDS, EVENT_FIELDS, PREDICTION_FIELDS

RAW_MARKERS = ("trial.json", "session.json", "emg.csv")
FILES = {"predictions": PREDICTION_FIELDS, "events": EVENT_FIELDS, "display": DISPLAY_FIELDS}


class LogRootError(ValueError):
    pass


def check_root(root: Path, model_path: Optional[Path] = None) -> Path:
    """Refuse log roots inside a recording / trial folder or the model's run folder."""
    root = Path(root).expanduser().resolve()
    for d in (root, *root.parents):
        if any((d / m).exists() for m in RAW_MARKERS):
            raise LogRootError(f"{root} is inside a recording folder ({d}); choose another")
    if model_path is not None:
        run_dir = Path(model_path).resolve().parent
        if root == run_dir or run_dir in root.parents:
            raise LogRootError(f"{root} is inside the model's run folder; choose another")
    return root


class PredictionLog:
    _CLOSE = object()

    def __init__(self, root: Path, run_id: str, meta: dict, simulation: bool,
                 maxsize: int = 20000):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.folder = Path(root) / f"{stamp}_{'sim_' if simulation else ''}live_{run_id}"
        self.meta = dict(meta)
        self.written = {k: 0 for k in FILES}
        self.dropped = 0
        self.error = ""
        self._closed = False
        self._final: dict = {}
        self._q: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, name="live-log", daemon=True)
        self._thread.start()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._done.wait(timeout)

    def _put(self, kind: str, row: dict):
        if self._closed:
            return
        try:
            self._q.put_nowait((kind, row))
        except queue.Full:
            self.dropped += 1

    def put_prediction(self, row: dict):
        self._put("predictions", row)

    def put_event(self, row: dict):
        self._put("events", row)

    def put_display(self, row: dict):
        self._put("display", row)

    def close(self, final: Optional[dict] = None):
        """Idempotent, non-blocking: the writer drains the queue, then finishes run.json."""
        if self._closed:
            return
        self._closed = True
        self._final = dict(final or {})
        while True:
            try:
                self._q.put_nowait((self._CLOSE, None))
                return
            except queue.Full:                    # make room: the close must get through
                try:
                    self._q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def _write_meta(self, final: bool):
        doc = {**self.meta, "log": {"folder": str(self.folder), "written": dict(self.written),
                                    "dropped": self.dropped, "error": self.error or None,
                                    "complete": final}}
        if final:
            doc["end"] = {"host_time": time.time(), **self._final}
        atomic_write_text(self.folder / "run.json", dumps(doc))

    def _run(self):
        files, writers = {}, {}
        try:
            self.folder.mkdir(parents=True, exist_ok=False)
            self._write_meta(False)
            for kind, fields in FILES.items():
                files[kind] = open(self.folder / f"{kind}.csv", "w", newline="")
                writers[kind] = csv.DictWriter(files[kind], fieldnames=fields,
                                               extrasaction="ignore")
                writers[kind].writeheader()
            last_flush = time.monotonic()
            while True:
                try:
                    kind, row = self._q.get(timeout=0.5)
                except queue.Empty:
                    kind = None
                if kind is self._CLOSE:
                    break
                if kind is not None:
                    writers[kind].writerow(row)
                    self.written[kind] += 1
                if time.monotonic() - last_flush > 1.0:
                    for f in files.values():
                        f.flush()
                    last_flush = time.monotonic()
        except Exception as e:                    # keep running the app; report only
            self.error = f"{type(e).__name__}: {e}"
            self._closed = True
        finally:
            for f in files.values():
                try:
                    f.close()
                except Exception:
                    pass
            try:
                if self.folder.is_dir():
                    self._write_meta(True)
            except Exception as e:
                self.error = self.error or f"{type(e).__name__}: {e}"
            self._done.set()
