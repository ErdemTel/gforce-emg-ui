"""Session recorder: writes EMG / IMU / contact / markers to CSV on a writer thread.

Each recording is a RecordingSession with its own queue, ID, marker state and
counters. Lifecycle: RECORDING -> FINALIZING -> SAVED, or ERROR at any point
(writer exception, queue overflow, unreconciled counts). SAVED is only set by
the writer thread after every enqueued item was written and session.json saved.

Per stream (emg/imu samples, markers) the session counts:
  received  accepted by the session callback
  enqueued  put on the writer queue      (received == enqueued + dropped)
  dropped   queue full (overflow -> ERROR)
  written   written by the writer        (written == enqueued when SAVED)
  rejected  arrived after the session closed (never written)

Layout of one session folder (data/<YYYYmmdd_HHMMSS>[_<name>][-N]/):
  session.json  id, device + settings, state, error, counts
  emg.csv       segment, sample_index, device_time, host_time, host_mono, lost,
                valid_mask, ch1..ch8   (invalid values are empty cells, never 0)
  imu.csv       segment, sample_index, device_time, host_time, host_mono, valid_mask,
                acc_x..gyro_z
  contact.csv   host_time, segment, imp1..imp8 (Ohm, empty = unknown), on change (~1 Hz)
  markers.csv   host_time, host_mono, segment, emg_sample_index, label

device_time is the SDK timestamp (sample_index / nominal rate), not a sensor clock.
host_time is wall-clock (time.time) for alignment; host_mono is time.monotonic for
durations. Only received samples are written; index gaps stay visible as jumps.
"""

import csv
import io
import json
import queue
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from . import __version__
from .config import EMG_CHANNELS
from .hub import DataHub, EmgBlock, ImuBlock

RECORDING, FINALIZING, SAVED, ERROR = "recording", "finalizing", "saved", "error"

_CH = [f"ch{i + 1}" for i in range(EMG_CHANNELS)]
HEADERS = {
    "emg": ["segment", "sample_index", "device_time", "host_time", "host_mono",
            "lost", "valid_mask"] + _CH,
    "imu": ["segment", "sample_index", "device_time", "host_time", "host_mono", "valid_mask",
            "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
    "contact": ["host_time", "segment"] + [f"imp{i + 1}" for i in range(EMG_CHANNELS)],
    "markers": ["host_time", "host_mono", "segment", "emg_sample_index", "label"],
}
# Times: microseconds. Values: 9 significant digits = exact float32 round trip.
_TIME = ["%.6f", "%.6f", "%.6f"]
EMG_FMT = ["%d", "%d"] + _TIME + ["%d", "%d"] + ["%.9g"] * EMG_CHANNELS
IMU_FMT = ["%d", "%d"] + _TIME + ["%d"] + ["%.9g"] * 6
CONTACT_FMT = ["%.6f", "%d"] + ["%.9g"] * EMG_CHANNELS
_NAN_CELL = re.compile(r"(?<=,)[-+]?nan(?=,|$)", re.MULTILINE)


def _format_rows(rows: np.ndarray, fmt) -> str:
    """CSV text for numeric rows; NaN (invalid / missing) becomes an empty cell."""
    buf = io.StringIO()
    np.savetxt(buf, rows, fmt=fmt, delimiter=",")
    return _NAN_CELL.sub("", buf.getvalue())


def _valid_mask(valid: np.ndarray) -> np.ndarray:
    """Per-sample bitmask of valid channels (bit c = channel c+1 valid)."""
    weights = (1 << np.arange(valid.shape[0], dtype=np.int64))[:, None]
    return (valid.astype(np.int64) * weights).sum(axis=0)
STREAMS = ("emg", "imu", "markers")
DEFAULT_QUEUE_SIZE = 4096          # blocks; ~1 min of EMG at 16 samples/packet


class RecordingSession:
    def __init__(self, hub: DataHub, folder: Path, name: str, metadata: dict,
                 queue_size: int = DEFAULT_QUEUE_SIZE):
        self.hub = hub
        self.folder = folder
        self.session_id = uuid.uuid4().hex[:12]
        self.state = RECORDING
        self.error = ""
        self.counts = {s: dict(received=0, enqueued=0, dropped=0, written=0, rejected=0)
                       for s in STREAMS}
        self._q: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._accepting = True
        self._stop = threading.Event()
        self._done = threading.Event()
        self._last_emg_index: Optional[int] = None
        self._last_segment: Optional[int] = None
        self._t0 = time.monotonic()
        self._t_end: Optional[float] = None
        self._meta = {
            "session_id": self.session_id,
            "app_version": __version__,
            "started": datetime.now().isoformat(timespec="milliseconds"),
            "start_host_time": time.time(),
            "name": name,
            **(metadata or {}),
        }
        self._sub_name = f"recorder-{self.session_id}"
        self._thread = threading.Thread(target=self._writer, name=self._sub_name, daemon=True)
        self._thread.start()
        hub.subscribe_emg(self._on_emg, name=self._sub_name)
        hub.subscribe_imu(self._on_imu, name=self._sub_name)

    # -- status ------------------------------------------------------------------
    @property
    def accepting(self) -> bool:
        return self._accepting

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def elapsed(self) -> float:
        end = self._t_end if self._t_end is not None else time.monotonic()
        return end - self._t0

    def wait(self, timeout: Optional[float] = None) -> bool:
        """True once the writer has finished (state is then SAVED or ERROR)."""
        return self._done.wait(timeout)

    # -- control (any thread, never blocks on I/O) ------------------------------
    def request_stop(self):
        with self._lock:
            if self._accepting:
                self._accepting = False
                self._t_end = time.monotonic()
            if self.state == RECORDING:
                self.state = FINALIZING
        self._unsubscribe()
        self._stop.set()

    def add_marker(self, label: str, host_time: Optional[float] = None,
                   host_mono: Optional[float] = None):
        """Queue a marker. host_time/host_mono default to now; a caller that
        timestamps its own event (e.g. a protocol phase change) may pass them."""
        t = time.time() if host_time is None else host_time
        mono = time.monotonic() if host_mono is None else host_mono
        with self._lock:
            if not self._accepting:
                self.counts["markers"]["rejected"] += 1
                return
            self._put("markers", ("marker", t, mono, self._last_emg_index,
                                  self._last_segment, label), 1)

    # -- data callbacks (source thread): only enqueue -----------------------------
    def _on_emg(self, block: EmgBlock):
        n = block.data.shape[1]
        with self._lock:
            if not self._accepting:
                self.counts["emg"]["rejected"] += n
                return
            self._last_emg_index = int(block.sample_index[-1])
            self._last_segment = block.segment
            self._put("emg", ("emg", block), n)

    def _on_imu(self, block: ImuBlock):
        n = block.data.shape[1]
        with self._lock:
            if not self._accepting:
                self.counts["imu"]["rejected"] += n
                return
            self._put("imu", ("imu", block), n)

    def _put(self, stream: str, item, n: int):
        # Caller holds self._lock
        c = self.counts[stream]
        c["received"] += n
        try:
            self._q.put_nowait(item)
            c["enqueued"] += n
        except queue.Full:
            c["dropped"] += n
            self._set_error(f"queue overflow: {c['dropped']} {stream} samples dropped")

    def _set_error(self, msg: str):
        if not self.error:
            self.error = msg
        elif msg.startswith("queue overflow") and self.error.startswith("queue overflow"):
            self.error = msg
        self.state = ERROR

    def _unsubscribe(self):
        self.hub.unsubscribe_emg(self._on_emg)
        self.hub.unsubscribe_imu(self._on_imu)

    # -- writer thread -------------------------------------------------------------
    def _write_emg(self, f, block: EmgBlock):
        n = block.data.shape[1]
        rows = np.column_stack((np.full(n, block.segment), block.sample_index, block.device_time,
                                np.full(n, block.host_time), np.full(n, block.host_mono),
                                block.lost.astype(int), _valid_mask(block.valid), block.data.T))
        f.write(_format_rows(rows, EMG_FMT))

    def _write_contact(self, f, block: EmgBlock):
        imp = block.impedance
        if imp is None or (self._last_imp is not None
                           and np.array_equal(imp, self._last_imp, equal_nan=True)):
            return
        self._last_imp = np.array(imp, dtype=float)
        row = np.concatenate(([block.host_time, block.segment], self._last_imp))[None, :]
        f.write(_format_rows(row, CONTACT_FMT))

    def _write_imu(self, f, block: ImuBlock):
        n = block.data.shape[1]
        rows = np.column_stack((np.full(n, block.segment), block.sample_index, block.device_time,
                                np.full(n, block.host_time), np.full(n, block.host_mono),
                                _valid_mask(block.valid), block.data.T))
        f.write(_format_rows(rows, IMU_FMT))

    def _write_marker(self, w, item):
        _, t, mono, idx, segment, label = item
        w.writerow([f"{t:.6f}", f"{mono:.6f}", "" if segment is None else segment,
                    "" if idx is None else idx, label])

    def _open_files(self):
        files = {k: open(self.folder / f"{k}.csv", "w", newline="")
                 for k in ("emg", "imu", "contact", "markers")}
        for k, header in HEADERS.items():
            files[k].write(",".join(header) + "\n")
        return files

    def _write_item(self, files, marker_writer, item):
        kind = item[0]
        if kind == "emg":
            block = item[1]
            self._write_emg(files["emg"], block)
            self.counts["emg"]["written"] += block.data.shape[1]
            self._write_contact(files["contact"], block)
        elif kind == "imu":
            block = item[1]
            self._write_imu(files["imu"], block)
            self.counts["imu"]["written"] += block.data.shape[1]
        elif kind == "marker":
            self._write_marker(marker_writer, item)
            files["markers"].flush()
            self.counts["markers"]["written"] += 1

    def _fail_from_writer(self, exc: BaseException):
        with self._lock:
            self._accepting = False
            if self._t_end is None:
                self._t_end = time.monotonic()
            self._set_error(f"writer failed: {type(exc).__name__}: {exc}")
        self._unsubscribe()

    def _writer(self):
        files = {}
        self._last_imp = None
        try:
            files = self._open_files()
            marker_writer = csv.writer(files["markers"])
            last_flush = time.monotonic()
            while True:
                try:
                    item = self._q.get(timeout=0.05)
                except queue.Empty:
                    if self._stop.is_set():
                        break
                    continue
                self._write_item(files, marker_writer, item)
                now = time.monotonic()
                if now - last_flush > 1.0:
                    for f in files.values():
                        f.flush()
                    last_flush = now
        except BaseException as e:      # noqa: BLE001 - surfaced as ERROR state
            self._fail_from_writer(e)
        finally:
            for f in files.values():
                try:
                    f.close()
                except Exception as e:
                    self._fail_from_writer(e)
            self._finalize()

    def _finalize(self):
        with self._lock:
            unreconciled = [s for s, c in self.counts.items()
                            if c["received"] != c["enqueued"] + c["dropped"]
                            or c["written"] != c["enqueued"]]
            if unreconciled and self.state != ERROR:
                self._set_error(f"counts do not reconcile for {', '.join(unreconciled)}")
            final = ERROR if self.state == ERROR else SAVED
            self._meta.update({
                "stopped": datetime.now().isoformat(timespec="milliseconds"),
                "duration_s": round(self.elapsed, 3),
                "state": final,
                "error": self.error,
                "counts": {s: dict(c) for s, c in self.counts.items()},
            })
        try:
            with open(self.folder / "session.json", "w") as f:
                json.dump(self._meta, f, indent=2, default=str)
        except Exception as e:
            final = ERROR
            with self._lock:
                self._set_error(f"could not write session.json: {e}")
        self.state = final
        self._done.set()


class Recorder:
    """Creates sessions; at most one may be RECORDING or FINALIZING at a time."""

    def __init__(self, hub: DataHub, root: Path, queue_size: int = DEFAULT_QUEUE_SIZE):
        self.hub = hub
        self.root = Path(root)
        self.queue_size = queue_size
        self.session: Optional[RecordingSession] = None

    @property
    def active(self) -> bool:
        """A session is accepting data."""
        return self.session is not None and self.session.accepting

    @property
    def busy(self) -> bool:
        """A session exists whose writer has not finished yet."""
        return self.session is not None and not self.session.done

    def start(self, name: str = "", metadata: Optional[dict] = None,
              root: Optional[Path] = None) -> RecordingSession:
        """root overrides self.root for this session only (self.root is unchanged)."""
        if self.busy:
            raise RuntimeError("previous recording is still in progress")
        root = self.root if root is None else Path(root)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
        base = f"{stamp}_{safe}" if safe else stamp
        root.mkdir(parents=True, exist_ok=True)
        for i in range(1, 1000):
            folder = root / (base if i == 1 else f"{base}-{i}")
            try:
                folder.mkdir()
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError(f"could not create a unique folder for {base}")
        self.session = RecordingSession(self.hub, folder, name, metadata or {}, self.queue_size)
        return self.session

    def stop(self) -> Optional[RecordingSession]:
        """Request stop; returns immediately. Poll session.state / session.done."""
        if self.session is not None:
            self.session.request_stop()
        return self.session

    def add_marker(self, label: str, host_time: Optional[float] = None,
                   host_mono: Optional[float] = None):
        if self.active:
            self.session.add_marker(label, host_time, host_mono)
