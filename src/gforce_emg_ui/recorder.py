"""Session recorder: writes EMG / IMU / contact / markers to CSV on a writer thread.

Layout of one session folder (data/<YYYYmmdd_HHMMSS>[_<name>]/):
  session.json  device + settings + summary
  emg.csv       sample_index, device_time, host_time, lost, ch1..ch8
  imu.csv       sample_index, device_time, host_time, acc_x..gyro_z
  contact.csv   host_time, imp1..imp8 (Ohm), written when values change (~1 Hz)
  markers.csv   host_time, emg_sample_index, label
"""

import json
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from . import __version__
from .config import EMG_CHANNELS
from .hub import DataHub, EmgBlock, ImuBlock

_STOP = object()


class Recorder:
    def __init__(self, hub: DataHub, root: Path):
        self.hub = hub
        self.root = Path(root)
        self.folder: Optional[Path] = None
        self._q: "queue.SimpleQueue" = queue.SimpleQueue()
        self._thread: Optional[threading.Thread] = None
        self._meta: dict = {}
        self._start_time = 0.0
        self._last_emg_index = -1
        self.emg_samples = 0
        self.imu_samples = 0
        self.markers = 0

    @property
    def active(self) -> bool:
        return self._thread is not None

    @property
    def elapsed(self) -> float:
        return time.time() - self._start_time if self.active else 0.0

    def start(self, name: str = "", metadata: Optional[dict] = None) -> Path:
        if self.active:
            raise RuntimeError("already recording")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
        self.folder = self.root / (f"{stamp}_{safe}" if safe else stamp)
        self.folder.mkdir(parents=True, exist_ok=False)
        self._start_time = time.time()
        self.emg_samples = self.imu_samples = self.markers = 0
        self._lost_samples0 = self.hub.lost_samples
        self._lost_packages0 = self.hub.lost_packages
        self._meta = {
            "app_version": __version__,
            "started": datetime.now().isoformat(timespec="milliseconds"),
            "start_host_time": self._start_time,
            "name": name,
            **(metadata or {}),
        }
        self._write_meta()
        self._thread = threading.Thread(target=self._writer, name="recorder", daemon=True)
        self._thread.start()
        self.hub.subscribe_emg(self._on_emg)
        self.hub.subscribe_imu(self._on_imu)
        return self.folder

    def stop(self) -> Optional[Path]:
        if not self.active:
            return None
        self.hub.unsubscribe_emg(self._on_emg)
        self.hub.unsubscribe_imu(self._on_imu)
        self._q.put(_STOP)
        self._thread.join(timeout=10)
        self._thread = None
        self._meta.update({
            "stopped": datetime.now().isoformat(timespec="milliseconds"),
            "duration_s": round(time.time() - self._start_time, 3),
            "emg_samples": self.emg_samples,
            "imu_samples": self.imu_samples,
            "markers": self.markers,
            "lost_samples": self.hub.lost_samples - self._lost_samples0,
            "lost_packages": self.hub.lost_packages - self._lost_packages0,
        })
        self._write_meta()
        return self.folder

    def add_marker(self, label: str):
        if self.active:
            self._q.put(("marker", time.time(), self._last_emg_index, label))
            self.markers += 1

    # -- called on the source thread: only enqueue ---------------------------
    def _on_emg(self, block: EmgBlock):
        self._last_emg_index = int(block.sample_index[-1])
        self.emg_samples += block.data.shape[1]
        self._q.put(("emg", block))

    def _on_imu(self, block: ImuBlock):
        self.imu_samples += block.data.shape[1]
        self._q.put(("imu", block))

    # -- writer thread ---------------------------------------------------------
    def _write_meta(self):
        with open(self.folder / "session.json", "w") as f:
            json.dump(self._meta, f, indent=2, default=str)

    def _writer(self):
        chans = [f"ch{i + 1}" for i in range(EMG_CHANNELS)]
        files = {
            "emg": open(self.folder / "emg.csv", "w"),
            "imu": open(self.folder / "imu.csv", "w"),
            "contact": open(self.folder / "contact.csv", "w"),
            "markers": open(self.folder / "markers.csv", "w"),
        }
        files["emg"].write(",".join(["sample_index", "device_time", "host_time", "lost"] + chans) + "\n")
        files["imu"].write("sample_index,device_time,host_time,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n")
        files["contact"].write(",".join(["host_time"] + [f"imp{i + 1}" for i in range(EMG_CHANNELS)]) + "\n")
        files["markers"].write("host_time,emg_sample_index,label\n")
        emg_fmt = ["%d", "%.6f", "%.6f", "%d"] + ["%.6g"] * EMG_CHANNELS
        imu_fmt = ["%d", "%.6f", "%.6f"] + ["%.6g"] * 6
        last_imp = None
        last_flush = time.time()
        try:
            while True:
                item = self._q.get()
                if item is _STOP:
                    break
                kind = item[0]
                if kind == "emg":
                    b: EmgBlock = item[1]
                    n = b.data.shape[1]
                    rows = np.column_stack((b.sample_index, b.device_time,
                                            np.full(n, b.host_time), b.lost.astype(int), b.data.T))
                    np.savetxt(files["emg"], rows, fmt=emg_fmt, delimiter=",")
                    if b.impedance is not None and (last_imp is None
                                                    or not np.array_equal(b.impedance, last_imp)):
                        last_imp = b.impedance.copy()
                        files["contact"].write(
                            f"{b.host_time:.6f}," + ",".join(f"{v:.0f}" for v in last_imp) + "\n")
                elif kind == "imu":
                    b: ImuBlock = item[1]
                    n = b.data.shape[1]
                    rows = np.column_stack((b.sample_index, b.device_time,
                                            np.full(n, b.host_time), b.data.T))
                    np.savetxt(files["imu"], rows, fmt=imu_fmt, delimiter=",")
                elif kind == "marker":
                    _, t, idx, label = item
                    label = label.replace('"', "")
                    files["markers"].write(f'{t:.6f},{idx},"{label}"\n')
                    files["markers"].flush()
                now = time.time()
                if now - last_flush > 1.0:
                    for f in files.values():
                        f.flush()
                    last_flush = now
        finally:
            for f in files.values():
                f.close()
