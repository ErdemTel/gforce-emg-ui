# gforce-emg-ui

Low-latency UI for recording and visualizing the **8-channel EMG** (+ 6-axis IMU) stream of the
OYMotion **gForceUltra** armband, built on the [`sensor-sdk`](https://pypi.org/project/sensor-sdk/)
Python package, PyQt5 and pyqtgraph.

## Setup (Ubuntu, Python ≥ 3.10)

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

The SDK picks a USB BLE dongle automatically if one is present, otherwise BlueZ.

## Run

```bash
.venv/bin/python -m gforce_emg_ui              # real band (Scan → select → Connect)
.venv/bin/python -m gforce_emg_ui --simulate   # synthetic data, no hardware needed
```

Options: `--emg-rate {500,1000}` (default 1000), `--package-samples N` (samples per BLE
callback, default 16; lower = less latency, the vendor demo uses 32), `--data-dir PATH`.

## Features

| Area | What it shows |
|---|---|
| **EMG – Equal scale** | 8 stacked channels on one identical y-range (auto-shared or fixed ±), optional DC removal (view only), and per-channel AC RMS bars (last RMS window, independent of display settings, invalid samples excluded; % = share of summed RMS) with the strongest channel highlighted |
| **EMG – Detail** | Like the vendor `pyqt_demo.py`: per channel an FFT spectrum + individually auto-scaled waveform, labelled with channel name and impedance |
| **IMU** | Accelerometer and gyroscope (gForceUltra: 50 Hz) |
| **Electrode contact** | 8 chips colored by impedance: green ≤ 500 kΩ, orange ≤ 999 kΩ, red = loose (thresholds from the vendor demo); channel labels on both EMG tabs use the same colors |
| **Battery** | Percentage bar, updated by the band |
| **Settings** | EMG sample rate 500/1000 Hz, on-board filters (50/60 Hz notch, 10 Hz HPF, 200 Hz LPF) |
| **Recording** | CSV session folders with markers (Ctrl+M) for labelling gestures |
| **Status bar** | Actual vs nominal rates (red warning on mismatch), lost packets/samples, index gaps, segment, callback gap p95, host-only callback→draw age; sensor→host latency is reported as not measured |

## Recordings

Each recording creates `data/<YYYYmmdd_HHMMSS>_<name>[-N]/`. Firmware filters and the EMG
rate are locked while a recording is open. Stopping returns immediately; the panel shows
**Finalizing…**, then **Saved** only after the writer confirmed every queued sample was
written and the counts reconcile, otherwise **ERROR** with the reason (writer exception,
queue overflow).

| File | Columns |
|---|---|
| `emg.csv` | `segment, sample_index, device_time, host_time, host_mono, lost, valid_mask, ch1..ch8` |
| `imu.csv` | `segment, sample_index, device_time, host_time, host_mono, valid_mask, acc_x..gyro_z` |
| `contact.csv` | `host_time, segment, imp1..imp8` (Ohm; written when values change, ~1 Hz) |
| `markers.csv` | `host_time, host_mono, segment, emg_sample_index, label` |
| `session.json` | session_id, device, rates, filters, state, error, duration, per-stream counts |

- Invalid values (lost samples, missing channels, unknown impedance) are **empty cells**, never 0.
  `valid_mask` is a per-sample bitmask of valid channels (255 = all 8).
- `segment` increments on reconnection, rate change, index reset or a jump > 1 s.
  Only received samples are written; shorter index gaps are visible as jumps in `sample_index`.
- Values use `%.9g` (exact float32 round trip); times are in seconds with µs resolution.
- `device_time` is the SDK timestamp (= sample_index / nominal rate), **not** a sensor clock.
  `host_time` is wall-clock for alignment; `host_mono` (monotonic) is for durations.
- Counts in `session.json`: `received == enqueued + dropped`, `written == enqueued` when saved;
  `rejected` = data that arrived after the session closed (not written).

## Connection states

`idle → scanning → connecting → streaming`. A failed connect cleans up and ends in `failed`.
An unexpected link loss goes to `reconnecting` (automatic retries, new data segment; an open
recording continues) and ends in `failed` if all retries fail. Disconnect → `idle`.

## Architecture

```
band (BLE) ─► SdkSource._on_data (SDK thread: copy out of native buffers, mark invalid as NaN)
                 │
                 ▼
              DataHub.push_emg / push_imu ── segments/gaps ──► RingBuffers ──► dashboards
                 │                                                (Qt timer ~30 FPS, visible tab,
                 │                                                 skipped when nothing changed)
                 ├──► Recorder session (bounded queue → writer thread)
                 └──► subscribe_emg_async(decoder) (own thread, bounded queue)
```

`SimSource` implements the same interface (and state machine) for hardware-free development,
with reproducible fault injection (`SimFaults`: lost samples, missing channels, index jumps,
failed connects; `inject_disconnect()`). For real-time processing (gesture recognition,
force/co-contraction estimation) use `hub.subscribe_emg_async(fn)`: `fn(EmgBlock)` runs on its
own thread, so slow inference never blocks the SDK callback. Subscriber exceptions are isolated
and counted. Features live in `features.py` (`ac_rms` excludes invalid samples).

## Tests and benchmark

```bash
.venv/bin/python -m pytest                     # unit + regression + fault-injection tests
.venv/bin/python -m gforce_emg_ui.bench        # host-only benchmark (add --quick, --json out.json)
```

The benchmark uses simulated data only. It does **not** measure the band, BLE throughput or
end-to-end sensor latency.

## Reference

The vendor SDK repo (`oymotion/SynchroniSDKPython`, branch `Testing`) is kept alongside this repo at
`../SynchroniSDKPython` for its API docs (`README.md`) and the reference demo `examples/pyqt_demo.py`.

Project context and progress log: [`memory.md`](memory.md).
