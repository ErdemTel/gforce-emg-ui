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
| **EMG – Equal scale** | 8 stacked channels on one identical y-range (auto-shared or fixed ±), DC removal, and per-channel RMS bars on a common axis with the strongest channel highlighted, to compare which channel fires most for a gesture |
| **EMG – Detail** | Like the vendor `pyqt_demo.py`: per channel an FFT spectrum + individually auto-scaled waveform, labelled with channel name and impedance |
| **IMU** | Accelerometer and gyroscope (gForceUltra: 50 Hz) |
| **Electrode contact** | 8 chips colored by impedance: green ≤ 500 kΩ, orange ≤ 999 kΩ, red = loose (thresholds from the vendor demo); channel labels on both EMG tabs use the same colors |
| **Battery** | Percentage bar, updated by the band |
| **Settings** | EMG sample rate 500/1000 Hz, on-board filters (50/60 Hz notch, 10 Hz HPF, 200 Hz LPF) |
| **Recording** | CSV session folders with markers (Ctrl+M) for labelling gestures |
| **Status bar** | Actual vs nominal rates, lost packets/samples, packet duration, data age at draw time |

## Recordings

Each recording creates `data/<YYYYmmdd_HHMMSS>_<name>/`:

| File | Columns |
|---|---|
| `emg.csv` | `sample_index, device_time, host_time, lost, ch1..ch8` |
| `imu.csv` | `sample_index, device_time, host_time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z` |
| `contact.csv` | `host_time, imp1..imp8` (Ohm, on change, ~1 Hz) |
| `markers.csv` | `host_time, emg_sample_index, label` |
| `session.json` | device info, rates, filters, duration, sample/lost counts |

## Architecture

```
band (BLE) ─► SdkSource._on_data (SDK thread: copy out of native buffers)
                 │
                 ▼
              DataHub.push_emg / push_imu ──► RingBuffers ──► dashboards (Qt timer, ~30 FPS, visible tab only)
                 │
                 └──► subscribers: Recorder (queue → writer thread), future real-time decoders
```

`SimSource` implements the same interface for hardware-free development. To add real-time
processing (gesture recognition, force/co-contraction estimation), subscribe to the hub:
`hub.subscribe_emg(fn)` — `fn(EmgBlock)` runs on the data thread, so keep it fast or queue work.

## Tests

```bash
.venv/bin/python -m pytest
```

## Reference

The vendor SDK repo (`oymotion/SynchroniSDKPython`, branch `Testing`) is kept alongside this repo at
`../SynchroniSDKPython` for its API docs (`README.md`) and the reference demo `examples/pyqt_demo.py`.

Project context and progress log: [`memory.md`](memory.md).
