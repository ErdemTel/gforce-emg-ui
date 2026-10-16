# gforce-emg-ui

Low-latency UI for recording and visualizing the **8-channel EMG** (+ 6-axis IMU) stream of the
OYMotion **gForceUltra** armband, built on the [`sensor-sdk`](https://pypi.org/project/sensor-sdk/)
Python package, PyQt5 and pyqtgraph.

## Setup

Requires Python ≥ 3.10 and git. `sensor-sdk` ships wheels for **Linux x86_64/arm64,
Windows x64 and macOS on Apple Silicon (arm64)**; Intel Macs and 32-bit/ARM Windows are
not supported by the SDK.

The SDK uses a supported USB BLE dongle automatically when one is plugged in and usable,
otherwise the OS Bluetooth stack (BlueZ / WinRT / CoreBluetooth). Check which one is active
with `SensorController().getBLEBackendName()`, or force one with
`SENSOR_SDK_BLE_BACKEND=dongle|bluez|winrt|darwin`.

### Linux (Ubuntu)

```bash
git clone https://github.com/ErdemTel/gforce-emg-ui.git
cd gforce-emg-ui
python3 -m venv .venv                 # Ubuntu: sudo apt install python3-venv if missing
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m gforce_emg_ui --simulate     # check the UI works without the band
```

If Qt fails to start with an `xcb` plugin error, install `sudo apt install libxcb-cursor0 libxkbcommon-x11-0`.

### macOS (Apple Silicon)

```bash
git clone https://github.com/ErdemTel/gforce-emg-ui.git
cd gforce-emg-ui
python3 -m venv .venv                 # python.org installer or `brew install python@3.12`
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m gforce_emg_ui --simulate
```

- On the first scan, macOS asks for **Bluetooth permission** for the app you launched from
  (Terminal, iTerm, VS Code…). Allow it; if you denied it, re-enable it in
  *System Settings → Privacy & Security → Bluetooth* and restart that app.
- A USB dongle needs no extra setup on macOS.

### Windows 10/11 (x64)

In PowerShell:

```powershell
git clone https://github.com/ErdemTel/gforce-emg-ui.git
cd gforce-emg-ui
py -3.12 -m venv .venv                # python.org installer; any 3.10+ works
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m gforce_emg_ui --simulate
```

- Turn Bluetooth on in *Settings → Bluetooth & devices*. Do **not** pair the band in Windows
  settings; the app scans and connects to it directly.
- To use a USB BLE dongle instead of the built-in adapter, run the SDK's one-time driver setup
  (it shows an administrator prompt) and expect `OK`:
  ```powershell
  .venv\Scripts\python -c "from sensor import SensorController; print(SensorController().checkSetupDongle())"
  ```
  The same command performs the permission setup on Linux (asks for `sudo`).

## Run

```bash
.venv/bin/python -m gforce_emg_ui              # real band (Scan → select → Connect)
.venv/bin/python -m gforce_emg_ui --simulate   # synthetic data, no hardware needed
```

On Windows use `.venv\Scripts\python` instead of `.venv/bin/python` (also for the test and
benchmark commands below).

Options: `--emg-rate {500,1000}` (default 1000), `--package-samples N` (samples per BLE
callback, default 16; lower = less latency, the vendor demo uses 32), `--data-dir PATH`,
`--sim-contact {default,good}` (simulator only; `good` = all electrodes good).

## Features

| Area | What it shows |
|---|---|
| **EMG – Equal scale** | 8 stacked channels on one identical y-range (auto-shared or fixed ±), optional DC removal (view only), and per-channel AC RMS bars (last RMS window, independent of display settings, invalid samples excluded; % = share of summed RMS) with the strongest channel highlighted |
| **EMG – Detail** | Like the vendor `pyqt_demo.py`: per channel an FFT spectrum + individually auto-scaled waveform, labelled with channel name and impedance |
| **IMU** | Accelerometer and gyroscope (gForceUltra: 50 Hz) |
| **Electrode contact** | 8 chips colored by impedance: green ≤ 500 kΩ, orange ≤ 999 kΩ, red = loose (thresholds from the vendor demo); channel labels on both EMG tabs use the same colors |
| **Battery** | Percentage bar, updated by the band |
| **Settings** | EMG sample rate 500/1000 Hz, on-board filters (50/60 Hz notch, 10 Hz HPF, 200 Hz LPF) |
| **Recording** | Two subtabs sharing one recorder: **Manual** (CSV session folders with markers, Ctrl+M) and **Stiffness Collection** (guided, timed, labelled trials; see below) |
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

## Stiffness collection

Recording → **Stiffness Collection** records timed trials with one instructed effort level
each (data collection only). Raw files are the same as for manual recordings.

```bash
.venv/bin/python -m gforce_emg_ui                                  # real band
.venv/bin/python -m gforce_emg_ui --simulate                       # software test
```

**Protocol** (durations editable under *Protocol & plan settings*, saved per trial):
PREPARE 5 s → SETTLE 3 s → **HOLD 5 s** (only training candidate) → REST 15 s, then stop,
save and review. *Extend rest* adds 15 s; *Abort* instead of pausing. Next trial starts
only when you click Start.

**Movement** (one per trial, saved as `motion_condition` with a versioned instruction):
stationary, wrist flexion/extension, forearm rotation, reach/sweep. Moving trials start the
movement in SETTLE and continue it through HOLD; HOLD means keeping the instructed effort,
not staying still. Movement is never rejected automatically; recording checks still apply.

**Labels** (`label_source = instructed_effort`; not measured stiffness or co-contraction):

| target | class | s | instruction |
|---|---|---|---|
| low | 0 | 0.0 | minimum comfortable effort, keep the grasp/fist (do not open the hand) |
| medium | 1 | 0.5 | repeatable moderate effort between low and high (not 50 % of anything) |
| high | 2 | 1.0 | firm, comfortable, repeatable, not maximal; rest on fatigue |

**Start is enabled only when**, for 3 s: streaming, newest EMG batch ≤ 0.5 s old (verify on
the band), all 8 channels delivering, form valid, recorder free. Electrode contact and EMG
amplitude are never checked: contact (even red) is only recorded in `trial.json`.

**Aborts** (no resume; cannot be accepted): link loss, stale stream, segment
reset, recorder or trial.json error, phase overrun > 1 s, manual Stop, Disconnect, closing
the app. Lost/invalid samples and gaps during HOLD flag it; flagged HOLDs cannot be accepted.

**Plan** (*Mode → Plan*, *New…*): 200 accepted development trials per participant, seeded
order, no target run > 2, saved before the first trial, resumed in later sessions. Each
condition × target cell is split over the four movements (≥ 2 each; 50 per movement).
Older plan files without movement are used unchanged; their trials are recorded stationary.

| empty_hand (fist) | obj01 | obj02 | obj03 | obj04 | obj05 |
|---|---|---|---|---|---|
| 13/13/14 | 11/11/10 | 11/11/10 | 11/11/10 | 11/10/11 | 10/11/11 |

Tick the objects you have under *Available objects*: the next trial is drawn at random from
the remaining planned trials for those objects, plus empty hand until its trials are done
(no target three times in a row; reproducible from the plan seed). Accept counts a planned
trial once; Reject moves on (trial comes back later); Repeat retries now. Only accepted
attempts are kept: Reject, Repeat or closing the app before review deletes the attempt
folder. Every attempt is a new folder. Progress is recomputed from `trial.json` files.
*Mode → Manual* is for practice / held-out tests (never counted).

```
<data-dir>/stiffness_collection/P01/
  plan_plan-1a2b3c4d.json
  S01/20261015_134056_obj05_medium_s0p5_t001_a01_f889bbb8/
    emg.csv imu.csv contact.csv markers.csv session.json trial.json
<data-dir>/stiffness_collection/simulated/...      simulator data, never real
```

`markers.csv`: `PREPARE_START … HOLD_START, HOLD_END … TRIAL_END` (or `ABORT`). `trial.json`:
IDs, condition, labels, settings, phase times, device/rates/filters, contact snapshot,
quality, outcomes, review; `training_candidate` = HOLD interval in `host_mono` (batch
receipt time: trim boundaries when processing).

**First hardware session:** check rate/filters, run a few practice trials per target,
inspect their `emg.csv` / `markers.csv` / `trial.json`, confirm no false stale aborts with
the 0.5 s limit, then start the plan.

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

Stiffness collection (`collection/`, `ui/collection_panel.py`) reuses the `Recorder`, adds a
small `subscribe_emg` tracker and writes its JSON files on one background thread.

`SimSource` implements the same interface (and state machine) for hardware-free development,
with reproducible fault injection (`SimFaults`: lost samples, missing channels, index jumps,
failed connects; `inject_disconnect()`). For real-time processing (gesture recognition,
force/co-contraction estimation) use `hub.subscribe_emg_async(fn)`: `fn(EmgBlock)` runs on its
own thread, so slow inference never blocks the SDK callback. Subscriber exceptions are isolated
and counted. Features live in `features.py` (`ac_rms` excludes invalid samples).

## Tests and benchmark

```bash
.venv/bin/python -m pytest                     # unit + regression + fault-injection tests
.venv/bin/python -m pytest tests/test_collection_*.py   # stiffness collection only
.venv/bin/python -m gforce_emg_ui.bench        # host-only benchmark (add --quick, --json out.json)
```

The benchmark uses simulated data only. It does **not** measure the band, BLE throughput or
end-to-end sensor latency.

## Reference

The vendor SDK repo (`oymotion/SynchroniSDKPython`, branch `Testing`) is kept alongside this repo at
`../SynchroniSDKPython` for its API docs (`README.md`) and the reference demo `examples/pyqt_demo.py`.

Project context and progress log: [`memory.md`](memory.md).
