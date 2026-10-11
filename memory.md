# Project Memory — gForceUltra EMG UI

Shared, version-controllable record of goals, decisions and progress.
**Rule:** at the end of every Claude session, append a dated entry to the *Progress Log* below.

## Project context

- **Repo:** `~/erdem/gforce-emg-ui` (our code, private GitHub repo `gforce-emg-ui`). SDK comes from pip `sensor-sdk>=1.3.3`.
- **Vendor reference:** `~/erdem/SynchroniSDKPython` (OYMotion, branch `Testing`) — API docs + demos only, no SDK source.
- **Hardware:** OYMotion **gForceUltra** armband — 8-channel EMG plus IMU (ACC, GYRO, Euler, Quaternion) and other sensors.
- **Primary focus:** the **8-channel EMG** stream, recorded with **as little latency as possible**.
- **Machine:** Ubuntu (host `yixiao-System-Product-Name`), Python 3.12.3. **Shared `undergrads` account** — keep git/gh credentials scoped to `~/erdem` (never change global git/gh config).
  Repo `.venv`: `sensor-sdk 1.3.3`, `PyQt5 5.15.11`, `pyqtgraph 0.14.0`, `numpy 2.x`.
- **BLE backend:** SDK auto-selects a USB dongle ("bumble" host-mode backend) if one is usable, otherwise BlueZ.
  Check with `SensorController.getBLEBackendName()`; force with `SENSOR_SDK_BLE_BACKEND=dongle|bluez`.

## Reference material in the repo

- `../SynchroniSDKPython/examples/pyqt_demo.py` (~4300 lines, PyQt5 + matplotlib) — vendor demo; **tested by us and works well**.
  Useful as a reference for scan → connect → `initDataTransfer` → notification flow, EMG sample rate
  selection (`EMG_SAMPLE_RATE_CANDIDATES = (500, 1000)`), auto-reconnect, and debug logging via `setParam`.
- `../SynchroniSDKPython/examples/console.py`, `async_console.py` — minimal SDK usage without a GUI.
- `../SynchroniSDKPython/README.md` — full SDK API reference (SensorController / SensorProfile methods, callbacks, data types).
- `../SynchroniSDKPython/Test Cases/Automation/gForceUltra/` — smoke, stress, longevity and sample-rate measurement scripts;
  `Test Cases/Automation/device_specs/gforce_ultra.py` — device spec.

## Goal

Build **our own UI** (replacing `pyqt_demo.py` for our applications) centred on low-latency
8-channel EMG visualization and recording.

## Open decisions / requirements to pin down

- UI toolkit and plotting library (matplotlib in the demo is slow for real-time; pyqtgraph/vispy are candidates).
- Recording format (CSV / NPY / HDF5 / EDF), timestamps, and metadata to save.
- Which non-EMG streams (IMU etc.) to show/record, if any.
- Target EMG sample rate (500 vs 1000 Hz) and how to measure end-to-end latency.
- Downstream use (real-time classification? offline analysis? LSL streaming?).

## Progress Log

### 2026-10-11
- Surveyed repo; confirmed `pyqt_demo.py` works on our Ubuntu machine.
- Created this `memory.md` and a `CLAUDE.md` that loads it automatically in every Claude Code session.
- Documented how to continue sessions remotely over SSH (tmux + `claude --continue`/`--resume`, Remote Control).
- Created fresh repo `~/erdem/gforce-emg-ui` (SDK via pip, vendor repo kept as reference); moved `memory.md`/`CLAUDE.md` here.
- Set up repo `.venv` (adds pyqtgraph) and repo-local, isolated GitHub credentials (shared Linux account).
- Remote-dev workflow decided: SSH + tmux as primary, Remote Control for phone check-ins (see *Remote workflow*).
- Pushed to private GitHub repo https://github.com/ErdemTel/gforce-emg-ui (gh login `ErdemTel` isolated in `~/erdem/.config/gh`).
- Next: define UI requirements (see *Open decisions*), then start the custom EMG UI.

## Remote workflow

- Primary: `ssh undergrads@128.32.164.118` → `tmux new -As emg` → `cd ~/erdem/gforce-emg-ui && claude --continue`.
  Detach with `Ctrl-b d`; the session keeps running. Optional: VS Code/Cursor Remote-SSH to browse/edit code.
- From phone/browser: `/remote-control` inside the tmux Claude session, then use claude.ai or the Claude app.
- Viewing the Qt UI remotely needs a remote desktop (RustDesk/NoMachine) or `ssh -X` (slow); BLE band must be near this machine.
