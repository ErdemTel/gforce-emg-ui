# Project Memory — gForceUltra EMG UI

Shared, version-controllable record of goals, decisions and progress.
**Rule:** at the end of every Claude session, append a dated entry to the *Progress Log* below.

## Project context

- **Repo:** `~/erdem/gforce-emg-ui` (our code, private GitHub repo `gforce-emg-ui`). SDK comes from pip `sensor-sdk>=1.3.3`.
- **Vendor reference:** `~/erdem/SynchroniSDKPython` (OYMotion, branch `Testing`) — API docs + demos only, no SDK source.
- **Hardware:** OYMotion **gForceUltra** armband (vendor spec, FW V1.0.8): 8-ch EMG @ 500|1000 Hz (default 1000);
  6-axis IMU (ACC+GYRO, aggregate `NTF_IMU`, 50 Hz only, no Euler/Quat); 8-ch electrode impedance (~1 Hz, delivered in
  the EMG samples' `impedance` field); gesture stream; on-board filters FILTER_50HZ/60HZ/HPF(10 Hz)/LPF(200 Hz).
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
8-channel EMG visualization and recording. Downstream: **live pattern recognition**, accurate gesture
control, **fingertip force estimation**, **compliance estimation** (muscle co-contraction).

## Decisions

- EMG at **1000 Hz**; record and show **IMU** too.
- PyQt5 + **pyqtgraph** (not matplotlib) for real-time plots; only the visible tab redraws (~30 FPS).
- Two EMG dashboards: (1) **equal scale** — all channels same y-range + RMS bars to compare which channel fires most;
  (2) **detail** — like `pyqt_demo.py` (FFT + per-channel autoscaled waveform).
- Track **battery** and **electrode contact** (green ≤500 kΩ / orange ≤999 kΩ / red loose — same thresholds as the demo).
- Recording: CSV session folders (emg/imu/contact/markers + session.json) written on a background thread.
  Markers (Ctrl+M) label gestures for training data.
- Real-time consumers (decoders) plug in via `DataHub.subscribe_emg`.
- `PACKAGE_SAMPLE_COUNT` default 16 (demo uses 32) to cut batching latency — **needs verification on the real band**.

## Hardware findings (band BC:93:2A:3F:80:E4, 2026-10-11)

- Reports ModelName `OB6000`, FW `V1.0.4_2026-08-21` (older than spec's V1.0.8), HW 1.40, MTU 247, dongle backend.
- **EMG delivered at ~505 Hz although EMG_SAMPLE_RATE=1000** (steady over 40 s, sample indices continuous, 0 lost).
  Setting 500 Hz → "Error: not supported on this device". IMU: DeviceInfo says 200 Hz, getParam says 50, ~100 Hz delivered.
  Same with package size 16 or 32 and with NTF_GEST off. Mains-hum test inconclusive (band not worn).
- **Data arrives in bursts every ~150 ms** (dongle backend) → dominant latency, outside our code.
- `absTimeStampInSec` = stream-start wall-clock anchor + sampleIndex / 1000 (checked 2026-10-13 in both recordings:
  anchor 1791919335.373, spread < 1 µs) → the SDK timeline runs at ~0.50× host time; not a sensor clock, not latency.
- Extra streams arrive: `NTF_GEST`, `NTF_IMPEDANCE` (ignored); impedance also present in EMG samples (MΩ when not worn).
- BlueZ backend (`SENSOR_SDK_BLE_BACKEND=bluez`) could not find the band in 25 s.
- To do: compare with pyqt_demo's "Actual:" rate line; ask OYMotion about FW update / true rate / 150 ms bursts.

## Lessons: EMG recording interface (2026-10-13)

Research use: EMG recorded during human demonstrations as training-time supervision for robot gain/compliance
selection. Pre-contact timing and avoiding artifact-based labels matter more than smooth plots. "Contact" in
contact.csv = electrode-skin electrical contact, never finger/object contact.

**sensor-sdk 1.3.3 (installed sources read; differs in places from the Testing-branch README)**
- `SensorData` in the data callback is a *view*: sample buffer and batch info point into native memory the SDK
  reuses. Copy everything before returning. `isDataValid(ch, i)` = same stream tag (`startTimeStamp`) AND slot
  sampleIndex == `startSampleIndex + i`; checking it before AND after the copy detects reused/torn buffers
  (the vendor demo calls it and logs "your data process runs too slow").
- Available per batch: `sampleRate`, `channelCount`, `channelMask` (`isChannelEnabled`), `sampleCount`,
  `lostPackageCount`, `startTimeStamp`, `startTimeSec` (wall-clock stream anchor, 0 = unknown), `delay`,
  `startSampleIndex`, `as_numpy()`, `clone()`. 1.3.3 has no `channelSamples` (README mentions it).
- `delay` = first raw packet's bin-record time minus `startTimeStamp` (vendor tests): NOT latency.
- One callback can carry several batches/streams; a batch is not necessarily one BLE packet. On our band we
  observed exactly 1 EMG batch (16 rows) per callback and 1 IMU sample per callback.
- `rawData` / `saturation` meanings for EMG are undocumented; the vendor demo reuses `impedance`/`saturation` as
  gesture probability/strength → never treat `saturation` as an ADC clipping flag. ADC rails/gain unknown.
- Filter cutoffs: SDK README lists 0.5 Hz HPF / 80 Hz LPF (other devices); gForceUltra test spec (FW V1.0.8) lists
  10 Hz / 200 Hz; neither verified for our FW V1.0.4.
- SDK README says `SensorProfile.autoReconnect` defaults to True (Python wrapper property reports False) — possible
  overlap with our own reconnect loop; not investigated.
- Diagnostics via supported keys (untested on hardware): controller `setParam("LOG_PATH", dir)`,
  `("DEBUG_ENABLED","True")`, optional `DATA_LOG_ENABLED`/`BLE_TRACE_ENABLED`; profile `DEBUG_LOG_PATH` and
  `DEBUG_BLE_DATA_PATH` (raw BLE .bin exported on stop/disconnect); `getBinFileInfo`, `parseBinToCsv`.
  Readable params: `EMG_SAMPLE_RATE(_LIST)`, `IMU_SAMPLE_RATE(_LIST)`, `FILTER`, `NTF`, `EMG_RESOLUTION(_LIST)`,
  controller `BACK_END`, timeouts; DeviceInfo has `ConnectionIntervalMs`, `PeripheralLatency`, `SupervisionTimeoutMs`.

**Timing evidence from the 2 local recordings (identified by rows/index range, v1 layout)**
- test1: 12,928 rows, idx 104640–117567, host span 25.631714 s, SDK span 12.927 s, 503.8 rows/s (regression 501.6/s).
  test2: 70,032 rows, idx 632928–702959, host span 139.397263 s, SDK span 70.031 s, 502.3 rows/s, block-index vs
  receipt-time regression 501.75/s; 0 gaps/NaN/lost. Bursts every 151.7 ms (~4.76 batches, ~76 rows), receipt gap
  p50 0.35 ms / p95 150.5 ms. IMU 100.0 rows/s vs 200 Hz reported. EMG/IMU index progression ratio 5.017 (nominal
  5.000) → possibly separate clocks (unverified).
- With all filters ON the spectrum has narrow 20–29 dB notches at 0.100/0.120/0.200/0.240/0.300/0.360/0.400
  cycles/sample and none at 0.05/0.06 → 50/60 Hz notch comb acting on a ~500 samples/s basis. A strong hint, NOT a
  verification (firmware filter design undocumented). Apparent quantization step ≈ 0.0795 µV.
- Rules: received throughput ≠ acquisition rate; continuous indices ≠ lossless acquisition; callback gaps, SDK
  `delay` and callback→draw age ≠ sensor→host latency. Never "correct" device_time, never substitute host time,
  never interpolate missing data. Keep requested / readback / reported / received / verified / assumed rates apart.
- External check (planned): record with FILTER_50HZ/60HZ OFF, find the mains peak (60 Hz here): 0.120 cycles/sample
  ⇒ ~500/s, 0.060 ⇒ ~1000/s; confirm it is mains (vanishes with notch ON, harmonics fit). Also compare against
  `pyqt_demo.py` with matched settings (demo uses packageSampleCount 32; its "Actual" = non-lost ch0 samples since
  stream start / wall-clock time). Caveat: the vendor rate script counts `max-min index + 1`, i.e. includes missing.

**Electrode-skin impedance**
- The SDK copies its latest impedance into every EMG sample: repeated values are cached observations, not new
  measurements; values can change mid-batch (keep the sample position); measurement age is unknown. Changes arrive
  ~every 1.06 s (receipt time). test2: CH3 > 500 kΩ in 9 of 141 rows (48–49 s, 62–68 s; max 940.7 kΩ), CH7 in 2
  (48–49 s; max 924.7 kΩ); large EMG transients (robust z > 10) mostly occur while the channel is green (CH3: 82 of
  98), some before a warning → association only, neither proves artifact.
- 500/999 kΩ colours are the vendor demo's display convention, not a validity threshold. 0, negative, inf or
  sentinel values should be "unknown", never green. Never auto-exclude channels on impedance alone.
- NTF_IMPEDANCE batches also arrive (vendor tests: 1 Hz, batched by packageSampleCount → one batch every ~16–20 s);
  which field (`data` vs `impedance`), units and channel order are unverified.

**Gaps in the current `main` code (0.0.1) found while working — not fixed on main**
- `_on_data` copies but never checks `isDataValid`/`startSampleIndex`; ignores `channelMask`; lost flag from ch0 only.
- contact.csv: last value per batch, only on change, `host_time` only (no host_mono / sample index / provenance),
  no initial row per segment; `contact_level(0)` → green.
- Plots, FFT and RMS "ms" windows use the nominal 1000 Hz → ~2× off on this band (status bar warns).
- An exception after STREAMING in `_connect_worker` (e.g. `getDeviceInfo`) runs cleanup → Disconnected → reconnect loop.
- Markers store the LAST RECEIVED EMG index, not the sample acquired at the event.

## Open questions

- True EMG acquisition rate on FW V1.0.4 (500 Hz ADC? 1000 Hz + decimation?), why `EMG_SAMPLE_RATE=500` is refused,
  FW update path, cause of the ~150 ms bursts (connection interval / peripheral latency?) — ask OYMotion.
- Sensor→host latency: not measured; needs an independent reference (synchronized trigger).
- NTF_IMPEDANCE field/units/channel order; rawData/saturation meaning; ADC rails/gain; channelMask semantics;
  IMU 200 vs 50 vs 100 Hz; EMG vs IMU clock domains.
- Recording format may later move to HDF5/NPY or LSL streaming for the pattern-recognition pipeline.
- Feature/decoder design for gesture, fingertip force and co-contraction estimation (causal windows that reset at
  gaps/segments; millisecond windows only with a verified rate).

## Progress Log

### 2026-10-11
- Surveyed repo; confirmed `pyqt_demo.py` works on our Ubuntu machine.
- Created this `memory.md` and a `CLAUDE.md` that loads it automatically in every Claude Code session.
- Documented how to continue sessions remotely over SSH (tmux + `claude --continue`/`--resume`, Remote Control).
- Created fresh repo `~/erdem/gforce-emg-ui` (SDK via pip, vendor repo kept as reference); moved `memory.md`/`CLAUDE.md` here.
- Set up repo `.venv` (adds pyqtgraph) and repo-local, isolated GitHub credentials (shared Linux account).
- Remote-dev workflow decided: SSH + tmux as primary, Remote Control for phone check-ins (see *Remote workflow*).
- Pushed to private GitHub repo https://github.com/ErdemTel/gforce-emg-ui (gh login `ErdemTel` isolated in `~/erdem/.config/gh`).
- Requirements defined (see *Decisions*). Built v0.0.1 of the UI (`python -m gforce_emg_ui [--simulate]`):
  `SdkSource` (real band) + `SimSource` (synthetic data), `DataHub` ring buffers, equal-scale + detail EMG dashboards,
  IMU tab, contact chips, battery, EMG-rate/filter settings, CSV recorder with markers, status bar with rates/loss/data age.
- Tests: 5 pytest unit tests pass; headless GUI smoke test with simulator: EMG ~997/1000 Hz, redraw 3–12 ms/frame,
  recording verified. Real-SDK path starts and scans via USB dongle, but **no band was on, so connect is untested**.
- Next: test with the real band (connect, latency, package size, units), then start the real-time processing layer.
- Real-band test: connect/stream/battery/params/disconnect all work via the UI's SdkSource. Found ~505 Hz effective EMG
  rate and 150 ms delivery bursts (see *Hardware findings*). Added red status-bar warning when actual ≠ nominal rate.

### 2026-10-11 (hardening, branch `hardening`)
- Added 36 failing regression tests first (commit 7657f62), then fixed in 4 commits:
  recording lifecycle (af9ed6b), data integrity (376c214), measurement consistency (7fac69d),
  reliability/performance + benchmark (ead5a90). Full suite: 40 passed, 5/5 repeated runs.
- Recording: per-session queue/ID/markers, SAVED only after writer confirmation, ERROR on writer
  exception/overflow, reconciled received/enqueued/dropped/written/rejected counts, non-blocking stop.
- Integrity: NaN + valid masks (no zero padding), segments + index gaps, settings locked while recording,
  CSV %.9g + empty cells, NaN-equal contact logging.
- Measurement: RMS from hub independent of view; "% = share of summed RMS"; monotonic timing;
  status shows host-only callback→draw age and "sensor→host latency: not measured".
- Reliability: isolated subscribers + subscribe_emg_async, connection states + reconnect + failed-connect
  cleanup (tested with fake SDK + simulator), redraw skipping, `python -m gforce_emg_ui.bench`.
- Bug found by the tests: hub unsubscribe used `is` on bound methods → closed sessions stayed subscribed.
- Benchmark (simulator, host only, this machine): push_emg p50 6 µs / p99 15 µs / max 5.7 ms; recorder
  80k samples reconciled, stop call 0.01 ms; refresh+paint p50 9.8 / 3.6 / 2.3 ms (equal/detail/IMU).
- Hardware check of `hardening` with band BC:93:2A:3F:80:E4 (FW V1.0.4, dongle backend), 60 s + reconnect:
  * connect 7.0 s to first data; EMG 503–509 Hz delivered (nominal 1000) in 16-sample batches arriving in
    ~150 ms bursts (callback gap p95 150.5 ms) — unchanged from the earlier test; IMU ~100 of 200 Hz.
  * 0 lost samples, 0 index gaps, 0 in-block anomalies; CSV audit: 35,152 rows, indices strictly
    increasing per segment, counts reconciled, session SAVED.
  * forced link drop (sensor.disconnect() behind the source's back) during recording -> reconnecting ->
    streaming in 8.9 s, new segment 2 (indices restart at 0), recording continued and saved.
  * user disconnect -> idle; connect to a non-existent address -> failed with sensor cleared.
  * host work on the SDK callback thread (hub.push_emg incl. recorder): p50 14 µs, p99 52 µs, max 227 µs.
  * full UI with the band: settings locked while recording, closing the window mid-recording finalized
    (9,152/9,152 samples) before closing.
  * found + fixed: queued state signals can arrive out of order -> UI now reads source.state (commit after docs).
  * NOT measured: sensor→host latency (no external reference); a real radio link loss (out of range/power off).
- Next: merge `hardening` into main; ask OYMotion about 500 Hz effective rate / 150 ms bursts on FW V1.0.4.
- Merged `hardening` into `main` (fast-forward, 41 tests passing) and pushed; branch deleted.
- Open items: user's interactive UI test with the band (incl. a real radio link loss: walk out of range /
  power-cycle during recording); ask OYMotion about effective rate + 150 ms bursts; then start the
  real-time processing layer (gesture recognition, fingertip force, co-contraction) via subscribe_emg_async.
- This chat lives in ~/erdem/SynchroniSDKPython (session c01fdaef-90a2-4a46-97a6-311aa3b13ac1);
  start new sessions in ~/erdem/gforce-emg-ui (inside tmux).

### 2026-10-12
- README: added setup instructions for macOS (Apple Silicon only; SDK has no Intel-Mac wheel) and
  Windows x64 (PowerShell, `.venv\Scripts\python`, dongle driver via `checkSetupDongle()`).
  Written from the SDK docs; not tried on a Mac or Windows machine yet.

### 2026-10-13
- Built a larger reliability update (0.1.0: rate separation, buffer validation, schema v2 recorder, contact
  provenance, offline audit, headless capture) and pushed it as branch `reliability-0.1.0`; user then decided to
  keep the `main` version (0.0.1). Branch deleted locally and on GitHub; only the lessons above were kept.
  (The commit 0e6a20f stays recoverable locally via reflog for a while: `git branch reliability-0.1.0 0e6a20f`.)
- Audited both local recordings with that tool (lessons above); no hardware run this session.
- Next: hardware validation (mains reference, vendor-demo comparison), ask OYMotion the open questions.

## Remote workflow

- Primary: `ssh undergrads@128.32.164.118` → `tmux new -As emg` → `cd ~/erdem/gforce-emg-ui && claude --continue`.
  Detach with `Ctrl-b d`; the session keeps running. Optional: VS Code/Cursor Remote-SSH to browse/edit code.
- From phone/browser: `/remote-control` inside the tmux Claude session, then use claude.ai or the Claude app.
- Viewing the Qt UI remotely needs a remote desktop (RustDesk/NoMachine) or `ssh -X` (slow); BLE band must be near this machine.
