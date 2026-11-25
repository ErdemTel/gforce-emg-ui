# gforce-emg-ui

Low-latency UI for recording and visualizing the **8-channel EMG** (+ 6-axis IMU) stream of the
OYMotion **gForceUltra** armband, built on the [`sensor-sdk`](https://pypi.org/project/sensor-sdk/)
Python package, PyQt5 and pyqtgraph.

## Current Sharpa workflow: one GUI, Manual or EMG

On the dual-arm PC, start the robot backend in one terminal:

```bash
~/ws_fanuc/src/dual_sharpa_wave_ros2/script/start_compliance.sh --hand right --mode hardware
```

Then open this existing GUI in a second terminal:

```bash
~/erdem/emg-sharpa-stiffness/gforce-emg-ui/start_gui.sh
```

The backend opens no window. This app opens its **Robot control** tab with
**Manual / EMG** selection, Kp/Kd editing, Start/Pause, reset to 1/1 and verified
device status. Manual works without a band, model, or EMG SDK initialization.
Scan/Connect initializes the band only when requested. An EMG connection or
classifier failure does not prevent manual robot control.

In Manual, enter gains → Apply gains → Start control. Requested gains and
**Last verified device gains** are separate; the display remains available while
paused. Start reads identity/gains before enabling updates. For a stationary
unloaded compliance test, explicitly use Hold current pose. Release that local
publisher before starting teleoperation.

**Return full robot home** pauses updates and releases this GUI's pose hold, then
moves **both FANUC arms to the configured lab home pose, followed by both Sharpa
hands to zero**, regardless of the hand selected for EMG. Current hand gains are
preserved; positive Kd is required on every joint. Exit teleop, close other motion
or gain senders, and clear the whole path first. This is joint interpolation, with
no collision planning. Both arm controller sets must be active, the force limiter
must be ARMED, and both hand drivers/feedback must be available. It refuses to
fall back to a partial return if components are missing.

The GUI reports HOMING, the current stage, and completion/failure. **Cancel return
home** stops new arm targets and cancels hand recovery; the last arm target is
retained, so cancellation is not an emergency stop. Updates remain paused and
pose hold remains off afterward. Restart the updated coordinator and GUI to load
this feature; the button stays disabled with an older backend or in Preview.

The current backend presets are Low (0): **0.15 / 0.30**, Medium (0.5):
**0.40 / 0.60**, High (1): **1 / 1** (Kp / Kd multipliers). The GUI displays the
loaded backend mapping; restart the coordinator after a preset configuration edit.
These development settings are separate from driver startup gains (0.6 / 0.8).

Intentional EMG level reversals no longer latch the current backend configuration.
Stable input, active pose hold/teleop and gain-write rate limits still apply.
The current EMG configuration allows a two-write burst and replenishes one write
every 2 seconds. A level must persist for 1 second, and the controller waits
1.5 seconds after each completed transition. Actual updates can take longer than
2 seconds because the driver must finish its write and readback first.
Last verified gains change only after device readback confirms an applied pair;
see Pending / blocked reason when the requested level has not yet been applied.

For EMG, connect the band and use the existing Live Classifier tab: load a trusted
model, review compatibility, enable stabilization and apply its settings, and
start classification. Select EMG in Robot control (Start if paused). The same app
manages prediction sending; there is no separate Send checkbox or second window.
Selecting Manual drains pending operations and selects 1/1; a paused session needs
Start. Acquisition, recording, collection, training/model workflows, prediction
logs and stabilization remain available.

The usual `python -m gforce_emg_ui` entrypoint opens these integrated controls.
`--robot-control HOST:PORT` selects the local coordinator; existing `--robot-udp`
also sets this target. Default is 127.0.0.1:6731 and right hand (serial
`C956943BC957`). For the left hand, pass `--hand left` to both the backend and GUI.
The GUI refuses state from the other hand. Fake sessions use backend `--hand right --mode fake` and GUI
`start_gui.sh --robot-control 127.0.0.1:6732`; no band or simulator is needed for
manual tests.

The coordinator and SDK ownership remain in shared `ws_fanuc`; the GUI imports
neither ROS nor Sharpa's SDK. Robot control and existing EMG packets share one
localhost UDP receiver. The authoritative operator guide is
`~/ws_fanuc/src/dual_sharpa_wave_ros2/docs/compliance_control.md`.

Compatibility modes remain explicit: `--legacy-robot-output` opens the previous
bridge controls; `--emg-producer` opens prediction-only controls. Neither is the
normal workflow. Do not run the old bridge alongside the new coordinator.
Current three gain pairs are development settings, not calibrated stiffness;
SDK write limits/faults still require hardware validation.

## Setup

Requires Python ≥ 3.10 (3.12–3.14 for the Live Classifier) and git. `sensor-sdk` ships wheels for **Linux x86_64/arm64,
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
py -3.13 -m venv .venv                # 64-bit python.org installer; 3.12–3.14 for the Live Classifier
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
`--sim-contact {default,good}` (simulator only; `good` = all electrodes good),
`--robot-udp HOST:PORT` (prefills the Live Classifier's robot output target, e.g.
`127.0.0.1:6731` or `[::1]:6731`; never arms it, see [Robot output](#robot-output-sharpa-stiffness)).

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
| **Live Classifier** | Frozen effort model's live output s = 0 / 0.5 / 1 (or `s = --`), class probabilities, history, host-side timings, optional robot output (UDP) to the Sharpa stiffness bridge; see below |
| **Status bar** | Actual vs nominal rates (red warning on mismatch), lost packets/samples, index gaps, segment, callback gap p95, host-only callback→draw age; sensor→host latency is reported as not measured; red **ROBOT OUTPUT ARMED → ip:port** on every tab while the robot output is armed |

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

**Folder mode** (default; Plan / Manual are hidden): *1. Folder* → *Choose…*, *2. Stiffness*
→ 0 or 1, *Start trial*. Same protocol, files and review. Attempts are saved directly in the
folder (simulation: `<folder>/simulated/`). `trial.json`: 0 = `low` (s 0.0), 1 = `high`
(s 1.0); object / posture `unspecified`, stationary, role `development` without a plan;
participant / session = parent / folder name; trial IDs m001, m002, …

**Protocol** (durations editable under *Protocol & plan settings*, saved per trial):
PREPARE 5 s → SETTLE 3 s → **HOLD 10 s** (only training candidate) → REST 3 s, then stop,
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

**Plan** (*Mode → Plan*, *New…*): 200 accepted development trials per participant (+30
Surface Press, +35 Open hand: below), seeded order, no target run > 2, saved before the first
trial, resumed in later sessions. Each
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

**Edit the next trial** (Plan mode, idle): Object, Target and Movement show the random pick.
Changing one changes only that field, for the next trial only; the app then uses a remaining
planned trial with exactly that object / target / movement (lowest plan index; its trial ID
may change, the plan file never does, the displaced pick stays pending). Posture follows the
object. No match: Start stays disabled with the reason; change a field or *Reset edits*
(back to the original pick). An explicitly chosen object may be unticked (warned, this trial
only). Edits end with Accept / Reject and when the mode, participant, session, plan or
output changes; Repeat keeps them. Example: pick `t045 obj03 / medium / reach_sweep`, Object →
obj02 gives e.g. `t017 obj02 / medium / reach_sweep`.

`trial.json` `selection.mode`: `automatic` (missing in older files) or `manual_override`, with
`rule`, `automatic_proposal` (trial, values, how it was drawn), `edited_fields`,
`differing_fields`, `enforced_fields`, `requested`, `resolved`, `available_objects`,
`availability_bypass`, `ordering_preference_bypass`, `recent_targets`, `repeat`.
Schema / protocol versions unchanged.

### Surface Press

Task condition `surface_press` (*Surface Press (Desk)*): the index fingertip presses on a
desk, always stationary, posture `index_finger_press`. Same protocol, recorder, labels
(low/medium/high = s 0/0.5/1, instructed effort, not measured force) and review as the
grasp trials; only the instructions differ:

| phase / target | instruction (shown and saved in `trial.json`) |
|---|---|
| PREPARE | fingertip lightly on the marked area, slight finger bend, steady wrist, forearm supported without pressing the armband, no body weight on the finger |
| LOW | light contact, no intentional downward pressure |
| MEDIUM | comfortable, repeatable moderate press, same finger and wrist position |
| HIGH | firm, comfortable, repeatable press; not maximal, no body weight |
| SETTLE / HOLD / REST | establish / keep / release the pressing effort |

**Add the trials to an existing plan** (once): select the plan → *Protocol & plan settings* →
**Add 30 Surface Press Trials** → confirm (shows plan ID and totals, e.g. 200 → 230).
Appends 10 low, 10 medium, 10 high (`t201`–`t230`, or the next free IDs), seeded order. The
plan ID, the first 200 trials, accepted recordings and progress stay as they are. The old file is copied
to `plan_backups/` first, then replaced atomically; a failed write leaves it unchanged.
A second click only says the plan is already extended; non-standard plans are refused.
Extended plans are `schema_version` 3, so older app versions refuse to load them.

**Record:** tick **Surface Press (Desk)** under *Available objects* to let the random pick
choose remaining surface-press trials (unticked: they stay in the plan, not picked).
Or set *Object* to `surface_press` for the next trial: the target is kept and the movement
is set to stationary (another movement is refused). Changing only the target of a proposed
surface-press trial keeps the condition. Progress shows
`Surface Press: LOW x/10 · MEDIUM y/10 · HIGH z/10 · total n/30` (accepted only).

`trial.json` adds `condition.trial_type` (`surface_press` or `grasp`) and
`label.task_instruction_version` (`surface-press/1`, `grasp/1`); raw CSVs are unchanged.
The classifier (`emg-stiffness-data`) reads these trials and does not filter by object:
accepted presses train with the grasps under the same LOW/MEDIUM/HIGH labels.

### Open hand

Task condition `open_hand` (*Open hand (no object)*): like empty hand, but the hand stays
comfortably open (no fist, no object), posture `open_hand`, all four movements. Same
protocol, recorder, labels and review. LOW is instructed relaxation; MEDIUM / HIGH are
forearm contractions with the hand still open (not measured force or co-contraction).

| phase / target | instruction (shown and saved in `trial.json`) |
|---|---|
| PREPARE | hand comfortably open, natural slight finger curve (not forced straight or spread), consistent arm position, nothing pressed; stationary: forearm may be supported without pressing the armband |
| LOW | relax hand and forearm as far as the posture allows; no tensing, squeezing, pressing or spreading |
| MEDIUM | comfortable, repeatable moderate forearm contraction; hand open, no fist, nothing pressed |
| HIGH | firm, comfortable, repeatable contraction, same open hand; not maximal, joints not forced straight |
| SETTLE | LOW: relax; MEDIUM / HIGH: establish the contraction (moving: start the movement) |
| HOLD / REST | keep the state with the hand open (moving: keep moving, effort unchanged) / release and rest |

Stationary LOW = quiet relaxation. Moving LOW = no added hand or grip tension while moving;
the movement itself still uses forearm muscles. `motion_condition` keeps them apart in the
data. A fist during MEDIUM / HIGH means the trial was not performed as instructed: Reject or
Repeat with *Wrong posture*.

**Add the trials** (once): select the plan → *Protocol & plan settings* → **Add 35 Open-Hand
Trials** → confirm. The dialog shows the plan ID, the actual totals (200 → 235, or 230 → 265
after Surface Press) and the allocation (15 / 10 / 10 in total, split evenly over the plan's
movements, so only 4 stationary relaxed LOW):

| movement | LOW | MEDIUM | HIGH |
|---|---|---|---|
| stationary | 4 | 3 | 2 |
| wrist flex/ext | 4 | 2 | 3 |
| forearm rotation | 4 | 2 | 3 |
| reach/sweep | 3 | 3 | 2 |

New trials get the next free IDs (e.g. `t231`–`t265`) in a seeded order; earlier trials,
extension records, recordings and progress stay as they are; backup
`plan_backups/<plan>.before-open-hand.json`. Either order with Surface Press works, each
once. Plans with open hand are `schema_version` 4 (older app versions refuse them);
inconsistent or partly extended plans are refused with the reason, never repaired.

**Record:** tick **Open hand (no object)** (opt-in, like Surface Press) or set *Object* to
`open_hand`: target and movement are kept, posture becomes `open_hand`; without a remaining
trial for that target / movement Start stays disabled with the reason. Progress shows
`Open hand: LOW x/15 · MEDIUM y/10 · HIGH z/10 · total n/35` (accepted only).

`trial.json`: `condition.object_id` / `trial_type` / `posture_id` = `open_hand`, the actual
`motion_condition`, `label.task_instruction_version` = `open-hand/1`, the exact instructions
and operator note; same class / s mapping and `label_source`. The live model does not learn
from new recordings: copy accepted trials to `emg-stiffness-data` and retrain before they can
affect live predictions (its reports list `open_hand` as its own object).

```
<data-dir>/stiffness_collection/P01/
  plan_plan-1a2b3c4d.json
  plan_backups/plan_plan-1a2b3c4d.before-surface-press.json   (and .before-open-hand.json)
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

## Research data in git

`data/` (recordings, `stiffness_collection/`, plans, backups, `trial.json`, CSVs, binary
recordings) is no longer git-ignored; credentials, `.env`, venvs, logs, `live_logs/` and
models stay ignored. Un-ignoring uploads nothing: add data explicitly.

```bash
GH_CONFIG_DIR=~/erdem/.config/gh gh repo view --json visibility   # must be PRIVATE
git status --short data/
git add data/stiffness_collection/<P>/   # review first; never `git add .`
git commit -m "Add <P> trials" && git push
```

Private is not anonymous: participant IDs (folder names, `trial.json`) and the band's MAC
address (`session.json`) are in the files. Size: ~2.6 MB per trial (~0.5 MB compressed),
largest file 11 MB; 265 trials ≈ 0.7 GB raw. Plain git is fine at this size; consider
Git LFS if single files approach 50 MB or the repo grows to several GB.

## Live classifier

Shows the frozen [emg-classifier](https://github.com/ErdemTel/emg-stiffness-data)
model's prediction of the **instructed effort** class from the live EMG: s = 0 (low),
0.5 (medium), 1 (high), or `s = --` when there is no fresh valid prediction. Not measured
stiffness. Low was trained as minimal effort *with the grasp kept* (the 77-trial MLP below
adds a few open-hand and surface-press trials with their own LOW); rest and a relaxed/open
hand are not validated live. Display and logs only, unless you arm the optional
[robot output](#robot-output-sharpa-stiffness): then the shown value is also sent to the
Sharpa stiffness bridge, which decides what the hand does.

Install the classifier into this `.venv` (pins = the model's training environment; nothing
else is upgraded). Clone the private data repo next to this one
(`git clone https://github.com/ErdemTel/emg-stiffness-data.git`, branch `main`; it contains the
models `runs/lda_hold_001/model.joblib` (LDA) and `runs/mlp_hold_30w_seed42_001/model.joblib`
(MLP); both are accepted), then from this folder (Windows: same
commands in PowerShell with `.venv\Scripts\pip`). Needs a 64-bit Python 3.12–3.14 `.venv` (the pinned
numpy / scipy have no older wheels; pip then reports a numpy "conflict"):

```bash
git --git-dir=../emg-stiffness-data/.git archive -o ../clf.tar main pyproject.toml README.md src
mkdir ../clf
tar -xf ../clf.tar -C ../clf
.venv/bin/pip wheel --no-deps -w ../clf/wheels ../clf
.venv/bin/pip install -c constraints-live.txt ../clf/wheels/emg_classifier-0.1.0-py3-none-any.whl
```

Which model: the current best is the MLP
`../emg-stiffness-data/runs/mlp_hold_30w_77trials_seed42_001/model.joblib`
(`short_window_mlp`, 100-sample windows / 25-sample hop, 77 trials / 2310 windows incl.
open-hand and surface-press trials, scikit-learn 1.9.1; development CV within one session:
window BA 0.948, trial BA 0.987, not a live guarantee). Use it for live sessions and robot
output; after loading, the *Model* box must say *fitted on 77 trials / 2310 windows*. The
older `lda_hold_001` and `mlp_hold_30w_seed42_001` (51 trials) stay accepted and are the
files the real-model tests pin (sha256, trial counts): do not point the test variables
below at the 77-trial model.

Two-class model (low / high only, ch6 left out: poor contact in S2.1):
`../emg-stiffness-data/runs/mlp_s21_7ch_binary_155w_seed42_001/model.joblib` (`train-subset`,
20 S2.1 trials, 155 windows each = the whole trimmed 10 s HOLD, 3100 windows; development CV
20/20 trials, window BA 0.993; leave-one-block-out 20/20, 0.987). The 30-window
`mlp_s21_7ch_binary_seed42_001` scores the same. The engine feeds only the model's channels;
bars, history ticks and the soft score follow the model's classes (no medium). Needs a
classifier wheel with channel / class subsets.

Use: connect the band → **Live Classifier** → *Load model…* (a trusted local
`model.joblib`; joblib/pickle runs code, nothing loads automatically) → review
*Compatibility* (incompatible settings block Start; warnings need a tick, which covers Start
and logging only) → *Start* → warm-up → comfortable low / medium / high effort → *Stop*.

- Windows and hop come from the model (sample counts, e.g. 100 / 25); features and
  probabilities come from the classifier's own `predict_windows`. Raw output: no smoothing,
  voting, rest gate or thresholds. Optional output stabilization (below) only changes the
  large displayed value (which is also what an armed robot output sends); the raw output
  stays visible and logged.
- Lost/invalid/non-finite samples, index gaps, segment or stream changes, dropped input and
  stale input (> 0.5 s, to verify on the band) reset the window; a reconnect resets it,
  a disconnect or a filter/rate change stops it.
- *Log predictions* writes `predictions.csv`, `stabilization.csv`, `events.csv`,
  `display.csv` and `run.json` to a new folder under `live_logs/` (git-ignored). No labels
  are recorded.
- Timings shown are host-side only (not sensor-to-display latency).

### Output stabilization

Display/output processing after the frozen classifier, to keep brief class excursions
(e.g. LOW → MEDIUM → LOW while curling the fingers) from changing the shown command. Not
measured stiffness, not a validated robot safety controller. OFF by default.

Enable: *Output stabilization* → tick *Enable stabilization* → *Apply settings*. Edits
are a draft ("Unapplied changes") until *Apply settings*; they can be changed while
classifying. Apply sets a new settings revision, clears the stabilizer, and uses only
predictions received afterwards (no replay); the model, buffer and recording continue.
Applied settings persist per user (Linux: `~/.config/gforce-emg-ui/live-classifier.ini`).
*Restore defaults* fills the controls (Enable kept); *Reset state* clears the stabilizer
only. Raw behavior again: untick *Enable stabilization* → *Apply settings*.

Algorithm (`live/stabilization.py`, persistence confirmation v1): a raw winner different
from the shown class becomes a candidate. It is accepted when that same candidate has won
every processed prediction, without a break, for the confirmation time: upward or
downward by s (0 < 0.5 < 1), or *startup* while no class is accepted (after Start, any
reset, a settings change, invalid input or an expired hold; `s = --` until then, never
LOW by default). LOW ↔ HIGH may change directly; nothing skips confirmation; 0 ms =
accept on the first prediction. A different candidate, a return to the shown class, a
failed probability gate, a skipped/expired window, a window-index jump or a receipt gap
> *max evidence gap* restarts the candidate from zero (bursts are never added up).

Time = host receipt time of each window's newest sample (`newest_receipt_mono`):
host-observed persistence, not acquisition time or physiological latency. Predictions
with the same receipt time add 0 ms; slow inference or GUI refresh adds nothing; only a
new prediction can accept a class. With the dongle's ~150 ms bursts, 100 ms in practice
means "still the winner in the next burst". The confirmation adds to window, transport
and inference delay.

| setting | default | range |
|---|---|---|
| Upward / downward / startup confirmation | 100 ms | 0–1000 ms |
| Probability gate (advanced) | off | – |
| Min. candidate probability | 0.60 | 0–1 |
| Min. probability margin (vs. the shown class) | 0.10 | 0–1 |
| Max. evidence gap | 250 ms | 1 ms – stale limit (500 ms) |
| Max. unsupported hold | 1000 ms | 100–10000 ms, ≥ longer confirmation + gap |

Defaults are initial engineering values, not validated optima. Probabilities are
classifier scores, not calibrated certainty. The gate applies to candidates only; the
shown class is "supported" by any prediction whose raw winner it is. If predictions keep
arriving but none supports the shown class for longer than the hold, the output becomes
unavailable ("unsupported hold expired"; separate from stale input). Invalid predictions,
stale input or predictions, buffer resets, stream/segment changes, Stop, disconnect,
model change and errors clear the stabilizer: `s = --`, never a fallback s = 0.

Display: large value = *Stabilized command* (ON) or *Raw output — stabilization OFF*;
the raw class, the RAW probability bars and a status line ("Holding LOW; MEDIUM pending:
50 / 100 ms") stay visible. History: raw s (dotted), stabilized s (thick), raw soft score
(dashed), each can be hidden; gaps are kept; grey lines mark resets and settings changes.

Logs (with *Log predictions*): `predictions.csv` unchanged (raw).
`stabilization.csv`: one `observe` row per processed prediction (join on `run_id` +
`pred_id`; raw class, probabilities, revision, accepted class/s or empty, `output_valid`,
pending class and evidence ms, last support, `accepted_pred_id` of the prediction that
established the shown class, evidence and decision times, transition, reason) plus
`settings`, `reset`, `break` and `ignored` rows. `events.csv`: `stabilization_settings`
(revision + settings JSON), `stabilization_reset`. `run.json`: initial settings,
algorithm version, end counters (raw class changes, accepted changes, cancelled
candidates, gate failures, hold expiries). No ground truth: rejected transitions are not
"corrected errors".

Limits: brief intentional commands are delayed or rejected too; sustained
misclassification is not repaired; synthetic tests and the simulator say nothing about
real human EMG.

### Robot output (Sharpa stiffness)

**Legacy integrated bridge workflow.** For current operation use the one-GUI workflow above.
The controls in this section are available explicitly with `--legacy-robot-output`.

Optional, and off at every start (the armed state is never stored). While armed, a sender
thread (`live/robot_out.py`, `live-robot-out`) streams the value this tab shows (the
stabilized command, raw output while stabilization is OFF, or manual selection) as UDP datagrams to the
robot bridge `emg_stiffness_bridge` of the parent repository
(`ros_ws/src/emg_stiffness_bridge`), which may turn it into Sharpa Wave MIT gains. This app
has no ROS or driver dependency; the bridge decides what the hand does. The separate
**Stiffness gains** section can configure the bridge's level multipliers while disarmed.

Enable EMG output (the default **Stiffness source → EMG classifier**):

1. Robot PC: start the bridge for one hand (it starts disarmed and in dry
   run; see its README and `docs/deployment.md` in the parent repository).
2. Here: connect the band → *Load model…* → *Start* → *Output stabilization*: *Enable
   stabilization*, every confirmation ≥ 100 ms → *Apply settings*.
3. *Robot output (UDP)*: *Robot IP* / *UDP port* of the bridge → tick *Send to robot* →
   confirm the target in the dialog. The status bar shows **ROBOT OUTPUT ARMED → ip:port**
   on every tab until it is disarmed. Then arm the controller on the robot PC: a new UI
   session disarms an already-armed bridge.
4. Untick *Send to robot* to stop (the last datagram is a `stop`).

In EMG mode, *Send to robot* is enabled only with a model, the band streaming, the classifier running, a
valid target, and a Compatibility check without warn / unknown items on a real band.
Warnings, unknowns or the simulator need the extra tick *Send despite the warnings /
simulator* (the Start acknowledgement does not count); it is cleared whenever the check
changes. Incompatible settings cannot be overridden. The box lists what is missing and
what the bridge will ignore with its defaults (*robot will ignore: stabilization OFF*,
*robot will ignore: simulated*).

In EMG mode it disarms by itself (the sender's last datagram is `stop`; re-arm by hand): model load or
replacement, *Apply settings* (also switching stabilization off: the robot never gets a new
meaning of s without a new arm), *Stop* or the run stopping by itself, the band leaving
*streaming* (disconnect, link lost / reconnecting, failed), closing the app.

| setting | default | range |
|---|---|---|
| Robot IP | `127.0.0.1` (bridge on this PC) | IPv4 / IPv6 literal; host names are never resolved; multicast, broadcast, `0.0.0.0` / `::` refused |
| UDP port | 6731 | 1–65535 |
| Send rate | 20 Hz | 5–50 Hz (`rate_hz` in the stored settings, not in the tab) |

The target persists per user (`robot_output/settings` in `live-classifier.ini`) when you
arm or finish editing a valid one. `--robot-udp HOST:PORT` prefills it for one session and
never arms.

**Manual stiffness (no band or model):** in *Live Classifier → Stiffness source*, select
**Manual (classifier off)**. This stops the classifier and any outgoing stream. Select
**0 · Low**, **0.5 · Medium**, or **1 · High**, then tick **Send to robot** and confirm the
target. Arm the bridge after starting the UI stream. While sending, clicking another level
requests that level through the same bridge and gain table. The large value shows the
**selection**; the Robot output status shows the robot's **applied s** from readback.

The bridge must be launched with **`accept_test_sender:=true`** for manual output.
Manual commands use the existing `source: send_stiffness` path with `simulated: false`,
even if the app was opened with `--simulate`: these are explicit operator commands, not
synthetic EMG. The confirmation dialog explains this. Default `dry_run: true`, robot-side
arming, active-command gating, write budgets, dwell times and fault checks still apply.
No need to disable stabilization checks: a new manual selection is debounced for **100 ms**
(`stabilized: true`, `stab_ms: 100/100/100`), with `s: null` while waiting. This describes
button confirmation, not model evidence; manual packets have no prediction or run IDs.
Bridge configurations requiring a confirmation longer than 100 ms will reject these commands.

Uncheck **Send to robot** to stop output, or change the source selector. Neither action
disables motors or restores different gains: the bridge holds the last applied gains.
Switching sources stops sending, clears the manual selection, and leaves the classifier
stopped. Returning to EMG requires **Start**, **Send to robot**, then robot-side arming.
Manual mode continues independently of EMG disconnects. Closing the app sends `stop`.
Mode, selected level and sending state are never restored on app restart.

**Manual Kp/Kd (live tuning):** choose the third *Stiffness source* option,
**Manual Kp/Kd (classifier off)**. No band or model is needed. The two controls multiply
each joint's baseline gain: `Kp_i = BASE_KP_i × Kp multiplier`, and likewise for Kd.
They do not modify the three presets. For example, **0.55 / 0.75** requests 55% of each
joint's baseline Kp and 75% of its baseline Kd.

1. Enter both multipliers and click **Apply gains** to select the pair locally.
2. Enable **Send to robot**, confirm the target, then arm the bridge.
3. During sending/teleop, edit the fields and click **Apply gains** again. Editing alone
   leaves the previously selected pair streaming. No stop or re-arm is needed per edit.
4. Check **robot: applied Kp / Kd** for readback. The large pair is the selected command;
   the fields are a draft. In dry run the applied pair stays at the actual device gains.

Both multipliers independently accept **0–1 inclusive**, with four decimal places in
the fields. Typing updates only the draft and enables **Apply gains** without needing
to press Enter or leave the field first. A click sends the complete pair while
**Send to robot** stays checked; the previous pair continues streaming until that click.
Empty/incomplete drafts disable Apply. The bridge independently rejects non-finite or
out-of-range pairs. The separate three-level preset editor retains its bounds and
damping-ratio constraints. Commands
use `emg-gains/1`, `source: send_stiffness`, and the same 100 ms debounce, command lease,
arming, persistence, rate limits, write budget and fault checks. Crossing the medium Kp
multiplier passes through the medium preset first. Repeated heartbeats do not rewrite gains.
Arbitrary pairs have no low/medium/high tone; use the applied multipliers as confirmation.

**Zero is an actual gain:** Kp=0 removes the position-stiffness term and Kd=0 removes the
damping term; neither turns off the motors. Restore positive Kd before using the shared
driver's default-pose recovery, which refuses zero damping. These wider pairs have not
been validated on hardware by the software tests.

**Without teleop:** start the bridge with `allow_manual_gains_when_idle:=true` in addition
to `hand:=right dry_run:=false accept_test_sender:=true` and the session's feedback-gap
setting. Leave the hand driver running, select Manual Kp/Kd, Apply the initial pair,
enable Send and arm the bridge. Further Apply clicks work without Quest or a pose-holding
GUI. Feedback may correctly show `teleop_active=false`; check applied Kp/Kd for success.
This changes gains only and does not send joint positions; the hand retains its existing
motor target. Changing gains can therefore change its physical pose. Fresh feedback,
normal driver state and other bridge/driver checks remain required. The option defaults
to false; it does not exempt EMG or manual 0/0.5/1 commands from the position-stream check.

Build and restart the updated bridge and restart this GUI before using this mode. Older
bridges reject `emg-gains/1`. The existing **Stiffness gains** preset editor still requires
Send off and a disarmed bridge. Mode and selected pair are never restored at startup.

**Stiffness gains:** six multiplier controls map levels 0, 0.5 and 1 to Kp/Kd scales.
Initial pairs are **0.40/0.65**, **0.60/0.80**, and **1.00/1.00**. Each scale multiplies
the hand's joint-specific baseline array; these are not identical absolute gains for all
22 joints.

1. Start the updated bridge for the selected hand. Keep *Send to robot* off and disarm the
   bridge using the robot-side controls.
2. In *Stiffness gains*, select the hand and click **Read bridge**. The panel displays the
   actual active table and whether the bridge is armed. Reading does not overwrite a draft.
3. Edit the six values, or use **Use bridge values** / **Defaults** to fill the draft.
4. Click **Apply table**. Wait for the bridge's confirmation. This updates the mapping and
   configuration hash without writing gains to the device. After re-arming, future level
   commands use the new mapping.

Invalid ranges, decreasing levels, duplicate gain pairs and incompatible damping ratios
are rejected locally and by the bridge. Edits cannot be applied while output is sending,
the bridge is armed/arming, a write/monitor operation is pending, or hand recovery is active.
A stale configuration is rejected; read it again before applying. Applying never clears a
fault latch, resets a write budget, enables sending or arms the bridge.

Drafts persist under `stiffness_gains/draft` in the per-user settings. A bridge restart
reloads its launch/YAML configuration: read and reapply the saved draft, or edit the robot's
startup YAML to retain it there. The GUI displays active bridge values separately from the
draft and refreshes them every three seconds after the first read. Before that read, the
panel sends nothing. Configuration uses a separate `emg-stiffness-config/1` exchange on
the same robot IP/UDP port; it does not interrupt the stiffness-command lease.

Protocol `emg-stiffness/1` (normative: `docs/design_emg_stiffness_2026-11-02.md` §5.1–5.2
and `docs/protocol/emg-stiffness-v1-golden.json` in the parent repository; an identical copy
is `tests/data/emg-stiffness-v1-golden.json`): one UTF-8 JSON object per datagram
(≤ 2048 bytes); a `heartbeat` every 1/rate s, and a `change` in the next period after the
shown s or stabilization ON/OFF changed (≤ 50 ms at 20 Hz). Keys: `proto`, `source`
(`gforce-emg-ui`), `session` (new random id per arm), `seq` (+1 per datagram, written by the
sender thread only), `kind` (`heartbeat` / `change` / `stop`), `s` (0, 0.5, 1 or `null`),
`valid`, `stabilized`, `simulated` (`--simulate`), `sent_wall`, `sent_mono`, `stab_ms`
(`up` / `down` / `startup` ms, when stabilized), and for diagnostics `age_ms`, `status`,
`reason`, `run_id`, `revision`, `accepted_pred_id`, `last_pred_id` (join the run's
`predictions.csv` / `stabilization.csv`). The bridge answers from its port with
`emg-stiffness-state/1` (~2 Hz and on change). The box's status line shows sent, errors,
last s and the robot's applied s, phase, armed and dry-run state, or *robot state unknown*
when none arrived for 2 s.

Invalid output and timeouts: `s = --` is sent as `"s": null, "valid": false`, never as 0;
the bridge then holds the gains it has (no write), and `stop` also means hold. If datagrams
stop (app closed or crashed, network), the bridge holds after its command timeout (1 s) and
frees the session after 2 s of silence. The UI never substitutes a level. Send errors
(unreachable network, full buffer) are counted in the status line, never raised; nothing
here blocks the GUI or the classifier.

What the bridge enforces (its defaults): commands from the one leased session only;
stabilization ON with every confirmation ≥ 100 ms; simulated EMG and the `send_stiffness`
test tool ignored; dry run (decisions published, no gain writes) until switched off after
the hardware validation; a level must persist 1 s before it is requested; rate limits and
no direct 0 ↔ 1; a failed write latches an error that only a re-arm on the robot side clears.

Model limits: trained on steady HOLD windows only; contraction / release transitions,
end-to-end latency and rest are not evaluated (REST was never a class); probabilities are
uncalibrated classifier scores; s is the instructed effort class, not measured stiffness.

Network: the default is this app on the robot PC over loopback. From another PC use a
non-robot network (not the hands' 192.168.10.x), add the UI PC to the bridge's allowed
sources and let UDP 6731 through the robot PC's firewall.

Tests: `.venv/bin/python -m pytest tests/test_live_robot_out.py` (Qt-free: golden-vector
conformance, loopback UDP, test-double classifier, fake clocks; no robot) and
`.venv/bin/python -m pytest tests/test_live_ui.py -k robot` (group box, arming and
auto-disarm rules, status-bar indicator; test-double model and loopback receiver).

Disable: `.venv/bin/pip uninstall emg-classifier` (the tab then says so; recording is
unaffected). Remove: delete `src/gforce_emg_ui/live/`, `ui/live_view.py`,
`tests/test_live_*.py`, `tests/data/emg-stiffness-v1-golden.json`, `constraints-live.txt`,
the `live` / `robot` lines in `ui/main_window.py` and `--robot-udp` in `__main__.py`.

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
                 ├──► LiveEngine (bounded inbox → worker: windows → frozen model;
                 │                watchdog; latest result read by the Live Classifier tab
                 │                and, only while armed, by the robot sender thread
                 │                ─UDP emg-stiffness/1─► robot bridge)
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
.venv/bin/python -m pytest tests/test_collection_selection.py   # next-trial edits
.venv/bin/python -m pytest tests/test_collection_surface.py     # surface press, plan extension
.venv/bin/python -m pytest tests/test_collection_open_hand.py   # open hand, both extensions
.venv/bin/python -m gforce_emg_ui.bench        # host-only benchmark (add --quick, --json out.json)
.venv/bin/python -m pytest tests/test_live_*.py           # live classifier
.venv/bin/python -m pytest tests/test_live_stabilization.py -s   # + replay demo counts
.venv/bin/python -m pytest tests/test_live_robot_out.py   # robot output (loopback UDP, no robot)
.venv/bin/python -m gforce_emg_ui.live.bench --model PATH # live classifier, simulated bursts
```

Real-model live tests use `../emg-stiffness-data/runs/lda_hold_001/model.joblib`,
`../emg-stiffness-data/runs/mlp_hold_30w_seed42_001/model.joblib` and
`../emg-stiffness-data/Erdem` (override with `GFORCE_LIVE_MODEL`, `GFORCE_LIVE_MLP_MODEL`,
`GFORCE_LIVE_DATA`) and are skipped when absent. They check those files' identity, so keep
them on the 51-trial models (not the 77-trial MLP recommended above).

The benchmark uses simulated data only. It does **not** measure the band, BLE throughput or
end-to-end sensor latency.

## Reference

The vendor SDK repo (`oymotion/SynchroniSDKPython`, branch `Testing`) is kept alongside this repo at
`../SynchroniSDKPython` for its API docs (`README.md`) and the reference demo `examples/pyqt_demo.py`.

Project context and progress log: [`memory.md`](memory.md).
