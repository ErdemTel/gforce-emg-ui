# gforce-emg-ui

Custom low-latency UI for recording and visualizing the **8-channel EMG** stream of the
OYMotion **gForceUltra** armband, built on the [`sensor-sdk`](https://pypi.org/project/sensor-sdk/) Python package.

## Setup (Ubuntu, Python ≥ 3.10)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The SDK picks a USB BLE dongle automatically if one is present, otherwise BlueZ.

## Reference

The vendor SDK repo (`oymotion/SynchroniSDKPython`, branch `Testing`) is kept alongside this repo at
`../SynchroniSDKPython` for its API docs (`README.md`) and the reference demo `examples/pyqt_demo.py`.

Project context and progress log: [`memory.md`](memory.md).
