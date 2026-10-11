"""Constants for the gForceUltra band and the UI.

Device facts come from the vendor spec (SynchroniSDKPython
`Test Cases/Automation/device_specs/gforce_ultra.py`, firmware V1.0.8):
8-ch EMG at 500|1000 Hz (default 1000), 6-axis IMU (ACC+GYRO) at 50 Hz,
8-ch electrode contact (impedance) at ~1 Hz, on-board filters
FILTER_50HZ / FILTER_60HZ / FILTER_HPF (10 Hz) / FILTER_LPF (200 Hz).
"""

EMG_CHANNELS = 8
EMG_RATES = (500, 1000)
DEFAULT_EMG_RATE = 1000

IMU_CHANNELS = 6                  # NTF_IMU aggregate: ACC x,y,z then GYRO x,y,z
IMU_LABELS = ("ACC-X", "ACC-Y", "ACC-Z", "GYRO-X", "GYRO-Y", "GYRO-Z")

# Samples per BLE data callback batch (SensorProfile.init packageSampleCount).
# Smaller = lower latency but more callback overhead. The vendor demo uses 32
# (= 32 ms of buffering at 1 kHz); we default lower for latency.
PACKAGE_SAMPLE_COUNT = 16
POWER_REFRESH_MS = 30_000
SCAN_PERIOD_MS = 3_000

# On-board filter switches (setParam key -> label)
FILTERS = {
    "FILTER_50HZ": "50 Hz notch",
    "FILTER_60HZ": "60 Hz notch",
    "FILTER_HPF": "High-pass 10 Hz",
    "FILTER_LPF": "Low-pass 200 Hz",
}

# Ring buffer lengths (seconds at the maximum rate)
EMG_BUFFER_SECONDS = 10
IMU_BUFFER_SECONDS = 10

# UI refresh
PLOT_INTERVAL_MS = 33             # ~30 FPS
STATUS_INTERVAL_MS = 500
FFT_INTERVAL_S = 0.5

# Electrode contact thresholds in kOhm, same as the vendor demo:
# <= 500 good (green), <= 999 marginal (orange), otherwise poor (red).
CONTACT_GOOD_KOHM = 500.0
CONTACT_MARGINAL_KOHM = 999.0

CHANNEL_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
)
