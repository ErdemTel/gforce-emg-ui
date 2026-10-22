"""Compare the live acquisition setup with the model's saved acquisition assumptions.

Levels: ok; block (known incompatibility: Start disabled, never acknowledgeable);
warn / unknown (differs with unknown effect, or not reported live: Start needs an
explicit acknowledgement and the warning stays visible). The nominal-rate vs host
receipt throughput discrepancy (~505 rows/s at a reported 1000 Hz) is shown, never
treated as a new incompatibility. Nothing here changes device settings.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .engine import HUB_CHANNELS

OK, WARN, UNKNOWN, BLOCK = "ok", "warn", "unknown", "block"

# field -> (label, level when different, note)
RULES = {
    "emg_channels": ("EMG channels", BLOCK, ""),
    "emg_channel_order": ("channel order", BLOCK, ""),
    "emg_rate_requested_hz": ("EMG rate requested", BLOCK, "nominal setting"),
    "emg_rate_readback_hz": ("EMG rate read back", BLOCK, "nominal setting"),
    "emg_rate_reported_by_stream_hz": ("EMG rate reported by stream", BLOCK, "nominal"),
    "onboard_filters": ("on-board filters", BLOCK, "filters change the signal"),
    "device_model": ("device model", WARN, "not in the training data; effect unknown"),
    "firmware": ("firmware", WARN, "not in the training data; effect unknown"),
    "hardware": ("hardware", WARN, "not in the training data; effect unknown"),
    "package_samples": ("samples per packet", WARN, "batching only; effect not verified"),
    "backend": ("BLE backend", WARN, "transport only; effect not verified"),
}


@dataclass(frozen=True)
class Check:
    field: str
    label: str
    expected: Any
    live: Any
    level: str
    note: str = ""


@dataclass(frozen=True)
class CompatReport:
    config_key: Optional[str]
    checks: Tuple[Check, ...]
    info: Tuple[str, ...]
    fingerprint: tuple              # live values that, if changed during a run, stop it

    @property
    def blocking(self):
        return [c for c in self.checks if c.level == BLOCK]

    @property
    def needs_ack(self):
        return [c for c in self.checks if c.level in (WARN, UNKNOWN)]

    def to_dict(self) -> dict:
        return {"config_key": self.config_key, "info": list(self.info),
                "checks": [c.__dict__ for c in self.checks]}


def live_acquisition(device: dict, params: dict, source, hub) -> dict:
    """Live values in the classifier's acquisition-summary layout; None = not known yet."""
    device = device or {}
    params = params or {}
    streaming = bool(getattr(hub, "last_emg_host_mono", 0))
    return {
        "device_model": device.get("model") or None,
        "firmware": device.get("firmware") or None,
        "hardware": device.get("hardware") or None,
        "emg_channels": device.get("emg_channels") or None,
        "package_samples": device.get("package_samples") or None,
        "backend": device.get("backend") or (source.backend_name() if source else None) or None,
        "emg_rate_requested_hz": getattr(source, "target_emg_rate", None),
        "emg_rate_readback_hz": params.get("emg_rate"),
        "emg_rate_reported_by_stream_hz": float(hub.emg_rate) if streaming else None,
        "onboard_filters": dict(params["filters"]) if params.get("filters") else None,
        "emg_channel_order": list(HUB_CHANNELS),
    }


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return float(a) == float(b)
    return a == b


def _checks(summary: dict, live: dict):
    out = []
    for f, (label, level, note) in RULES.items():
        exp, cur = summary.get(f), live.get(f)
        if f == "onboard_filters":
            exp = exp or {}
            for k in sorted(set(exp) | set(cur or {})):
                e, c = exp.get(k), (cur or {}).get(k)
                lv = UNKNOWN if c is None else OK if e == c else BLOCK
                out.append(Check(f"{f}.{k}", f"filter {k}", e, c, lv,
                                 "not read back yet" if c is None else
                                 "" if lv == OK else note))
            continue
        if exp is None:
            out.append(Check(f, label, None, cur, UNKNOWN, "not saved with the model"))
        elif cur is None:
            out.append(Check(f, label, exp, None, UNKNOWN, "not reported live yet"))
        else:
            out.append(Check(f, label, exp, cur, OK if _same(exp, cur) else level,
                             "" if _same(exp, cur) else note))
    return out


def _score(checks):
    return (sum(c.level == BLOCK for c in checks), sum(c.level in (WARN, UNKNOWN) for c in checks))


def check_compatibility(model_acquisition: dict, live: dict, simulation: bool,
                        receipt_rate: Optional[float] = None) -> CompatReport:
    """Best-matching saved configuration (fewest blocks, then fewest warnings)."""
    configs = model_acquisition.get("configurations") or {}
    best = None
    for key in model_acquisition.get("keys") or sorted(configs):
        summary = (configs.get(key) or {}).get("summary") or {}
        checks = _checks(summary, live)
        if best is None or _score(checks) < _score(best[1]):
            best = (key, checks)
    key, checks = best if best else (None, [])
    if simulation:
        checks.append(Check("source", "data source", "real band", "simulator", WARN,
                            "software test only: synthetic EMG, predictions are meaningless"))
    rates = model_acquisition.get("rates") or {}
    thr = (rates.get("host_receipt_throughput_rows_per_s") or {}).get("median")
    info = (
        "Window and hop are sample counts; no rate is used by the features.",
        f"Training: nominal {rates.get('nominal_requested_or_reported_hz')} Hz, host receipt "
        f"throughput median {thr:.1f} rows/s." if thr else "Training throughput not saved.",
        *([f"Live host receipt throughput: {receipt_rate:.1f} rows/s."] if receipt_rate else []),
        "Neither is a verified physical acquisition rate (nominal vs throughput mismatch is "
        "the documented, supported setup).",
    )
    fingerprint = tuple(sorted((k, repr(v)) for k, v in live.items()))
    return CompatReport(key, tuple(checks), info, fingerprint)
