"""EMG features. Invalid samples are NaN and are excluded, never treated as zeros."""

import warnings

import numpy as np

MIN_VALID_FRACTION = 0.5    # fewer valid samples than this in a window -> NaN feature


def _valid_enough(x: np.ndarray, min_fraction: float) -> np.ndarray:
    n = x.shape[-1]
    return np.count_nonzero(np.isfinite(x), axis=-1) >= max(2, int(np.ceil(min_fraction * n)))


def ac_rms(x: np.ndarray, min_valid_fraction: float = MIN_VALID_FRACTION) -> np.ndarray:
    """Per-channel RMS after removing each channel's mean over the window (AC RMS).

    x: (channels, n) with NaN for invalid samples. Channels with too few valid
    samples return NaN.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] == 0:
        return np.full(x.shape[:-1], np.nan)
    ok = _valid_enough(x, min_valid_fraction)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centered = x - np.nanmean(x, axis=-1, keepdims=True)
        rms = np.sqrt(np.nanmean(centered ** 2, axis=-1))
    return np.where(ok, rms, np.nan)


def nan_center(x: np.ndarray) -> np.ndarray:
    """Subtract each channel's mean over valid samples; NaNs stay NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(x, axis=-1, keepdims=True)
    return x - np.where(np.isfinite(m), m, 0)


def trailing_valid_run(x: np.ndarray) -> int:
    """Length of the trailing run of finite samples in a 1-D array."""
    bad = np.flatnonzero(~np.isfinite(x))
    return len(x) if bad.size == 0 else len(x) - 1 - int(bad[-1])


def rms_shares(rms: np.ndarray) -> np.ndarray:
    """Each channel's RMS as % of the summed RMS over channels with a valid RMS.

    Not a percentage of maximum voluntary contraction or of signal power; NaN
    for channels without a valid RMS.
    """
    rms = np.asarray(rms, dtype=np.float64)
    finite = np.isfinite(rms)
    total = rms[finite].sum()
    if total <= 0:
        return np.where(finite, 0.0, np.nan)
    return np.where(finite, 100.0 * rms / total, np.nan)
