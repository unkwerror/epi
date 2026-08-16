import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt


def _array(x):
    x = np.asarray(x)
    if x.ndim not in (2, 3):
        raise ValueError("expected (channels, samples) or (windows, channels, samples)")
    return x.astype(np.float64, copy=False)


def bandpass(x, rate, low=0.5, high=40.0, order=4):
    x = _array(x)
    sos = butter(order, [low, high], btype="bandpass", fs=rate, output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def notch(x, rate, freq=50.0, quality=30.0):
    x = _array(x)
    freqs = np.atleast_1d(freq).astype(float)
    out = x
    for value in freqs:
        if value <= 0 or value >= rate / 2:
            continue
        b, a = iirnotch(value, quality, fs=rate)
        out = filtfilt(b, a, out, axis=-1)
    return out.astype(np.float32)


def normalize(x, eps=1e-8):
    x = _array(x)
    center = np.median(x, axis=-1, keepdims=True)
    scale = np.median(np.abs(x - center), axis=-1, keepdims=True) * 1.4826
    return ((x - center) / np.maximum(scale, eps)).astype(np.float32)


def quality(x, flat_eps=1e-6, bad_low=0.1, bad_high=10.0):
    single = np.asarray(x).ndim == 2
    x = _array(x)
    if single:
        x = x[None]

    finite = np.isfinite(x)
    safe = np.where(finite, x, 0.0)
    centered = safe - np.median(safe, axis=-1, keepdims=True)
    std = np.std(centered, axis=-1)
    median_std = np.median(std, axis=1, keepdims=True)
    median_std = np.maximum(median_std, 1e-8)

    flat = np.mean(np.all(np.abs(safe) <= flat_eps, axis=1), axis=1)
    amplitude = np.percentile(np.abs(centered), 99.5, axis=(1, 2))
    jump = np.percentile(np.abs(np.diff(centered, axis=-1)), 99.5, axis=(1, 2))

    minimum = np.min(safe, axis=-1, keepdims=True)
    maximum = np.max(safe, axis=-1, keepdims=True)
    clipped = np.isclose(safe, minimum, atol=flat_eps) | np.isclose(
        safe, maximum, atol=flat_eps
    )
    clipping = np.max(np.mean(clipped, axis=-1), axis=1)

    ratio = std / median_std
    bad_channels = np.mean((ratio < bad_low) | (ratio > bad_high), axis=1)
    nan_fraction = 1.0 - np.mean(finite, axis=(1, 2))

    result = {
        "flat_fraction": flat,
        "amplitude_uv": amplitude,
        "jump_uv": jump,
        "clipping_fraction": clipping,
        "bad_channel_fraction": bad_channels,
        "nan_fraction": nan_fraction,
    }
    if single:
        return {name: float(value[0]) for name, value in result.items()}
    return result


def clean_mask(x, max_flat=0.01, max_amplitude_uv=1000.0, max_jump_uv=250.0,
               max_clipping=0.01, max_bad_channels=0.25, max_nan=0.0):
    q = quality(x)
    single = np.asarray(x).ndim == 2
    if single:
        return bool(
            q["flat_fraction"] <= max_flat
            and q["amplitude_uv"] <= max_amplitude_uv
            and q["jump_uv"] <= max_jump_uv
            and q["clipping_fraction"] <= max_clipping
            and q["bad_channel_fraction"] <= max_bad_channels
            and q["nan_fraction"] <= max_nan
        )
    return (
        (q["flat_fraction"] <= max_flat)
        & (q["amplitude_uv"] <= max_amplitude_uv)
        & (q["jump_uv"] <= max_jump_uv)
        & (q["clipping_fraction"] <= max_clipping)
        & (q["bad_channel_fraction"] <= max_bad_channels)
        & (q["nan_fraction"] <= max_nan)
    )


def preprocess(x, rate, low=0.5, high=40.0, notch_freq=50.0, normalize_signal=False):
    x = notch(x, rate, notch_freq)
    x = bandpass(x, rate, low, high)
    if normalize_signal:
        x = normalize(x)
    return x
