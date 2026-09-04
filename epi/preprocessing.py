import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt


FATAL = ("flat_signal", "nan")
WARNING = ("high_amplitude", "large_jump", "clipping", "bad_channels")


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


def signal_scale(x, method="robust", eps=1e-8):
    x = _array(x)
    if method == "robust":
        center = np.median(x, axis=-1, keepdims=True)
        scale = np.median(np.abs(x - center), axis=-1, keepdims=True) * 1.4826
    elif method == "zscore":
        center = np.mean(x, axis=-1, keepdims=True)
        scale = np.std(x, axis=-1, keepdims=True)
    else:
        raise ValueError("method must be 'robust' or 'zscore'")
    return center, np.maximum(scale, eps)


def normalize(x, method="robust", eps=1e-8, clip=None):
    x = _array(x)
    center, scale = signal_scale(x, method=method, eps=eps)
    out = (x - center) / scale
    if clip is not None:
        out = np.clip(out, -clip, clip)
    return out.astype(np.float32)


def quality(x, flat_eps=1e-6, bad_low=0.1, bad_high=10.0):
    single = np.asarray(x).ndim == 2
    x = _array(x)
    if single:
        x = x[None]

    finite = np.isfinite(x)
    safe = np.where(finite, x, 0.0)
    center, scale = signal_scale(safe)
    centered = safe - center
    normalized = centered / scale
    channel_scale = scale[..., 0]
    median_scale = np.median(channel_scale, axis=1, keepdims=True)
    median_scale = np.maximum(median_scale, 1e-8)

    flat = np.mean(np.all(np.abs(safe) <= flat_eps, axis=1), axis=1)
    amplitude = np.percentile(np.abs(centered), 99.5, axis=(1, 2))
    jump = np.percentile(np.abs(np.diff(centered, axis=-1)), 99.5, axis=(1, 2))
    amplitude_scale = np.percentile(np.abs(normalized), 99.5, axis=(1, 2))
    jump_scale = np.percentile(np.abs(np.diff(normalized, axis=-1)), 99.5, axis=(1, 2))

    minimum = np.min(safe, axis=-1, keepdims=True)
    maximum = np.max(safe, axis=-1, keepdims=True)
    clipped = np.isclose(safe, minimum, atol=flat_eps) | np.isclose(
        safe, maximum, atol=flat_eps
    )
    clipping = np.max(np.mean(clipped, axis=-1), axis=1)

    ratio = channel_scale / median_scale
    bad_channels = np.mean((ratio < bad_low) | (ratio > bad_high), axis=1)
    nan_fraction = 1.0 - np.mean(finite, axis=(1, 2))

    result = {
        "flat_fraction": flat,
        "amplitude_uv": amplitude,
        "jump_uv": jump,
        "amplitude_scale": amplitude_scale,
        "jump_scale": jump_scale,
        "signal_scale_uv": median_scale[:, 0],
        "clipping_fraction": clipping,
        "bad_channel_fraction": bad_channels,
        "nan_fraction": nan_fraction,
    }
    if single:
        return {name: float(value[0]) for name, value in result.items()}
    return result


def _over(value, limit):
    if limit is None:
        return np.zeros_like(value, dtype=bool)
    return value > limit


def _quality_flags(q, max_flat, max_amplitude_uv, max_jump_uv, max_clipping,
                   max_bad_channels, max_nan, max_amplitude_scale=None,
                   max_jump_scale=None):
    return {
        "flat_signal": _over(q["flat_fraction"], max_flat),
        "high_amplitude": _over(q["amplitude_uv"], max_amplitude_uv)
        | _over(q["amplitude_scale"], max_amplitude_scale),
        "large_jump": _over(q["jump_uv"], max_jump_uv)
        | _over(q["jump_scale"], max_jump_scale),
        "clipping": _over(q["clipping_fraction"], max_clipping),
        "bad_channels": _over(q["bad_channel_fraction"], max_bad_channels),
        "nan": _over(q["nan_fraction"], max_nan),
    }


def quality_flags(x, max_flat=0.01, max_amplitude_uv=1000.0, max_jump_uv=250.0,
                  max_clipping=0.01, max_bad_channels=0.25, max_nan=0.0,
                  max_amplitude_scale=None, max_jump_scale=None):
    q = quality(x)
    return _quality_flags(
        q, max_flat, max_amplitude_uv, max_jump_uv, max_clipping,
        max_bad_channels, max_nan, max_amplitude_scale, max_jump_scale
    )


def quality_report(x, max_flat=0.01, max_amplitude_uv=1000.0, max_jump_uv=250.0,
                   max_clipping=0.01, max_bad_channels=0.25, max_nan=0.0,
                   max_amplitude_scale=None, max_jump_scale=None):
    single = np.asarray(x).ndim == 2
    q = quality(x)
    flags = _quality_flags(
        q, max_flat, max_amplitude_uv, max_jump_uv, max_clipping,
        max_bad_channels, max_nan, max_amplitude_scale, max_jump_scale
    )

    if single:
        fatal = any(bool(flags[name]) for name in FATAL)
        warning = any(bool(flags[name]) for name in WARNING)
        reasons = [name for name, bad in flags.items() if bad]
        return {
            **q,
            "fatal": fatal,
            "warning": warning,
            "keep": not fatal,
            "reasons": reasons,
        }

    n = len(next(iter(flags.values())))
    fatal = np.zeros(n, dtype=bool)
    warning = np.zeros(n, dtype=bool)

    for name in FATAL:
        fatal |= flags[name]
    for name in WARNING:
        warning |= flags[name]

    reasons = np.array([
        ";".join(name for name, bad in flags.items() if bad[i]) or "ok"
        for i in range(n)
    ], dtype=object)

    return {
        **q,
        "fatal": fatal,
        "warning": warning,
        "keep": ~fatal,
        "reasons": reasons,
    }


def fit_quality_limits(x, quantile=0.995, max_flat=0.01, max_clipping=0.01,
                       max_bad_channels=0.25, max_nan=0.0):
    q = quality(x)
    valid = (q["flat_fraction"] <= max_flat) & (q["nan_fraction"] <= max_nan)
    if not np.any(valid):
        raise ValueError("no valid windows for quality limits")
    return {
        "max_flat": max_flat,
        "max_amplitude_uv": None,
        "max_jump_uv": None,
        "max_clipping": max_clipping,
        "max_bad_channels": max_bad_channels,
        "max_nan": max_nan,
        "max_amplitude_scale": float(np.quantile(q["amplitude_scale"][valid], quantile)),
        "max_jump_scale": float(np.quantile(q["jump_scale"][valid], quantile)),
    }


def clean_mask(x, max_flat=0.01, max_amplitude_uv=1000.0, max_jump_uv=250.0,
               max_clipping=0.01, max_bad_channels=0.25, max_nan=0.0,
               strict=False, max_amplitude_scale=None, max_jump_scale=None):
    flags = quality_flags(
        x,
        max_flat=max_flat,
        max_amplitude_uv=max_amplitude_uv,
        max_jump_uv=max_jump_uv,
        max_clipping=max_clipping,
        max_bad_channels=max_bad_channels,
        max_nan=max_nan,
        max_amplitude_scale=max_amplitude_scale,
        max_jump_scale=max_jump_scale,
    )
    single = np.asarray(x).ndim == 2

    if single:
        fatal = any(bool(flags[name]) for name in FATAL)
        warning = any(bool(flags[name]) for name in WARNING)
        return not (fatal or (strict and warning))

    n = len(next(iter(flags.values())))
    fatal = np.zeros(n, dtype=bool)
    warning = np.zeros(n, dtype=bool)

    for name in FATAL:
        fatal |= flags[name]
    for name in WARNING:
        warning |= flags[name]

    return ~(fatal | warning) if strict else ~fatal


def preprocess(x, rate, low=0.5, high=40.0, notch_freq=50.0, normalize_signal=False,
               normalize_method="robust", clip=None):
    x = notch(x, rate, notch_freq)
    x = bandpass(x, rate, low, high)
    if normalize_signal:
        x = normalize(x, method=normalize_method, clip=clip)
    return x
