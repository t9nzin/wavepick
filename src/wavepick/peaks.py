"""High-level chromatographic peak detection: CWT ridges -> bounded, integrated peaks."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .cwt import default_scales, ridge_peaks
from .denoise import wavelet_denoise
from .integrate import estimate_baseline, integrate


@dataclass
class Peak:
    apex: int  #: index of the apex
    left: int  #: left bound (inclusive)
    right: int  #: right bound (inclusive)
    height: float  #: raw intensity at apex minus baseline
    area: float  #: baseline-corrected area on the raw signal (in x units if x given)
    sigma: float  #: Gaussian-equivalent width from the best CWT scale (samples)
    scale: float  #: best CWT scale
    snr: float
    rt: float  #: x at apex (equals ``apex`` if no x given)
    rt_left: float
    rt_right: float


def detect_peaks(
    y: np.ndarray,
    x: np.ndarray | None = None,
    *,
    scales: np.ndarray | None = None,
    min_snr: float = 3.0,
    min_length: int | None = None,
    max_gap: int = 2,
    denoise: str | None = None,
    baseline: str | float | np.ndarray = "auto",
    noise_window: int | None = None,
) -> list[Peak]:
    """Detect peaks in a chromatogram.

    Detection runs on the (optionally) denoised signal. Heights and areas are
    always measured on the *raw* signal, because soft thresholding shrinks
    peaks and would bias quantification.

    Parameters
    ----------
    scales: CWT scales in samples. The default (1-32) covers peak sigmas of about
        0.45 to 14 samples.
    denoise: ``None``, ``"sure"`` or ``"visushrink"``.
    baseline: ``"auto"`` (running median), ``"none"``, a scalar, or an array.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    scales = default_scales(max_scale=min(32.0, n / 4)) if scales is None else np.asarray(scales)
    work = wavelet_denoise(y, method=denoise) if denoise else y

    rps, W, scales = ridge_peaks(
        work,
        scales,
        min_snr=min_snr,
        min_length=min_length,
        max_gap=max_gap,
        noise_window=noise_window,
    )

    if isinstance(baseline, str) and baseline == "auto":
        base = estimate_baseline(work, window=int(6 * scales[-1]))
    elif isinstance(baseline, str) and baseline == "none":
        base = np.zeros(n)
    else:
        b = np.asarray(baseline, dtype=float)
        base = np.full(n, float(b)) if b.ndim == 0 else b

    cand = []
    for rp in rps:
        # apex: highest point of the working signal within ~sigma of the ridge
        r = max(1, round(rp.sigma))
        lo, hi = max(0, rp.col - r), min(n, rp.col + r + 1)
        apex = lo + int(np.argmax(work[lo:hi]))
        # bounds: zero crossings of the coefficient row at the best scale
        row = W[rp.best_row]
        cap = max(2, int(np.ceil(4 * rp.sigma)))
        left = rp.best_col
        while left > 0 and row[left - 1] > 0 and apex - left < cap:
            left -= 1
        right = rp.best_col
        while right < n - 1 and row[right + 1] > 0 and right - apex < cap:
            right += 1
        # step onto the sample at/after the zero crossing so the window spans it
        left, right = max(0, left - 1), min(n - 1, right + 1)
        left, right = min(left, apex), max(right, apex)
        cand.append([apex, left, right, rp])

    cand.sort(key=lambda c: c[0])
    # merge candidates sharing an apex; split overlapping neighbours at the valley
    merged: list[list] = []
    for c in cand:
        if merged and merged[-1][0] == c[0]:
            if c[3].snr > merged[-1][3].snr:
                merged[-1] = c
            continue
        merged.append(c)
    for a, b in itertools.pairwise(merged):
        if b[1] <= a[2]:
            split = a[0] + int(np.argmin(work[a[0] : b[0] + 1]))
            a[2], b[1] = split, split

    xs = np.arange(n, dtype=float) if x is None else np.asarray(x, dtype=float)
    peaks = []
    for apex, left, right, rp in merged:
        peaks.append(
            Peak(
                apex=apex,
                left=left,
                right=right,
                height=float(y[apex] - base[apex]),
                area=integrate(y, left, right, x=xs, baseline=base),
                sigma=rp.sigma,
                scale=rp.scale,
                snr=rp.snr,
                rt=float(xs[apex]),
                rt_left=float(xs[left]),
                rt_right=float(xs[right]),
            )
        )
    return peaks
