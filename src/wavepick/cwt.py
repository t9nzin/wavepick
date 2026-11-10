"""Continuous wavelet transform and ridge-line peak finding, written from scratch.

The approach follows Du, Kibbe & Lin (2006), "Improved peak detection in mass
spectrum by incorporating continuous wavelet transform-based pattern matching"
(the basis of MassSpecWavelet and, indirectly, XCMS centWave):

1. Transform the signal with a Mexican-hat (Ricker) wavelet at many scales.
2. At each scale, find local maxima of the coefficients.
3. Link maxima across scales, from coarse to fine, into *ridge lines*.
4. Keep ridges that are long (persist across scales) and whose strongest
   coefficient stands out from the local noise level (SNR).

Key facts used throughout (derivations in the README):

* The wavelet is L2-normalised, so white noise of std ``s`` gives coefficients of
  std ``s`` at *every* scale. That is why noise can be estimated at the finest
  scale and compared with a peak's coefficient at any scale.
* A Gaussian peak of width ``sigma`` gives its largest response at scale
  ``a = sqrt(5) * sigma``, so ``sigma = a / sqrt(5)``.
* At that scale the coefficient row crosses zero at ``apex +/- sqrt(6) * sigma``,
  which contains about 98.6 % of the peak area. Those crossings are used as
  peak bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import maximum_filter1d

_RICKER_NORM = 2.0 / (np.sqrt(3.0) * np.pi**0.25)

#: Multiply a best-response scale by this to get the Gaussian sigma.
SCALE_TO_SIGMA = 1.0 / np.sqrt(5.0)


def ricker(points: int, a: float) -> np.ndarray:
    """Mexican-hat wavelet sampled at ``points`` integer positions centred on 0.

    psi(t) = 2 / (sqrt(3) pi^(1/4)) * a^(-1/2) * (1 - (t/a)^2) * exp(-(t/a)^2 / 2)

    The ``a^(-1/2)`` factor makes the L2 norm 1 at every scale.
    """
    if points < 1:
        raise ValueError("points must be >= 1")
    if a <= 0:
        raise ValueError("scale must be positive")
    t = np.arange(points) - (points - 1) / 2.0
    x = t / a
    return _RICKER_NORM / np.sqrt(a) * (1.0 - x**2) * np.exp(-(x**2) / 2.0)


def default_scales(min_scale: float = 1.0, max_scale: float = 32.0, num: int = 24) -> np.ndarray:
    """Geometrically spaced scales: equal resolution in log(width)."""
    return np.geomspace(min_scale, max_scale, num)


def cwt(signal: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Ricker CWT. Returns an array of shape ``(len(scales), len(signal))``.

    The kernel is truncated at +/- 5a (where the wavelet is < 1e-5 of its peak)
    and the signal is reflect-padded, so a constant baseline gives ~0 response at
    the edges instead of the step artefact that zero padding would create.
    """
    y = np.asarray(signal, dtype=float)
    n = y.size
    if n < 3:
        raise ValueError("signal must have at least 3 points")
    out = np.empty((len(scales), n))
    for i, a in enumerate(scales):
        m = int(min(10 * a + 1, n))
        if m % 2 == 0:
            m -= 1
        half = m // 2
        kernel = ricker(m, a)
        # When the signal is shorter than +/- 5a the kernel gets cut off and no
        # longer sums to zero; re-centre it so a constant still maps to 0.
        kernel -= kernel.mean()
        padded = np.pad(y, half, mode="reflect") if half else y
        # Ricker is symmetric, so convolution == correlation.
        out[i] = np.convolve(padded, kernel, mode="valid")
    return out


def scale_maxima(W: np.ndarray, scales: np.ndarray) -> list[np.ndarray]:
    """Columns of positive local maxima in each row of ``W``.

    The neighbourhood half-width grows with scale (``round(a)``), because two
    maxima closer than ~a apart at scale a cannot be separate features there.
    """
    maxima = []
    for i, a in enumerate(scales):
        w = max(1, round(a))
        row = W[i]
        mf = maximum_filter1d(row, size=2 * w + 1, mode="nearest")
        cols = np.flatnonzero((row == mf) & (row > 0))
        if cols.size > 1:  # collapse flat plateaus to their first point
            keep = np.ones(cols.size, dtype=bool)
            keep[1:] = ~((np.diff(cols) == 1) & (row[cols[1:]] == row[cols[:-1]]))
            cols = cols[keep]
        maxima.append(cols)
    return maxima


@dataclass
class Ridge:
    """A chain of local maxima linked from coarse (first) to fine (last) scale."""

    rows: list[int] = field(default_factory=list)
    cols: list[int] = field(default_factory=list)
    gap: int = 0

    def __len__(self) -> int:
        return len(self.rows)


def track_ridges(
    W: np.ndarray,
    scales: np.ndarray,
    max_gap: int = 2,
    link_tol: int | None = None,
) -> list[Ridge]:
    """Link per-scale maxima into ridge lines, from the coarsest scale down.

    Parameters
    ----------
    max_gap:
        How many consecutive scales a ridge may miss a maximum before it is
        closed. Noise can briefly suppress a maximum at one scale.
    link_tol:
        Maximum column shift when linking to the next finer scale. Defaults to
        ``round(a)`` for that scale: a maximum can drift by about one scale
        width as the wavelet narrows.

    Stronger ridges choose first, and each maximum joins at most one ridge.
    Maxima that join no ridge start new ones.
    """
    maxima = scale_maxima(W, scales)
    active: list[Ridge] = []
    finished: list[Ridge] = []
    for i in range(len(scales) - 1, -1, -1):
        cols = maxima[i]
        tol = max(1, round(scales[i])) if link_tol is None else link_tol
        claimed = np.zeros(cols.size, dtype=bool)
        active.sort(key=lambda r: W[r.rows[-1], r.cols[-1]], reverse=True)
        survivors: list[Ridge] = []
        for r in active:
            if cols.size:
                d = np.abs(cols - r.cols[-1]).astype(float)
                d[claimed] = np.inf
                j = int(np.argmin(d))
                if d[j] <= tol:
                    claimed[j] = True
                    r.rows.append(i)
                    r.cols.append(int(cols[j]))
                    r.gap = 0
                    survivors.append(r)
                    continue
            r.gap += 1
            (finished if r.gap > max_gap else survivors).append(r)
        for j in np.flatnonzero(~claimed):
            survivors.append(Ridge(rows=[i], cols=[int(cols[j])]))
        active = survivors
    return finished + active


@dataclass
class RidgePeak:
    col: int  #: location at the finest scale the ridge reaches
    best_row: int  #: scale index of the strongest coefficient on the ridge
    best_col: int  #: column of that coefficient
    scale: float  #: scale of the strongest coefficient
    sigma: float  #: implied Gaussian width, ``scale / sqrt(5)`` (samples)
    coef: float  #: strongest coefficient on the ridge
    noise: float  #: local noise level at the finest scale
    snr: float
    length: int  #: number of scales the ridge spans


def ridge_peaks(
    signal: np.ndarray,
    scales: np.ndarray | None = None,
    *,
    min_snr: float = 3.0,
    min_length: int | None = None,
    max_gap: int = 2,
    noise_window: int | None = None,
    noise_quantile: float = 0.95,
) -> tuple[list[RidgePeak], np.ndarray, np.ndarray]:
    """Find peaks as ridge lines in the CWT.

    Parameters
    ----------
    min_snr:
        Threshold on ``coef / noise``.
    min_length:
        Minimum number of scales a ridge must span. Defaults to a quarter of the
        scales; isolated noise maxima rarely persist that long.
    noise_window:
        Half-width (samples) of the window used for the local noise estimate.
        Defaults to ``max(50, 3 * max_scale)``.
    noise_quantile:
        Noise is this quantile of |W| at the finest scale inside the window
        (0.95, as in MassSpecWavelet).

    Returns ``(peaks sorted by column, W, scales)``.
    """
    y = np.asarray(signal, dtype=float)
    scales = default_scales() if scales is None else np.asarray(scales, dtype=float)
    W = cwt(y, scales)
    ridges = track_ridges(W, scales, max_gap=max_gap)
    if min_length is None:
        min_length = max(3, len(scales) // 4)
    half = noise_window if noise_window is not None else max(50, int(3 * scales[-1]))
    fine = np.abs(W[0])
    # Noise floor tied to the signal's magnitude, so float round-off on a flat or
    # noiseless trace (|W| ~ 1e-15 * |y|) cannot pose as a high-SNR peak.
    floor = max(np.finfo(float).tiny, 1e-8 * float(np.max(np.abs(y))))

    peaks: dict[int, RidgePeak] = {}
    for r in ridges:
        if len(r) < min_length:
            continue
        coefs = W[r.rows, r.cols]
        k = int(np.argmax(coefs))
        coef = float(coefs[k])
        if coef <= floor:
            continue
        col = r.cols[-1]
        lo, hi = max(0, col - half), min(y.size, col + half + 1)
        noise = max(float(np.quantile(fine[lo:hi], noise_quantile)), floor)
        snr = coef / noise
        if snr < min_snr:
            continue
        scale = float(scales[r.rows[k]])
        p = RidgePeak(
            col=col,
            best_row=r.rows[k],
            best_col=r.cols[k],
            scale=scale,
            sigma=scale * SCALE_TO_SIGMA,
            coef=coef,
            noise=noise,
            snr=snr,
            length=len(r),
        )
        if col not in peaks or peaks[col].coef < coef:
            peaks[col] = p
    return sorted(peaks.values(), key=lambda p: p.col), W, scales
