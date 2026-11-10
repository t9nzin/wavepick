"""Discrete wavelet denoising: VisuShrink (universal) and SureShrink thresholds.

Both estimate the noise level from the finest detail coefficients,
``sigma = MAD / 0.6745``, and shrink detail coefficients towards zero. The
approximation coefficients (the smooth part) are left alone.

* **VisuShrink** uses one threshold for all levels, ``sigma * sqrt(2 ln n)``.
  With high probability it removes all pure-noise coefficients, but it also
  over-smooths, so small peaks next to large ones get flattened.
* **SureShrink** picks a threshold per level that minimises Stein's Unbiased
  Risk Estimate of the MSE. Where a level is nearly all noise it falls back
  to the universal threshold (the "hybrid" rule of Donoho & Johnstone, 1995).

LC-MS intensities have roughly Poisson noise (variance grows with signal), which
breaks the constant-sigma assumption. ``vst=True`` applies the Anscombe
transform ``2 sqrt(y + 3/8)``, which makes the variance roughly constant,
denoises, then inverts it.
"""

from __future__ import annotations

import numpy as np
import pywt


def mad_sigma(detail: np.ndarray) -> float:
    """Robust noise std from detail coefficients (median absolute deviation)."""
    d = np.asarray(detail, dtype=float)
    return float(np.median(np.abs(d - np.median(d))) / 0.6745)


def universal_threshold(sigma: float, n: int) -> float:
    return float(sigma * np.sqrt(2.0 * np.log(max(n, 2))))


def sure_threshold(x: np.ndarray) -> float:
    """Threshold minimising SURE for soft thresholding of unit-variance data.

    For a threshold t equal to the k-th smallest |x| (squared values a_i sorted):

        SURE(t) = n - 2k + sum_{i<=k} a_i + (n - k) a_k
    """
    a = np.sort(np.abs(np.asarray(x, dtype=float))) ** 2
    n = a.size
    if n == 0:
        return 0.0
    k = np.arange(1, n + 1)
    risks = n - 2 * k + np.cumsum(a) + (n - k) * a
    return float(np.sqrt(a[int(np.argmin(risks))]))


def sureshrink_threshold(detail: np.ndarray, sigma: float) -> float:
    """Hybrid SureShrink threshold for one level, in the units of ``detail``."""
    n = detail.size
    if n == 0 or sigma <= 0:
        return 0.0
    x = detail / sigma
    t_univ = np.sqrt(2.0 * np.log(max(n, 2)))
    eta = (np.sum(x**2) - n) / n
    crit = np.log2(max(n, 2)) ** 1.5 / np.sqrt(n)
    if eta < crit:  # level is sparse: SURE is unreliable, use universal
        return float(sigma * t_univ)
    return float(sigma * min(sure_threshold(x), t_univ))


def _anscombe(y: np.ndarray) -> np.ndarray:
    return 2.0 * np.sqrt(np.clip(y, 0, None) + 3.0 / 8.0)


def _inverse_anscombe(z: np.ndarray) -> np.ndarray:
    return np.clip((z / 2.0) ** 2 - 3.0 / 8.0, 0, None)


def wavelet_denoise(
    y: np.ndarray,
    *,
    wavelet: str = "sym8",
    method: str = "sure",
    mode: str = "soft",
    level: int | None = None,
    sigma: float | None = None,
    vst: bool = False,
) -> np.ndarray:
    """Denoise a 1-D signal by thresholding its DWT detail coefficients.

    Parameters
    ----------
    method: ``"sure"`` (SureShrink, per-level) or ``"visushrink"`` (universal).
    mode: ``"soft"`` or ``"hard"`` thresholding.
    level: decomposition depth; defaults to the maximum useful level.
    sigma: noise std; estimated from the finest level if not given.
    vst: apply the Anscombe variance-stabilising transform first.
    """
    if method not in {"sure", "visushrink"}:
        raise ValueError(f"unknown method {method!r}")
    x = np.asarray(y, dtype=float)
    n = x.size
    work = _anscombe(x) if vst else x
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(n, w.dec_len)
    if max_level < 1:
        return x.copy()
    level = max_level if level is None else min(level, max_level)
    coeffs = pywt.wavedec(work, w, level=level, mode="symmetric")
    if sigma is None:
        sigma = mad_sigma(coeffs[-1])
    elif vst:
        sigma = 1.0  # Anscombe-transformed Poisson noise has unit variance
    out = [coeffs[0]]
    for d in coeffs[1:]:
        if method == "visushrink":
            t = universal_threshold(sigma, n)
        else:
            t = sureshrink_threshold(d, sigma)
        out.append(pywt.threshold(d, t, mode=mode))
    rec = pywt.waverec(out, w, mode="symmetric")[:n]
    return _inverse_anscombe(rec) if vst else rec
