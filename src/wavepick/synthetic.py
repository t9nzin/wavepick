"""Synthetic chromatographic peaks with known parameters, for tests and LOD studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erfc, erfcx


def gaussian(x: np.ndarray, center: float, sigma: float, height: float = 1.0) -> np.ndarray:
    return height * np.exp(-0.5 * ((np.asarray(x, float) - center) / sigma) ** 2)


def emg(x: np.ndarray, center: float, sigma: float, tau: float, height: float = 1.0) -> np.ndarray:
    """Exponentially modified Gaussian (a tailing peak), scaled so its max is ``height``.

    Uses the ``erfcx`` form where z >= 0 (stable for small tau) and the direct
    ``erfc`` form where z < 0 (where erfcx overflows).
    """
    if tau <= 0:
        return gaussian(x, center, sigma, height)
    t = np.asarray(x, float) - center
    z = (sigma / tau - t / sigma) / np.sqrt(2.0)
    f = np.empty_like(t)
    pos = z >= 0
    f[pos] = np.exp(-0.5 * (t[pos] / sigma) ** 2) * erfcx(z[pos])
    f[~pos] = np.exp(0.5 * (sigma / tau) ** 2 - t[~pos] / tau) * erfc(z[~pos])
    m = f.max()
    return height * f / m if m > 0 else f


@dataclass
class TruePeak:
    center: float
    sigma: float
    height: float
    tau: float = 0.0

    @property
    def area(self) -> float:
        """Analytic area (Gaussian only; for EMG, integrate the clean trace)."""
        return self.height * self.sigma * np.sqrt(2 * np.pi)


def synthetic_eic(
    n: int,
    peaks: list[TruePeak],
    *,
    noise_sd: float = 0.0,
    baseline: float = 0.0,
    poisson_gain: float | None = None,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(noisy, clean)`` traces of length ``n`` (x = sample index).

    ``poisson_gain``: if set, adds shot noise (variance = gain * signal) on top
    of the Gaussian read noise, as in real ion counts.
    """
    rng = np.random.default_rng(rng)
    x = np.arange(n, dtype=float)
    clean = np.full(n, float(baseline))
    for p in peaks:
        clean += emg(x, p.center, p.sigma, p.tau, p.height)
    noisy = clean.copy()
    if poisson_gain:
        noisy = rng.poisson(np.clip(clean, 0, None) / poisson_gain) * poisson_gain
        noisy = noisy.astype(float)
    if noise_sd:
        noisy = noisy + rng.normal(0.0, noise_sd, n)
    return noisy, clean
