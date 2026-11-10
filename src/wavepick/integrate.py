"""Peak area integration and baseline estimation."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d


def integrate(
    y: np.ndarray,
    left: int,
    right: int,
    x: np.ndarray | None = None,
    baseline: str | float | np.ndarray = "none",
) -> float:
    """Trapezoidal area of ``y[left:right+1]`` above a baseline.

    ``baseline`` can be:

    * ``"none"``: integrate y as is
    * ``"linear"``: subtract the straight line joining the two bound points
    * a scalar: subtract a constant
    * an array the same length as ``y``: subtract it point by point
    """
    y = np.asarray(y, dtype=float)
    if not 0 <= left <= right < y.size:
        raise ValueError(f"invalid bounds [{left}, {right}] for length {y.size}")
    if right == left:
        return 0.0
    seg = y[left : right + 1]
    xs = (
        np.arange(left, right + 1, dtype=float)
        if x is None
        else np.asarray(x, float)[left : right + 1]
    )
    if isinstance(baseline, str):
        if baseline == "none":
            b = 0.0
        elif baseline == "linear":
            b = np.interp(xs, [xs[0], xs[-1]], [seg[0], seg[-1]])
        else:
            raise ValueError(f"unknown baseline {baseline!r}")
    else:
        b = np.asarray(baseline, dtype=float)
        if b.ndim == 1:
            if b.size != y.size:
                raise ValueError("baseline array must match y")
            b = b[left : right + 1]
    return float(np.trapezoid(seg - b, xs))


def estimate_baseline(y: np.ndarray, window: int) -> np.ndarray:
    """Slowly varying baseline: running median, then running mean.

    The median ignores peaks narrower than about half the window. The mean then
    removes the median's staircase steps. Unlike a running minimum, the median
    is not biased low by noise.
    """
    y = np.asarray(y, dtype=float)
    window = max(3, min(int(window) | 1, y.size if y.size % 2 else y.size - 1))
    med = median_filter(y, size=window, mode="nearest")
    return uniform_filter1d(med, size=window, mode="nearest")
