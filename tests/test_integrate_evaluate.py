import numpy as np
import pytest

from wavepick.evaluate import match_peaks
from wavepick.integrate import estimate_baseline, integrate


def test_integrate_trapezoid_triangle():
    y = np.array([0, 1, 2, 1, 0], dtype=float)
    assert integrate(y, 0, 4) == pytest.approx(4.0)


def test_integrate_gaussian_full_area():
    x = np.arange(1000, dtype=float)
    y = 5 * np.exp(-0.5 * ((x - 500) / 7) ** 2)
    assert integrate(y, 400, 600) == pytest.approx(5 * 7 * np.sqrt(2 * np.pi), rel=1e-6)


def test_integrate_linear_baseline_removes_ramp():
    x = np.arange(101, dtype=float)
    ramp = 0.3 * x + 10
    peak = np.exp(-0.5 * ((x - 50) / 5) ** 2)
    a = integrate(ramp + peak, 0, 100, baseline="linear")
    assert a == pytest.approx(5 * np.sqrt(2 * np.pi), rel=1e-3)


def test_integrate_scalar_and_array_baseline_and_x():
    y = np.full(11, 3.0)
    assert integrate(y, 0, 10, baseline=3.0) == 0.0
    assert integrate(y, 0, 10, baseline=np.full(11, 1.0)) == pytest.approx(20.0)
    assert integrate(y, 0, 10, x=np.linspace(0, 1, 11)) == pytest.approx(3.0)


def test_integrate_bad_bounds():
    with pytest.raises(ValueError):
        integrate(np.zeros(5), 3, 1)
    with pytest.raises(ValueError):
        integrate(np.zeros(5), 0, 5)
    assert integrate(np.ones(5), 2, 2) == 0.0


def test_estimate_baseline_ignores_narrow_peak():
    x = np.arange(500, dtype=float)
    base = 10 + 0.01 * x
    y = base + 100 * np.exp(-0.5 * ((x - 250) / 4) ** 2)
    est = estimate_baseline(y, window=101)
    assert np.max(np.abs(est[60:-60] - base[60:-60])) < 0.5


def test_match_peaks_counts():
    m = match_peaks([10, 21, 50, 52], [10, 20, 50, 90], tol=2)
    assert (m.tp, m.fp, m.fn) == (3, 1, 1)
    assert m.precision == 0.75 and m.recall == 0.75
    assert m.f1 == pytest.approx(0.75)


def test_match_peaks_one_to_one_prefers_closest():
    m = match_peaks([11, 12], [12], tol=3)
    assert m.pairs == [(1, 0)]


def test_match_peaks_empty():
    m = match_peaks([], [1, 2], tol=1)
    assert (m.tp, m.fp, m.fn, m.recall, m.precision) == (0, 0, 2, 0.0, 1.0)
