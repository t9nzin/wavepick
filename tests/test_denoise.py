import numpy as np
import pytest

from wavepick.denoise import mad_sigma, sure_threshold, universal_threshold, wavelet_denoise
from wavepick.synthetic import TruePeak, synthetic_eic


def _sure_brute(x, t):
    x = np.abs(x)
    return x.size - 2 * np.sum(x <= t) + np.sum(np.minimum(x, t) ** 2)


def test_sure_threshold_minimises_sure_risk():
    x = np.random.default_rng(0).normal(0, 1, 300)
    x[:20] += 6
    t = sure_threshold(x)
    grid = np.sort(np.abs(x))
    risks = [_sure_brute(x, g) for g in grid]
    assert _sure_brute(x, t) == pytest.approx(min(risks))


def test_mad_sigma_estimates_gaussian_std():
    x = np.random.default_rng(1).normal(0, 3.0, 50000)
    assert mad_sigma(x) == pytest.approx(3.0, rel=0.02)


def test_universal_threshold_value():
    assert universal_threshold(2.0, 1000) == pytest.approx(2.0 * np.sqrt(2 * np.log(1000)))


@pytest.fixture
def mixture():
    truth = [TruePeak(100, 3, 100), TruePeak(300, 10, 40), TruePeak(330, 2, 8)]
    return synthetic_eic(512, truth, noise_sd=3.0, rng=1)


@pytest.mark.parametrize("method", ["sure", "visushrink"])
def test_denoising_reduces_error(mixture, method):
    y, clean = mixture
    d = wavelet_denoise(y, method=method)
    assert d.shape == y.shape
    assert np.sqrt(np.mean((d - clean) ** 2)) < np.sqrt(np.mean((y - clean) ** 2))


def test_sure_preserves_small_peak_better_than_visushrink(mixture):
    y, clean = mixture
    sure = wavelet_denoise(y, method="sure")
    visu = wavelet_denoise(y, method="visushrink")

    def err(d):
        return abs(d[330] - clean[330])

    assert err(sure) < err(visu)


def test_vst_output_is_nonnegative():
    y, _ = synthetic_eic(256, [TruePeak(128, 4, 500)], poisson_gain=1.0, rng=2)
    d = wavelet_denoise(y, vst=True)
    assert np.all(d >= 0)


def test_short_signal_passthrough():
    y = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(wavelet_denoise(y), y)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        wavelet_denoise(np.zeros(64), method="bogus")
