import numpy as np
import pytest

from wavepick.cwt import SCALE_TO_SIGMA, cwt, ricker, ridge_peaks, track_ridges


@pytest.mark.parametrize("a", [2.0, 5.0, 10.0])
def test_ricker_is_zero_mean_and_unit_norm(a):
    k = ricker(int(10 * a) + 1, a)
    assert abs(k.sum()) < 1e-3
    assert np.sum(k**2) == pytest.approx(1.0, abs=1e-3)


def test_ricker_rejects_bad_args():
    with pytest.raises(ValueError):
        ricker(0, 1.0)
    with pytest.raises(ValueError):
        ricker(11, 0.0)


@pytest.mark.parametrize("sigma", [2.0, 4.0, 8.0])
def test_gaussian_best_scale_is_sqrt5_sigma(sigma):
    n = 1001
    y = np.exp(-0.5 * ((np.arange(n) - 500) / sigma) ** 2)
    scales = np.geomspace(1, 60, 300)
    W = cwt(y, scales)
    best = scales[np.argmax(W[:, 500])]
    assert best * SCALE_TO_SIGMA == pytest.approx(sigma, rel=0.03)


def test_cwt_constant_signal_has_no_edge_artifact():
    W = cwt(np.full(200, 7.0), np.array([1.0, 4.0, 16.0]))
    assert np.max(np.abs(W)) < 1e-3 * 7.0 * 16


def test_white_noise_coefficients_have_same_std_at_every_scale():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 20000)
    W = cwt(y, np.array([1.0, 4.0, 16.0]))
    stds = W[:, 500:-500].std(axis=1)
    np.testing.assert_allclose(stds, 1.0, rtol=0.1)


def test_single_peak_gives_one_long_ridge():
    y = np.exp(-0.5 * ((np.arange(400) - 200) / 5.0) ** 2)
    scales = np.geomspace(1, 32, 24)
    W = cwt(y, scales)
    ridges = [r for r in track_ridges(W, scales) if len(r) >= 6]
    assert len(ridges) == 1
    assert abs(ridges[0].cols[-1] - 200) <= 1


def test_ridge_peaks_recovers_narrow_and_wide_peaks():
    x = np.arange(600)
    y = 100 * np.exp(-0.5 * ((x - 150) / 2.0) ** 2) + 40 * np.exp(-0.5 * ((x - 400) / 10.0) ** 2)
    y += np.random.default_rng(3).normal(0, 2, x.size)
    peaks, _, _ = ridge_peaks(y, np.geomspace(1, 40, 30))
    cols = [p.col for p in peaks]
    assert len(peaks) == 2
    assert abs(cols[0] - 150) <= 2 and abs(cols[1] - 400) <= 3
    assert peaks[0].sigma == pytest.approx(2.0, rel=0.25)
    assert peaks[1].sigma == pytest.approx(10.0, rel=0.25)


def test_pure_noise_gives_no_peaks_at_default_snr():
    """Negative control: white noise should not produce detections."""
    total = 0
    for seed in range(10):
        y = np.random.default_rng(seed).normal(0, 1, 1000)
        peaks, _, _ = ridge_peaks(y)
        total += len(peaks)
    assert total <= 1
