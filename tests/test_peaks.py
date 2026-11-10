import numpy as np
import pytest

from wavepick.peaks import detect_peaks
from wavepick.synthetic import TruePeak, emg, synthetic_eic


def test_clean_gaussian_area_matches_sqrt6_sigma_window():
    # bounds at +/- sqrt(6) sigma should capture erf(sqrt(3)) = 98.6 % of the area
    p_true = TruePeak(200, 6.0, 50.0)
    _, clean = synthetic_eic(400, [p_true])
    (p,) = detect_peaks(clean, baseline="none")
    assert p.apex == 200
    assert p.area == pytest.approx(0.986 * p_true.area, rel=0.02)
    assert p.height == pytest.approx(50.0)


def test_area_with_offset_baseline_and_noise():
    truth = [TruePeak(100, 3.0, 80.0), TruePeak(280, 8.0, 40.0)]
    y, _ = synthetic_eic(400, truth, noise_sd=2.0, baseline=20.0, rng=7)
    peaks = detect_peaks(y)
    assert [p.apex for p in peaks] == pytest.approx([100, 280], abs=2)
    for p, t in zip(peaks, truth):
        assert p.area == pytest.approx(t.area, rel=0.08)


def test_x_axis_units_are_used_for_rt_and_area():
    _, clean = synthetic_eic(300, [TruePeak(150, 5.0, 10.0)])
    rt = 0.5 * np.arange(300) + 60.0  # 0.5 s per scan, starting at 60 s
    (p_idx,) = detect_peaks(clean, baseline="none")
    (p_rt,) = detect_peaks(clean, rt, baseline="none")
    assert p_rt.rt == pytest.approx(135.0)
    assert p_rt.area == pytest.approx(0.5 * p_idx.area)


@pytest.mark.parametrize("sep_sigmas", [3, 4, 6])
def test_coeluting_pair_is_resolved_and_bounds_do_not_overlap(sep_sigmas):
    s = 4.0
    _, clean = synthetic_eic(400, [TruePeak(200, s, 100), TruePeak(200 + sep_sigmas * s, s, 60)])
    peaks = detect_peaks(clean, baseline="none")
    assert len(peaks) == 2
    assert peaks[0].right <= peaks[1].left


def test_tailing_peak_detected_once():
    x = np.arange(400)
    y = emg(x, 200, 4.0, 8.0, 100.0) + np.random.default_rng(0).normal(0, 1, 400)
    peaks = detect_peaks(y)
    assert len(peaks) == 1
    assert 195 <= peaks[0].apex <= 215


def test_denoise_option_runs_and_keeps_peak():
    y, _ = synthetic_eic(512, [TruePeak(256, 4.0, 10.0)], noise_sd=2.0, rng=1)
    for method in ("sure", "visushrink"):
        peaks = detect_peaks(y, denoise=method)
        assert any(abs(p.apex - 256) <= 3 for p in peaks)


def test_flat_signal_has_no_peaks():
    assert detect_peaks(np.full(200, 5.0)) == []
