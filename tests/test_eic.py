import numpy as np
import pytest

from wavepick.eic import Run, build_rois, extract_eic, roi_to_eic


def _run_with_traces(traces, n_scans=50, jitter_ppm=2.0, seed=0):
    """traces: list of (mz, first_scan, last_scan, intensity)."""
    rng = np.random.default_rng(seed)
    mzs, ints = [], []
    for s in range(n_scans):
        m, i = [], []
        for mz, a, b, inten in traces:
            if a <= s <= b:
                m.append(mz * (1 + rng.normal(0, jitter_ppm) * 1e-6))
                i.append(inten)
        mzs.append(np.array(m))
        ints.append(np.array(i))
    return Run(rt=np.arange(n_scans) * 0.5, mz=mzs, intensity=ints)


def test_run_sorts_mz_within_scan():
    r = Run(rt=[0.0], mz=[[300.0, 100.0, 200.0]], intensity=[[3.0, 1.0, 2.0]])
    np.testing.assert_array_equal(r.mz[0], [100, 200, 300])
    np.testing.assert_array_equal(r.intensity[0], [1, 2, 3])


def test_run_validates_shapes():
    with pytest.raises(ValueError):
        Run(rt=[0.0, 1.0], mz=[[1.0]], intensity=[[1.0]])
    with pytest.raises(ValueError):
        Run(rt=[1.0, 0.0], mz=[[1.0], [1.0]], intensity=[[1.0], [1.0]])


def test_extract_eic_respects_ppm_window():
    # 500.0000 and 500.0100 are 20 ppm apart
    r = Run(
        rt=[0, 1, 2],
        mz=[[500.0, 500.01], [500.002], [500.01]],
        intensity=[[10.0, 99.0], [7.0], [5.0]],
    )
    np.testing.assert_array_equal(extract_eic(r, 500.0, ppm=10), [10, 7, 0])
    np.testing.assert_array_equal(extract_eic(r, 500.0, ppm=30), [99, 7, 5])
    np.testing.assert_array_equal(extract_eic(r, 500.0, ppm=30, agg="sum"), [109, 7, 5])


def test_build_rois_follows_traces_and_separates_close_masses():
    run = _run_with_traces(
        [(400.0000, 5, 25, 100.0), (400.0200, 10, 30, 50.0), (750.5, 0, 49, 10.0)]
    )  # first two are 50 ppm apart
    rois = build_rois(run, ppm=10, min_length=5)
    assert len(rois) == 3
    by_mz = sorted(rois, key=lambda r: r.mz)
    assert by_mz[0].mz == pytest.approx(400.0, abs=400 * 5e-6)
    assert (by_mz[0].scans[0], by_mz[0].scans[-1]) == (5, 25)
    assert (by_mz[1].scans[0], by_mz[1].scans[-1]) == (10, 30)
    assert len(by_mz[2]) == 50


def test_build_rois_gap_handling():
    run = _run_with_traces([(300.0, 0, 9, 1.0), (300.0, 11, 20, 1.0), (300.0, 24, 35, 1.0)])
    rois = build_rois(run, ppm=10, min_length=3, max_gap=1)
    # gap of 1 missing scan is bridged; gap of 3 is not
    assert [(r.scans[0], r.scans[-1]) for r in rois] == [(0, 20), (24, 35)]


def test_build_rois_min_length_and_intensity():
    run = _run_with_traces([(200.0, 0, 2, 5.0), (600.0, 0, 20, 0.5)])
    assert build_rois(run, min_length=5) == build_rois(run, min_length=5, min_intensity=0)
    assert [round(r.mz) for r in build_rois(run, min_length=5)] == [600]
    assert build_rois(run, min_length=5, min_intensity=1.0) == []


def test_roi_to_eic_fills_gaps_and_pads():
    run = _run_with_traces([(300.0, 3, 5, 2.0), (300.0, 7, 9, 2.0)], n_scans=12)
    (roi,) = build_rois(run, min_length=3, max_gap=1)
    idx, y = roi_to_eic(roi, run.n_scans, pad=2)
    np.testing.assert_array_equal(idx, np.arange(1, 12))
    np.testing.assert_array_equal(y, [0, 0, 2, 2, 2, 0, 2, 2, 2, 0, 0])
