"""Extracted ion chromatograms (EICs) and regions of interest (ROIs).

A run is a sequence of centroided MS1 spectra, each with a retention time and
paired, m/z-sorted arrays of m/z and intensity.

* :func:`extract_eic` is the targeted path: given an m/z, take the most intense
  centroid within +/- ppm in every scan.
* :func:`build_rois` is the untargeted path, close to the ROI step in centWave.
  It follows each m/z trace through consecutive scans, so peak detection only
  has to run on short, mass-pure chromatograms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Run:
    rt: np.ndarray  #: retention time per scan (seconds), increasing
    mz: list[np.ndarray]  #: per-scan m/z, sorted ascending
    intensity: list[np.ndarray]  #: per-scan intensities, aligned with ``mz``

    def __post_init__(self) -> None:
        self.rt = np.asarray(self.rt, dtype=float)
        if not (len(self.rt) == len(self.mz) == len(self.intensity)):
            raise ValueError("rt, mz and intensity must have one entry per scan")
        if np.any(np.diff(self.rt) < 0):
            raise ValueError("rt must be non-decreasing")
        mzs, ints = [], []
        for m, i in zip(self.mz, self.intensity):
            m = np.asarray(m, dtype=float)
            i = np.asarray(i, dtype=float)
            if m.shape != i.shape:
                raise ValueError("mz and intensity arrays must match in each scan")
            order = np.argsort(m, kind="stable")
            mzs.append(m[order])
            ints.append(i[order])
        self.mz, self.intensity = mzs, ints

    @property
    def n_scans(self) -> int:
        return len(self.rt)


def ppm_window(mz: float, ppm: float) -> tuple[float, float]:
    d = mz * ppm * 1e-6
    return mz - d, mz + d


def extract_eic(run: Run, mz: float, ppm: float = 10.0, agg: str = "max") -> np.ndarray:
    """Intensity per scan of centroids within ``mz +/- ppm``.

    ``agg`` is ``"max"`` (default; robust to a stray neighbouring centroid) or
    ``"sum"``. Scans with no centroid in the window give 0.
    """
    if agg not in {"max", "sum"}:
        raise ValueError("agg must be 'max' or 'sum'")
    lo, hi = ppm_window(mz, ppm)
    out = np.zeros(run.n_scans)
    for s, (m, inten) in enumerate(zip(run.mz, run.intensity)):
        a, b = np.searchsorted(m, lo, "left"), np.searchsorted(m, hi, "right")
        if b > a:
            out[s] = inten[a:b].max() if agg == "max" else inten[a:b].sum()
    return out


@dataclass
class ROI:
    scans: list[int] = field(default_factory=list)
    mzs: list[float] = field(default_factory=list)
    intensities: list[float] = field(default_factory=list)
    last_scan: int = -1

    @property
    def mz(self) -> float:
        """Intensity-weighted mean m/z."""
        w = np.asarray(self.intensities)
        m = np.asarray(self.mzs)
        return float(np.sum(w * m) / np.sum(w)) if w.sum() > 0 else float(m.mean())

    @property
    def mean_mz(self) -> float:
        return float(np.mean(self.mzs))

    def __len__(self) -> int:
        return len(self.scans)

    def add(self, scan: int, mz: float, intensity: float) -> None:
        if self.scans and self.scans[-1] == scan:
            # two centroids hit the same ROI in one scan: keep the stronger
            if intensity > self.intensities[-1]:
                self.mzs[-1], self.intensities[-1] = mz, intensity
            return
        self.scans.append(scan)
        self.mzs.append(mz)
        self.intensities.append(intensity)
        self.last_scan = scan


def build_rois(
    run: Run,
    ppm: float = 10.0,
    min_length: int = 5,
    max_gap: int = 1,
    min_intensity: float = 0.0,
) -> list[ROI]:
    """Group centroids into m/z traces across consecutive scans.

    A centroid joins the open ROI whose running mean m/z is closest, provided it
    is within ``ppm``. Otherwise it starts a new ROI. An ROI closes once it has
    gone more than ``max_gap`` scans without a new point. Closed ROIs shorter
    than ``min_length`` scans are discarded.
    """
    open_rois: list[ROI] = []
    done: list[ROI] = []
    for s in range(run.n_scans):
        m_all, i_all = run.mz[s], run.intensity[s]
        keep = i_all > min_intensity
        m_all, i_all = m_all[keep], i_all[keep]

        centres = np.array([r.mean_mz for r in open_rois])
        order = np.argsort(centres)
        sorted_centres = centres[order]
        new: list[ROI] = []
        for mz, inten in zip(m_all, i_all):
            target = None
            if sorted_centres.size:
                j = np.searchsorted(sorted_centres, mz)
                best = np.inf
                for c in (j - 1, j):
                    if 0 <= c < sorted_centres.size:
                        d = abs(sorted_centres[c] - mz)
                        if d < best:
                            best, target = d, open_rois[order[c]]
                if target is not None and best > target.mean_mz * ppm * 1e-6:
                    target = None
            if target is None:
                r = ROI()
                r.add(s, float(mz), float(inten))
                new.append(r)
            else:
                target.add(s, float(mz), float(inten))

        still_open = []
        for r in open_rois:
            (still_open if s - r.last_scan <= max_gap else done).append(r)
        open_rois = still_open + new
    done.extend(open_rois)
    return sorted((r for r in done if len(r) >= min_length), key=lambda r: (r.scans[0], r.mz))


def roi_to_eic(roi: ROI, n_scans: int, pad: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Dense chromatogram for an ROI. Returns ``(scan_indices, intensities)``.

    Missing scans inside the ROI are 0. ``pad`` adds zero scans on each side
    (clipped to the run), which gives the CWT some baseline at the edges.
    """
    start = max(0, roi.scans[0] - pad)
    stop = min(n_scans, roi.scans[-1] + pad + 1)
    idx = np.arange(start, stop)
    y = np.zeros(idx.size)
    y[np.asarray(roi.scans) - start] = roi.intensities
    return idx, y
