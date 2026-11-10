"""mzML reading (needs the optional ``mzml`` extra: ``pip install wavepick[mzml]``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .eic import Run


def read_mzml(path: str | Path, ms_level: int = 1) -> Run:
    """Load the MS1 (or other level) spectra of an mzML file as centroids.

    Profile spectra are centroided by pymzml. Retention times are in seconds.
    """
    try:
        import pymzml
    except ImportError as e:  # pragma: no cover
        raise ImportError("read_mzml needs pymzml: pip install 'wavepick[mzml]'") from e

    rts, mzs, ints = [], [], []
    with pymzml.run.Reader(str(path)) as reader:
        for spec in reader:
            if spec.ms_level != ms_level:
                continue
            peaks = spec.peaks("centroided")
            if peaks is None or len(peaks) == 0:
                peaks = np.empty((0, 2))
            peaks = np.asarray(peaks, dtype=float)
            rt = spec.scan_time_in_minutes() * 60.0
            rts.append(rt)
            mzs.append(peaks[:, 0])
            ints.append(peaks[:, 1])
    return Run(rt=np.asarray(rts), mz=mzs, intensity=ints)
