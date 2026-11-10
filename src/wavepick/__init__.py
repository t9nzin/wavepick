"""wavepick: wavelet-based peak detection for LC-MS."""

from .cwt import cwt, ricker, ridge_peaks, track_ridges
from .denoise import wavelet_denoise
from .eic import ROI, Run, build_rois, extract_eic, roi_to_eic
from .evaluate import match_peaks
from .integrate import estimate_baseline, integrate
from .peaks import Peak, detect_peaks

__all__ = [
    "ROI",
    "Peak",
    "Run",
    "build_rois",
    "cwt",
    "detect_peaks",
    "estimate_baseline",
    "extract_eic",
    "integrate",
    "match_peaks",
    "ricker",
    "ridge_peaks",
    "roi_to_eic",
    "track_ridges",
    "wavelet_denoise",
]
