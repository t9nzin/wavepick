# wavepick

Wavelet-based chromatographic peak detection for LC-MS metabolomics,
benchmarked against XCMS centWave on data where the true compounds are known.

Peak detection is the first step of every untargeted metabolomics pipeline.
centWave is itself built on a continuous wavelet transform. This project
rebuilds that idea from first principles, so every parameter can be explained
and tested, and then measures it against centWave and MZmine using ground truth.

## Status

| Step | What | State |
|---|---|---|
| 1 | mzML → centroids → EICs / ROIs (`eic.py`, `io.py`) | done, unit tested |
| 2 | DWT denoising: VisuShrink vs SureShrink, optional Anscombe VST (`denoise.py`) | done, unit tested |
| 3 | Mexican-hat CWT + ridge-line tracking, written from scratch (`cwt.py`, `peaks.py`) | done, unit tested |
| – | Peak bounds, baseline, integration, P/R scoring (`integrate.py`, `evaluate.py`) | done, unit tested |
| – | MTBLS733 benchmark vs centWave / MZmine | next |
| – | Synthetic-injection LOD study, blank / scrambled-RT negative controls | next |
| 4 | 2-D (RT × m/z) wavelet decomposition | extension |
| 5 | 1-D CNN false-peak filter on MTBLS1455 labels | extension |

## Install and test

```bash
uv venv --python 3.11 && uv pip install -e '.[dev,mzml]'
.venv/bin/pytest -q
```

```python
from wavepick import detect_peaks
from wavepick.io import read_mzml
from wavepick.eic import build_rois, roi_to_eic

run = read_mzml("sample.mzML")
for roi in build_rois(run, ppm=10, min_length=5):
    idx, y = roi_to_eic(roi, run.n_scans, pad=20)
    for p in detect_peaks(y, run.rt[idx], min_snr=3):
        print(f"{roi.mz:.4f}  rt={p.rt:.1f}s  area={p.area:.3g}  snr={p.snr:.1f}")
```

## The method, and why each parameter is what it is

**Wavelet.** The Mexican hat (Ricker) wavelet
`ψ_a(t) = 2/(√3 π^¼) · a^-½ · (1 − t²/a²) · exp(−t²/2a²)`.
It is the negative second derivative of a Gaussian, so it acts as a matched
filter for Gaussian-like peaks. It integrates to zero, so a constant baseline
gives no response.

**Normalisation (a^-½).** This makes ‖ψ_a‖₂ = 1 at every scale. As a result,
white noise of std *s* gives coefficients of std *s* at every scale (tested in
`test_white_noise_coefficients_have_same_std_at_every_scale`). That is why noise
can be measured at the finest scale and compared with a peak's coefficient at
any scale.

**Scale ↔ peak width.** Convolving ψ_a with a Gaussian of width σ gives a
centre response ∝ σ · a^{5/2} / (a² + σ²)^{3/2}. Setting the derivative to zero
gives **a\* = √5 σ**. The detector reports `sigma = best_scale / √5`. The test
`test_gaussian_best_scale_is_sqrt5_sigma` checks this to within 3%.

**Peak bounds.** At scale a\*, the coefficient row is a Mexican hat of width
√(a\*² + σ²) = √6 σ. It crosses zero at **apex ± √6 σ ≈ ±2.45 σ**, which contains
erf(√3) = 98.6% of a Gaussian's area. Those zero crossings are used as the
integration bounds. `test_clean_gaussian_area_matches_sqrt6_sigma_window` checks
the 98.6%.

**Ridge lines** (Du et al. 2006). At each scale, local maxima are found within
±round(a). They are then linked from the coarsest scale to the finest, allowing
a drift of up to round(a) columns and gaps of up to `max_gap` scales. A real peak
gives a ridge that persists across many scales. A noise spike gives a short one.
`min_length` (default: a quarter of the scales) enforces that.

**SNR.** The ridge's strongest coefficient is divided by the 95th percentile of
|W| at the finest scale within ±`noise_window` samples (the MassSpecWavelet
convention). Because the transform is L2-normalised, this ratio means the same
thing for narrow and wide peaks.

**Quantification** is done on the raw signal, never the denoised one: soft
thresholding shrinks peak heights by design. The baseline is a running median
followed by a running mean. The median ignores peaks narrower than half the
window and, unlike a running minimum, is not biased low by noise.

**Denoising.** Noise σ is estimated as MAD/0.6745 of the finest DWT detail
coefficients.
- VisuShrink uses one threshold for all levels, σ√(2 ln n). It removes nearly
  all noise and also flattens small peaks.
- SureShrink uses a per-level threshold that minimises Stein's unbiased risk,
  with a fall-back to the universal threshold on sparse levels.

On the test mixture, SureShrink keeps a small peak next to large ones (height
8.6 vs a true value of about 8), while VisuShrink cuts it to 4.8. For
Poisson-like ion counts, `vst=True` applies the Anscombe transform first.

## Data

- **MTBLS733**: standard mixtures of 1100 compounds with known concentration
  ratios (Q Exactive HF). Main precision/recall and quantification benchmark.
- **MTBLS1455**: 255k EICs manually labelled by three experts. Second test set
  and CNN training data.
- **GNPS/MassIVE natural-product extract**: qualitative demo at the end.

## How this was built with Claude Code

Claude Code wrote the code. My job was to specify each task tightly and to
check its output with tests derived from theory, not from whatever the code
happened to return.

- **Specification.** Each module was specified as a contract: inputs, outputs,
  and the property it must satisfy. Examples: "the CWT of a constant is 0 at
  every scale, including at the edges", "best scale for a Gaussian is √5σ",
  "bounds capture 98.6% of the area". The expected values were derived by hand
  first; the tests encode those derivations.
- **Negative controls in the unit tests.** Pure white noise must give no
  detections, and a flat trace must give none.
- **What the checks caught.** The first version passed its happy-path tests but
  failed three checks, each a real bug:
  1. Integration bounds stopped one sample inside the zero crossing, which
     undercounted areas by 5–7%. Caught by comparing against the analytic area.
  2. When the signal is shorter than ±5a, the truncated wavelet no longer sums
     to zero, so a flat trace produced a "peak" with SNR around 10⁵. Fixed by
     re-centring the kernel. Caught by the flat-signal control.
  3. The exponentially-modified-Gaussian generator returned NaN on the tail
     (`0 · ∞` in the `erfcx` form). Fixed with a piecewise stable formula.
     Caught by the tailing-peak test.

## Layout

```
src/wavepick/
  eic.py        Run container, targeted EIC, untargeted ROI builder
  io.py         mzML reader (pymzml)
  denoise.py    DWT denoising: VisuShrink, SureShrink, Anscombe VST
  cwt.py        Ricker wavelet, CWT, ridge tracking, ridge peaks + SNR
  peaks.py      detect_peaks: apex, bounds, overlap split, baseline, area
  integrate.py  trapezoidal integration, running-median baseline
  synthetic.py  Gaussian / EMG peaks with known parameters
  evaluate.py   one-to-one matching, precision / recall / F1
tests/          46 tests
```
