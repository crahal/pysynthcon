# Upstream compatibility and extensions

The adapters delegate estimation to the two requested primary libraries and
convert their outputs to `Estimate`. Backend dependencies remain optional.
`adapter.backend_` exposes the fitted upstream object for advanced diagnostics.
This attribute is established only after successful fitting.

## pysyncon

The `backends` extra pins commit
`b77dc20e29b012ab03ca1d4caeef7b41861e86f5` (version 1.7.0), because its
augmented-SCM replication fixes are newer than older releases. The replication
requirements pin NumPy/SciPy as well: nested optimization can vary by BLAS,
CPU and package versions. Routine core functionality is not tied to those
numerical environment pins.

```python
from pysynth import PySyncon

aug = PySyncon("augsynth").fit(panel)
pen = PySyncon("pensynth", options={"lambda_": 0.01}).fit(panel)
robust = PySyncon("robust", options={"lambda_": 0.1, "sv_count": 3}).fit(panel)
```

Without a custom Dataprep, synth and pensynth use individual pre-outcomes as
predictors with equal V importance **after upstream row standardization**.
That differs from native equal-outcome-period SCM. AugSynth uses the
no-covariate Ridge specification, with its upstream cross-validation.
RobustSynth requires explicit ridge and singular-value truncation choices;
it uses all donor periods for its denoising step. Results preserve the
upstream object's donor-weight prediction on the original outcome matrix.

Internally generated data use integer times so quarterly or date-labelled
panels do not encounter upstream integer-calendar assumptions. Labels are
restored on the returned result. For a custom Dataprep, unit labels, full
outcome paths, and fitting periods must match the Panel. The user retains
responsibility for pre-treatment predictor windows and a valid specification.

```python
# dp is a pysyncon.Dataprep with your predictor and special-predictor choices.
adapter = PySyncon("synth", dataprep=dp, options={"optim_method": "Nelder-Mead"})
result = adapter.fit(panel)
print(adapter.backend_.summary())
```

Custom backend fit options are passed through; unsupported options raise.
Backend-specific uncertainty procedures are available on the exposed object
where supported, but are not presented as a uniform PySynth interval.

## SyntheticControlMethods

Version 1.1.17 needs local compatibility handling with modern pandas/SciPy:

- Its predictor loss transposes a column target, causing a predictor-by-predictor
  broadcast residual. The adapter supplies the intended row target so the
  inner loss is the weighted squared discrepancy for each predictor.
- Its normalization takes standard deviations across predictors within each
  unit. The adapter standardizes each predictor across units, preserving the
  convex-combination relationship and the intended predictor-variance scaling.
- Its random optimization restarts produce a two-dimensional initialization
  where SciPy requires a vector. The adapter uses one deterministic start.
- Its DiffSynth groupby assignment can misalign pandas indices. A small local
  subclass computes outcome differences within units and drops each unit's
  first period. Lag predictors remain fixed, so their meaning is explicit.
- Unit identifiers are converted to stable strings for upstream display
  routines and restored in the result.

The adapter constructs fixed pre-outcome lag predictors plus upstream's
outcome-mean predictor. `n_lags=None` uses every pre-period, while an integer
chooses at most that many evenly spaced lags. The notebooks set `n_lags=10`
explicitly so the nested optimization remains practical on quarterly data.
It does not monkey-patch global functions or modify
the installed package. Differenced SCM applies the upstream mean pre-gap
level correction. These choices are recorded in metadata, and tests compare
the returned paths with the fitted upstream paths. This is a compatibility
implementation; not every upstream plotting/placebo method is re-exported.
Independent exact-mixture tests also verify the corrected predictor objective;
the adapter deliberately does not reproduce the uncorrected release's two
predictor-array bugs. Use the exposed backend object for inspection, not its
untested upstream placebo routines, whose shape conventions can differ.

The single initialization can be a local solution in nested optimization.
Inspect fit quality and use sensitivity analysis; it is not a claim of global
optimality. Constant or degenerate predictor designs can also make upstream
normalization undefined. Native methods provide clearer handling of many
degenerate cases; upstream failures are surfaced rather than replaced with a
different estimator.
