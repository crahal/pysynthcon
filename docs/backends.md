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
predictors. Classical SCM estimates V with the full upstream nested
optimization; it no longer injects a fixed V. PenalizedSynth uses the original
method's identity metric unless the caller explicitly supplies another metric.
AugSynth uses the
no-covariate Ridge specification, with its upstream cross-validation. The
authors’ `augsynth` code omits the last possible holdout block; that convention
and its one-standard-error selection are deliberately retained. An explicit
zero ridge penalty requires a centered design with full row rank. With
dependent predictor rows, choose a positive penalty. Residualizing auxiliary
covariates requires linearly independent centered covariates. When uniform
weights already balance outcomes or covariates exactly, the corresponding
scaled imbalance has a zero denominator. PySynth preserves the fitted weights
and reports NaN for that diagnostic.

RobustSynth requires explicit ridge and singular-value truncation choices;
it uses all donor periods for its denoising step and predicts from the
denoised matrix, following the paper’s Algorithm 1. Because `Panel` requires
complete data, observed zeros count as observations: the observation fraction
is one. Threshold selection checks every singular value. A least-squares
form of the ridge solve avoids squaring the condition number and also handles
zero-penalty rank-deficient designs. The exposed object records the denoised
matrix, singular values and retained rank. This corrects the pinned upstream
missingness/truncation behavior; unmodified upstream predictions can differ
on affected inputs.

Internally generated data use integer times so quarterly or date-labelled
panels do not encounter upstream integer-calendar assumptions. Labels are
restored on the returned result. For a custom Dataprep, unit labels, full
outcome paths, and fitting periods must match the Panel exactly. Predictor
windows must be observed and strictly pretreatment. The explicit
`allow_post_predictors=True` exception is available for a documented historical
specification; the used post-periods are recorded in metadata. Germany’s
replication uses this exception for its published 1990 convention. The caller
remains responsible for which covariates are substantively eligible.

```python
# dp is a pysyncon.Dataprep with your predictor and special-predictor choices.
adapter = PySyncon("synth", dataprep=dp, options={"optim_method": "Nelder-Mead"})
result = adapter.fit(panel)
print(adapter.backend_.summary())
```

Both nested optimization stages now check solver termination and simplex
feasibility. If upstream SLSQP fails, CLARABEL solves the identical weighted
quadratic, including the pairwise penalty where applicable. A default outer
search that reaches its cap can recover with Powell on the same full nested
objective; explicitly supplied outer optimization settings are respected and
fail visibly. Every recovery and outer termination is recorded in metadata.
A successful SLSQP status also receives a check for numerical cancellation.
PySynth evaluates an attainable decrease along a feasible direction using
centered predictor differences. If that decrease exceeds the upstream
objective tolerance, CLARABEL refits the same quadratic and records the
reason in `metadata["optimization"]["inner_recoveries"]`. This matters when
predictors have large common levels relative to the differences among donors.

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
  where SciPy requires a vector. The adapter flattens the Dirichlet samples
  and restores all **ten** original starts, with a reproducible random seed.
- Its DiffSynth groupby assignment can misalign pandas indices. A small local
  subclass computes outcome differences within units and drops each unit's
  first period. Lag predictors remain fixed, so their meaning is explicit.
- Unit identifiers are converted to stable strings for upstream display
  routines and restored in the result.

The adapter constructs fixed pre-outcome lag predictors plus upstream's
outcome-mean predictor. `n_lags=None` uses every pre-period, while an integer
chooses at most that many evenly spaced lags as an explicit alternative
predictor specification. The notebooks use **every** pretreatment lag and
**ten** optimization starts, including all 89 Kansas pretreatment quarters.
It does not monkey-patch global functions or modify
the installed package. Differenced SCM applies the upstream mean pre-gap
level correction. These choices are recorded in metadata, and tests compare
the returned paths with the fitted upstream paths. This is a compatibility
implementation; not every upstream plotting/placebo method is re-exported.
Independent exact-mixture tests also verify the corrected predictor objective;
the adapter deliberately does not reproduce the uncorrected release's two
predictor-array bugs. Use the exposed backend object for inspection, not its
untested upstream placebo routines, whose shape conventions can differ.

### Line-search failure recovery

The previous adapter inherited unchecked `ABNORMAL_TERMINATION_IN_LNSRCH`
exits from upstream L-BFGS-B. A nested objective can change active donor sets;
finite differences of an approximate inner solve need not provide a reliable
line-search direction. Hiding the printed message does not solve that problem.

The local optimizer retains the statistical objective, bounds, uniform first
start, and seeded Dirichlet restarts. It solves the inner constrained quadratic
program with explicit tight CLARABEL tolerances. Writing the residual as
`(X - target) @ w` on `sum(w)=1` avoids subtraction of large levels without
changing the objective. The outer loss is divided by a constant for numerical
conditioning.

The outer L-BFGS-B gradient is calculated by differentiating the inner
quadratic program's Karush–Kuhn–Tucker (KKT) equations. An active-set
polishing step removes spurious tiny interior-point weights, checks dual
feasibility, and adds any donors with violated reduced-gradient conditions.
An adjoint linear solve then computes derivatives with respect to **every**
predictor weight. This is an analytic derivative of the same objective, not
a replacement estimator, omitted predictor or reduced number of starts.
It avoids subtracting nearly equal, approximate QP solutions in finite
differences; this was especially costly for all 89 Kansas pretreatment lags.

Singular KKT systems, nondifferentiable active-set boundaries, or a failed
outer line search trigger bounded Powell optimization of the **same** full
objective without a gradient. The original convex QP solution remains
available when a derivative cannot be certified. Failure of both outer
solvers raises an exception. An unrecovered solver failure is never silently
accepted.

Each start's solver, termination message, evaluation count, and initial/final
loss is saved in `result.metadata['optimization']`. The best evaluated feasible
weights across all starts are retained, as in upstream. Reusing the compiled
quadratic problem changes neither its data nor its solution; it does not remove
predictors, starts, or optimization stages. Tests compare the implicit gradient
with central differences of an independently derived two-donor solution,
including a nonzero penalty. They also check recovery for duplicate donors,
failed line searches, and an independently known convex-mixture solution.

These are explicit numerical and compatibility corrections to release 1.1.17,
not a claim of byte-identical results from its buggy unmodified code. Nested
optimization remains nonconvex: ten successful starts do not prove global
optimality. Review fit and sensitivity for substantive applications.
