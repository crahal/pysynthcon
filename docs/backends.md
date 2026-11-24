# Estimator implementations and compatibility

Install the optional numerical backends with `pip install -e '.[backends]'`.
Each estimator returns the same `Estimate` type. For statistical definitions
and the original papers, see the [methods guide](methods.md).

## Predictor-based, augmented and penalized SCM

These estimators use the checked routines from Stiofán Fordham's
[pysyncon](https://github.com/sdfordham/pysyncon/tree/b77dc20e29b012ab03ca1d4caeef7b41861e86f5).
The replication requirements pin this revision and the numerical libraries
used for the published examples.

```python
from pysynth import (
    AugmentedSyntheticControl, PenalizedSyntheticControl, PredictorSyntheticControl,
)

# Pretreatment predictor windows are supplied as original Panel time labels.
pre = list(panel.times[:panel.n_pre])
mid = len(pre) // 2
predictor = PredictorSyntheticControl(predictor_windows=[pre[:mid], pre[mid:]])
predictor_fit = predictor.fit(panel)
ridge_augmented = AugmentedSyntheticControl().fit(panel)
penalized = PenalizedSyntheticControl(lambda_=0.01).fit(panel)
```

Predictor SCM estimates the relative importance of the predictors in a nested
optimization or accepts a diagonal metric through `custom_V`. Penalized SCM
uses identity by default after standardizing the predictors. Ridge-augmented
SCM uses the no-covariate specification by default for a plain panel.
Its pretreatment cross-validation follows the reference implementation’s
holdout blocks and one-standard-error rule. An explicit zero ridge penalty
requires a centered design with full row rank. Residualizing auxiliary
covariates requires linearly independent centered covariates. If uniform
weights already balance a predictor exactly, the corresponding scaled
imbalance can have a zero denominator. The fit retains its weights and reports
NaN for that diagnostic.

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
# dp supplies the study's custom predictors and observation windows.
estimator = PredictorSyntheticControl(dataprep=dp, optim_method="Nelder-Mead")
result = estimator.fit(panel)
print(estimator.backend_.summary())
```

Both nested optimization stages now check solver termination and simplex
feasibility. If upstream SLSQP fails, CLARABEL solves the identical weighted
quadratic, including the pairwise penalty where applicable. A default outer
search that reaches its cap can recover with Powell on the same full nested
objective; explicitly supplied outer optimization settings are respected and
fail visibly. Every recovery and outer termination is recorded in metadata.
A successful SLSQP status also receives a check for numerical cancellation.
PySynthCon evaluates an attainable decrease along a feasible direction using
centered predictor differences. If that decrease exceeds the upstream
objective tolerance, CLARABEL refits the same quadratic and records the
reason in `metadata["optimization"]["inner_recoveries"]`. This matters when
predictors have large common levels relative to the differences among donors.

For existing code that passes backend-specific fit options, the low-level
`PySyncon` selector remains available. The named estimators expose their
statistical choices directly. Backend-specific uncertainty procedures on a
fitted upstream object retain their own definitions.

## Robust SCM

`RobustSyntheticControl(lambda_=0.1, sv_count=3)` uses NumPy for truncated
singular-value denoising and the ridge fit. Both training and prediction use
the denoised donor matrix. With a complete panel all observed zeros count as
observations. The solver also supports a zero ridge penalty and a
rank-deficient design. [Amjad, Shah & Shen (2018), Algorithm 1](https://jmlr.org/papers/v19/17-777.html)

## Differenced SCM and the alternative classical solver

`DifferencedSyntheticControl()` uses the unpenalized specification in
[Engelbrektson’s 2021 working paper](https://arxiv.org/abs/2111.10784).
`SCM("synth")` remains a lower-level alternative implementation of classical
SCM. PySynthCon uses version 1.1.17 of the author's
[SyntheticControlMethods](https://github.com/OscarEngelbrektson/SyntheticControlMethods)
software for these routines. The following adjustments support modern pandas
and SciPy:

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
predictor specification. The notebooks’ differenced specification uses every pretreatment lag and
ten optimization starts, including all 89 Kansas pretreatment quarters.
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
