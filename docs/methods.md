# Methods, objectives and assumptions

Let $C$ be the time-by-donor matrix, $y$ the treated outcome (or treated-group
mean), $T_0$ the number of pre-periods, and $w$ donor weights. Every estimated
path has a gap $y_t-\widehat y_t(0)$. ATT averages those gaps over the observed
post-period. Good pre-fit does not establish no spillovers, no anticipation,
or stability of the untreated relationship after treatment.

## Convex synthetic control

`SyntheticControl` minimizes

$$T_0^{-1}\|y_{pre}-C_{pre}w\|_2^2 + r\|w\|_2^2,
\qquad w\geq0,\quad \mathbf{1}'w=1.$$

The default $r=0$ gives outcome-only SCM, with no intercept. Every pre-period
has equal importance. The solver uses analytic gradients, a common numerical
rescaling, and explicit convergence failures. Result metadata includes the
convex optimality-gap upper bound in original objective units and in scaled
units. This diagnoses possible suboptimality; it is not a certificate for the
separate nonconvex predictor-weight search. `ridge` is not Abadie and
L'Hour's pairwise-distance penalty; use `PySyncon("pensynth")` for that method.
For classical nested optimization over predictor importance, pass a
`pysyncon.Dataprep` to the `PySyncon` adapter. Neither selecting predictors
nor a global optimum of the nested nonconvex problem is automatic.

## SCUL

`SCUL()` executes the [original Hollingsworth–Wing R
functions](https://github.com/hollina/scul/tree/121b588cf8e4eebcea68518b57a7c754040153bb/R)
with **glmnet 4.0-2**, the version recorded in their tutorial. The default
`backend="r"` never substitutes the earlier sklearn adaptation. Install the
[reference dependencies](reference_backends.md) before fitting.

The lasso objective is $\|y-a-Zb\|_2^2/(2n)+\alpha\|b\|_1$.
Each training fold standardizes its donors; glmnet handles the intercept and
response scaling. Weights can be negative and need not sum to one. PySynth's
`alpha` means glmnet's penalty **lambda**, not its elastic-net mixing parameter.

Expanding training windows advance by one period. Both `initial_window` and
`horizon` default to the post-treatment length, as in the original setup
function. Short panels require an explicit feasible design; the library never
silently shortens the horizon. At least one validation origin must fit wholly
within the pretreatment period; the sample-SD `1se` selector requires two.
Original preprocessing applies caret’s near-zero-variance rule to the full
series and every expanding training window. It excludes constant columns and
rare-value columns with a dominant/second-most-common frequency ratio above
19 and at most 10% distinct values. Removed donors retain zero weights in the
returned result; a rejected target produces an error. This original screening
can depend on post-period values, including the target’s, although regression
and penalty selection use only pretreatment outcomes. This is an explicit
source convention, not a claim that the complete preprocessing pipeline is
independent of treated post-outcomes.

| `selection` | Original R rule |
|---|---|
| `"median"` (default) | Median of the fold-optimal penalties on each fold's own automatically truncated glmnet path |
| `"min"` | Lowest mean validation MSE on the original 100-penalty common grid, rounded to 10 decimal places |
| `"1se"` | Largest penalty within one **sample SD** of fold MSE at the best penalty |

The historical `1se` rule does not divide the SD by the square root of the fold
count. The common grid uses the maximum training-fold lambda maximum and
0.01 times the minimum, so its construction uses all pretreatment training
windows. Final prediction uses the original automatic path and exact refit at
the selected penalty. No solver tolerances, path lengths, or iteration limits
are reduced. `alpha` explicitly bypasses cross-validation when the researcher
has specified a penalty in advance.

`backend="python"` is retained as an explicitly requested adaptation. It uses
sklearn and a common grid even for median selection; it is not an exact
replication. None of the empirical notebooks uses this backend.

## Individual synthetic control (ISC)

`IndividualSC()` fits each case over its **full eligible donor pool**, following
[Vagni and Breen (2021)](https://doi.org/10.1093/esr/jcab014) and the
[costofcare replication code](https://github.com/centre-for-care/costofcare/blob/6e7537d67b86fb08709d35b035c9ee9f45e83e93/replication_attempt/src/main.R).
The default `backend="mscmt"` calls **MSCMT 1.4.0**, using its complete default
inner weighted nonnegative least-squares solver and outer `DEoptC` search.
When the original `wnnlsExt` routine reports failure during final fitting or
weight/predictor checks, a checked numerical repair solves its unchanged
weighted quadratic at the same predictor weights. Successful calls, the
compiled `DEoptC` search and `wnnlsOpt` remain unchanged. The repair's convex
optimality gap must be at most `1e-8` on its commonly scaled mean-squared
objective, not in raw outcome units; see the
[reference backend guide](reference_backends.md). No draw or outer-search
step is dropped, and every repaired call is recorded in metadata.
Each fit includes all pretreatment outcomes. Additional covariates are supplied
as named time-series groups:

```python
fit = IndividualSC().fit(panel, predictors={
    "age": age_pre,                 # shape (panel.n_pre, number_of_units)
    "age_left_school": education_pre,
})
```

The outcome and each named covariate are distinct predictor-weight groups.
There is no temporal averaging: MSCMT receives the full time series and its
original default aggregation (`id`). With only the outcome group, predictor
weight optimization is mathematically unnecessary; MSCMT itself recognizes
this case. Creating an artificial predictor group for every lag would change
the original method. The inner donor-weight problem remains a simplex fit.
MSCMT records its own scaling and optimization, rather than using the earlier
fixed Euclidean metric in the Python adaptation.

The default averages **50 fits to donor samples drawn with replacement**, as
in costofcare's `run(r=50)`. The original standalone ISC demonstration uses 10,
so specify `n_bootstrap=10` to reproduce that design. `n_bootstrap=0` explicitly
requests an unresampled fit. Sampling uses R's Mersenne-Twister RNG with
`random_state=42`; all draws are generated before each optimizer is seeded
with `optimizer_seed=1`, matching the source's ordering. Duplicate sampled
donors receive distinct optimizer columns, then their weights are summed back
to the original donor identities before averaging. Metadata retains donor
samples, each draw's ATT, predictor weights, and package versions.

For multiple cases, use `fit_many(panels, predictors=case_predictors,
person_weights=weights)`. Each dictionary is keyed by the same case IDs; the
predictor value for a case is its own named matrix dictionary. Effects align
by event time, and the overall ATT averages the case means with the supplied
person weights, rather than overweighting people observed for more periods.
The R donor-sampling stream advances through cases in their supplied order,
as in the original parallel ISC script. The state is saved immediately after
sampling, before MSCMT sets its optimizer seed; numerical optimization cannot
alter the next case's donor draws. Case order consequently matters for exact
replication. Each standalone `fit` starts afresh from `random_state`.

## Two-Stage Individual Synthetic Control (2SISC)

`TwoStageIndividualSC(n_neighbors=10)` adds the requested first stage:

1. Select up to $k$ eligible donors with the smallest squared Euclidean
   distance to the treated pretreatment design, stacking the outcome and
   supplied covariate series.
2. Run the **same original ISC/MSCMT procedure**, including all 50 donor
   resamples, within that selected group. Return zero weights for excluded
   donors.

This is nearest-neighbor screening. Distances are unscaled, so measurement
units matter; choose scaling and $k$ before inspecting effects. Ties follow
Panel order. If all donors are retained, 2SISC and ISC agree, including donor
resamples under the same seed. With one selected donor, the simplex contains
only weight one; the adapter handles that exact solution because MSCMT
requires at least two control columns.

2SISC is a requested extension, not a published method with a canonical paper
estimate. Its resampling is conditional on the screened pool and does not
rerun screening or estimate selection uncertainty.

### Scope and explicit alternatives

The costofcare source does not record its MSCMT version; 1.4.0 is the release
available at its December 2024 revision. This is a pinned, reproducible
implementation of its algorithm, not proof of bitwise identity to an
unrecorded historical environment. The original survey eligibility decisions,
additional study covariates, and restricted microdata remain the researcher's
responsibility. Aggregate-panel examples are methodological comparisons,
not reproductions of the motherhood or care study's results.

The earlier fixed-metric implementation is available only by explicitly
requesting `backend="fixed_metric"`. It accepts a `features` matrix and solves
$\min_{w\geq0,\,\mathbf{1}'w=1}\|x_i-X_0w\|_2^2$; it does not run MSCMT's
predictor-weight optimization and is not used in the empirical notebooks.
Donor-resampling draws alone are not calibrated confidence intervals or a
full survey-design variance; individuals can also share donors.

## Synthetic difference-in-differences

`SyntheticDID` follows Algorithm 1 of [Arkhangelsky et al. (2021)](https://doi.org/10.1257/aer.20190159)
and the no-covariate R synthdid implementation. Unit weights match treated
pre-outcomes with an intercept and squared-weight penalty. Time weights
match each donor's mean post-outcome using its pre-outcomes, also with an
intercept. Both weights lie on their respective probability simplices.

The noise estimate is the sample SD (denominator $m-1$) of all donor
pre-treatment first differences, matching the authors’ R code. The paper’s
equation (2.2) instead writes a population denominator; we preserve and disclose
the reference implementation’s finite-sample convention. Default regularization is
$\zeta_\omega=(N_{tr}T_{post})^{1/4}\widehat\sigma$ and
$\zeta_\lambda=10^{-6}\widehat\sigma$. The quadratic objectives contain
**squared** zeta values. An explicit `noise` permits degenerate tiny panels;
otherwise at least two first differences are needed.

Let $g_t=y_t-C_tw$. Then

$$\widehat\tau=\frac{1}{T_{post}}\sum_{t\in post}g_t-
\sum_{t\in pre}\lambda_tg_t.$$

The returned counterfactual adds the time-weighted pre-gap to $C_tw$, so its
mean post-gap equals this ATT. The level correction is not the unweighted
intercept profiled during unit-weight estimation.

Optimization follows upstream's exact-line-search Frank–Wolfe algorithm:
100 initial steps, weights at or below one-quarter of the largest weight
zeroed, then up to 10,000 steps. Stop when objective decrease is no greater
than $(10^{-5}\widehat\sigma)^2$. `sparsify=False` skips the first phase.
Objective-decrease convergence is not a global-accuracy certificate; final
objectives, iteration counts, stopping status and dual gaps are returned.
The California R reference itself reaches the time-weight iteration cap;
PySynth warns while retaining that numerically matching result.

Donor post-outcomes are intentionally used for time weights. Treated
post-outcomes never determine weights. This version does not implement
synthdid's time-varying covariate adjustment or staggered adoption extension.
Bootstrap, jackknife, and placebo variance estimators follow the original
procedures; see the [inference guide](inference.md).

## Difference-in-differences

`DifferenceInDifferences` uses uniform donor and pre-time weights in the
same gap formula. It recovers the conventional four-cell contrast for a
simultaneously treated block. Its identification relies on an appropriate
parallel-trends condition; synthetic weighting does not make this assumption
testable after treatment.
