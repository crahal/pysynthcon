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
response scaling. Weights can be negative and need not sum to one. PySynthCon's
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
has specified a penalty in advance. The original glmnet backend can fit
this specification with a single post-treatment observation.

`backend="python"` is retained as an explicitly requested adaptation. It uses
sklearn and a common grid even for median selection; it is not an exact
replication. This grid uses the scale of the training-fold correlations.
Folds with identically zero correlations contribute validation errors at every
penalty and do not set the grid endpoints. The dual-gap check uses the
training outcome variance. None of the empirical notebooks uses this backend.

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

`TwoStageIndividualSC(n_neighbors=10)` follows the two-stage approach of
[Petrillo et al. (2025), section 2.3](https://eprints.whiterose.ac.uk/id/eprint/224118/1/Binder2025004.pdf).
The default `backend="pysyncon", n_bootstrap=0` follows the core fitting
specification of the authors' [Python `sc_b` function](https://github.com/centre-for-care/costofcare/blob/6e7537d67b86fb08709d35b035c9ee9f45e83e93/src/utils.py):

1. Select up to $k$ eligible donors by Euclidean distance on the raw, stacked
   pretreatment outcome and named covariate series. The ranking uses every
   supplied pretreatment observation, without temporal averaging or scaling.
2. Fit nested SCM within that pool, using each series' **pretreatment mean**
   as a predictor and all pretreatment outcomes for the outer fitting loss.
   PySyncon supplies its original predictor scaling and optimizer defaults,
   with the checked numerical recovery described in the backend guide.
3. Return weights in the original donor order, with zeros for excluded donors.

`predictors` accepts the same named, `n_pre`-by-units matrices as ISC. The
outcome is always included. `options` passes explicit settings to
`pysyncon.Synth.fit`; `backend_` exposes the fitted object. A constant predictor
mean across selected units makes upstream variance scaling undefined; the fit
raises rather than silently changing the specified predictors. With one donor,
the simplex fixes its weight at one and needs no optimizer.

With outcomes alone, stage two has just one mean predictor. Multiple donor
combinations can match that mean, so a close mean match need not reproduce the
pretreatment path or identify unique donor weights. This is a limitation of
that reduced specification; the canonical aggregate examples do not supply
the care study's additional covariates.

PySynthCon counts actual donors in `n_neighbors`. The authors' KDTree query counts
the treated case in `k_n`, normally leaving `k_n - 1` donors. Away from distance
ties, query `k_n=k+1` corresponds to PySynthCon's `k` donors when `k` is smaller
than the eligible donor count. The source also caps `k_n` at that count;
PySynthCon permits retaining every eligible donor. Ties here follow Panel order. Panel construction determines
eligibility and the observed time window; the study's survey filters and
historical event window are not applied automatically.

The default fit performs **no within-donor resampling**. The care paper's
population bootstrap samples individual effect estimates; that is a different
procedure and is not implemented here. To retain the earlier PySynthCon variant,
request it explicitly:

```python
legacy_variant = TwoStageIndividualSC(
    n_neighbors=10, backend="mscmt", n_bootstrap=50,
)
```

That variant runs the original ISC time-series predictor groups and 50 donor
resamples within the screened pool. With every donor retained, it agrees with
`IndividualSC()` under the same seed. Its resampling is conditional on the
screened pool and does not estimate selection uncertainty. It is separate from
the authors' mean-predictor two-stage fitting pipeline.

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
PySynthCon warns while retaining that numerically matching result.

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
