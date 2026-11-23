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
rescaling, and explicit convergence failures. `ridge` is not Abadie and
L'Hour's pairwise-distance penalty; use `PySyncon("pensynth")` for that method.
For classical nested optimization over predictor importance, pass a
`pysyncon.Dataprep` to the `PySyncon` adapter. Neither selecting predictors
nor a global optimum of the nested nonconvex problem is automatic.

## SCUL

[Hollingsworth and Wing's SCUL](https://hollina.github.io/scul/) uses lasso
to construct an extrapolating synthetic series. PySynth minimizes

$$\frac{1}{2n}\|y-a-Zb\|_2^2 + \alpha\|b\|_1.$$

Within each training window, donor columns are centered and divided by their
population standard deviations. The response is centered, and the intercept
is unpenalized. Constant columns get zero fitted coefficients. Coefficients
and intercept are transformed back to original units before prediction.
Weights can be negative and need not sum to one. In scikit-learn, `alpha`
is glmnet's **lambda**, not glmnet's elastic-net mixing parameter alpha.

Automatic selection uses expanding training windows, origins advancing by
one period, and the next `horizon` periods for validation. The default first
window is half the pre-periods, with at least three observations. At least
two validation origins are required. All windows end before treatment.

The default common grid has 100 log-spaced penalties from the largest
training-window $\alpha_{max}$ to 0.01 times the smallest, where
$\alpha_{max}=\max_j|Z_j'y|/n$. This follows scul's common-grid design:
**preprocessing is local to each fold, but the grid uses all pre-treatment
training windows**. It is not a fully isolated, nested forecasting exercise;
earlier validation outcomes can influence later training-window grid bounds.
For a grid fixed independently of validation outcomes, pass explicit
`alphas`. No treated post-period data enter selection or fitting.

| `selection` | Rule |
|---|---|
| `"min"` (default) | Lowest average rolling validation MSE; ties favor larger penalties |
| `"1se"` | Largest penalty within one **sample SD** of fold MSE at the best penalty, reproducing upstream's historical rule |
| `"median"` | Median fold-optimal penalty on the explicit common grid |

The `1se` name does not mean SD divided by the square root of fold count.
The median option differs from upstream's default, which uses fold-specific,
automatically truncated glmnet paths. PySynth's default is `min`, not the
R package's `lambda.median`. A supplied positive `alpha` bypasses validation.
Convergence issues are surfaced. Native and R SCUL `lambda.min` predictions
are checked on an independent R-generated fixture; exact default-path parity
is not claimed for the other options.

## Individual synthetic control (ISC)

`IndividualSC()` is vanilla ISC: fit a synthetic control separately for each
individual using their **full eligible donor pool**. This per-case design is
motivated by [Vagni and Breen (2021)](https://doi.org/10.1093/esr/jcab014).
For pretreatment feature vector $x_i$ and donor feature matrix $X_0$, solve

$$\widehat w_i=\arg\min_{w\geq0,\;\mathbf{1}'w=1}
\|x_i-X_0w\|_2^2.$$

There is no intercept. By default, features are all pretreatment outcomes;
`fit(panel, features=matrix)` replaces them with caller-supplied pretreatment
features. Predictions use the weighted donor outcome paths. With default
features and no donor resampling, this optimizer is the same as native
outcome SCM applied to one individual. The individual workflow adds separate
eligible panels, person weighting and event-time aggregation through `fit_many`.

## Two-Stage Individual Synthetic Control (2SISC)

`TwoStageIndividualSC(n_neighbors=10)` adds an explicit first stage:

1. **Nearest-neighbor donor screening.** From the eligible untreated pool,
   select up to $k$ donors with the smallest $d_{ij}^2=\|x_i-x_j\|_2^2$.
2. **Synthetic-control fitting.** Solve ISC's convex-weight problem within
   that selected group, using the same features. Excluded donors receive
   zero weight in the returned estimate.

This implements the requested nearest-neighbor grouping stage; it is not
k-means clustering. `n_neighbors` must be a positive integer and counts
actual donors, excluding the treated individual. The effective count is
`min(n_neighbors, panel.n_control)`. Ties at the cutoff follow panel unit
order. Selected donors retain that order during fitting. When all eligible
donors are retained, 2SISC agrees with vanilla ISC, including resampled fits
with the same random seed.

Both stages use pretreatment information. Distance and fitting features are
unscaled: changing their measurement units can change selection and weights.
Choose the feature scaling and neighbor count before inspecting treatment
effects. 2SISC metadata records the requested count, eligible and selected
counts, selected identities, and `squared_distances` in original donor order
(`panel.units[:panel.n_control]`). Both estimators record the feature source.

For example, `compare({"ISC": IndividualSC(), "2SISC":
TwoStageIndividualSC(n_neighbors=10)}, panel)` shows both estimates. With
individual-specific panels, either class supports `fit_many(panels,
person_weights=...)`. The older `IndividualSC(n_neighbors=k)` configuration
is replaced by `TwoStageIndividualSC(n_neighbors=k)`; constructor options
are keyword-only.

### Resampling and scope

With `n_bootstrap > 0`, both estimators sample donors with replacement and
average the fitted weights and paths. Vanilla ISC samples from the full
eligible pool; 2SISC samples from its fixed stage-one group. Screening is
not repeated, so these draws do not measure selection uncertainty.

These are independent implementations, not ports of MSCMT. The original ISC
workflow includes case-specific sampling, predictor-weight optimization,
and survey decisions that are not recreated automatically. Donor bootstrap
draws describe sensitivity conditional on the selected donor set; they are
not calibrated confidence intervals or a full survey-design variance.
Individuals can share donors, so their estimates need not be independent.

## Synthetic difference-in-differences

`SyntheticDID` follows Algorithm 1 of [Arkhangelsky et al. (2021)](https://doi.org/10.1257/aer.20190159)
and the no-covariate R synthdid implementation. Unit weights match treated
pre-outcomes with an intercept and squared-weight penalty. Time weights
match each donor's mean post-outcome using its pre-outcomes, also with an
intercept. Both weights lie on their respective probability simplices.

The noise estimate is the sample SD (denominator $m-1$) of all donor
pre-treatment first differences. Default regularization is
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
synthdid's time-varying covariate adjustment, staggered adoption extension,
or its bootstrap/jackknife/placebo variance estimators.

## Difference-in-differences

`DifferenceInDifferences` uses uniform donor and pre-time weights in the
same gap formula. It recovers the conventional four-cell contrast for a
simultaneously treated block. Its identification relies on an appropriate
parallel-trends condition; synthetic weighting does not make this assumption
testable after treatment.
