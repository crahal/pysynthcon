# Reference SDID inference and placebo diagnostics

`Estimate.att` is a point estimate. No standard error is derived by treating
serially correlated time observations as independent.

## Synthetic difference-in-differences

`SyntheticDID.inference` implements Algorithms 2–4 from Arkhangelsky et al.
(2021), following the [pinned original R variance implementation](https://github.com/synth-inference/synthdid/blob/70c1ce3eac58e28c30b67435ca377bb48baa9b8a/R/vcov.R).
It accepts a fitted estimate so that the original point estimate is not recomputed.

```python
from pysynth import SyntheticDID

model = SyntheticDID()
estimate = model.fit(panel)
uncertainty = model.inference(
    panel, estimate=estimate, method="placebo", replications=200, random_state=0
)
print(estimate.att, uncertainty.standard_error)
print(uncertainty.confidence_interval(level=0.95))
```

The default number of bootstrap or placebo draws is 200, as in R synthdid.
Each bootstrap and placebo draw re-estimates both sets of weights using the
reference Frank–Wolfe solver, its two optimization stages and its complete
iteration budget. No reduced iteration count or fixed-weight approximation is
used for these two procedures.

| Procedure | Sampling and weight fitting | Availability |
|---|---|---|
| `bootstrap` | Sample all units with replacement; reject samples missing either group; re-estimate donor and time weights. | At least two treated units. |
| `jackknife` | Leave each unit out; normalize the remaining fitted donor weights and keep time weights fixed. | At least two treated units and two donors with nonzero weight. |
| `placebo` | Randomly permute the original controls, assigning the last `n_treated` as pseudo-treated; re-estimate both sets of weights. | More controls than treated units; supports one treated unit. |

The fixed-weight jackknife follows the authors’ R implementation of
Algorithm 3, with the centering convention described below. Bootstrap and
placebo preserve the original fit's penalties and stopping settings and warm-start
from its fitted weights. The original treated observations never enter a placebo
resample. Reference bootstrap and placebo variance is the population variance of
the resulting draw estimates; jackknife variance is `(n_units - 1)` times the
population variance of its leave-one-unit-out estimates, centered on their
mean. The paper’s Algorithm 3 and appendix equation (11.3) instead center on
the original ATT. These need not coincide in finite samples: the literal-paper
variance is the reference variance plus `(n_units - 1) * (mean(draws) - ATT)**2`.
We retain R’s convention to reproduce its reported software results and label
it explicitly; no claim of exact equality of those two formulas is made.

As in the pinned R resampling implementation, resampled optimizations use the
default sparsification step and 100 first-stage iterations even when the original
point estimate was fit with `sparsify=False`. The full default final-stage budget
is 10,000 iterations. A draw reaching that limit remains available, as it does in
R; the Python API reports one warning with the number of affected draws and keeps
complete convergence diagnostics in `uncertainty.metadata["solver_diagnostics"]`.
The original penalties are not recalculated from each resample.

`uncertainty.draws` contains every resampled ATT, and `uncertainty.resamples`
contains the zero-based unit indices. Passing those indices back through
`resamples=...` reproduces the draws exactly. Identical numeric seeds in NumPy and
R do **not** define identical resampling plans. The independent R fixtures use
identical explicit plans, including bootstrap draws with different treated counts,
and verify every ATT and all three variances to an absolute tolerance of `1e-10`.
The numerical fixtures are checked by the normal Python test suite. With the
separately supplied [local reference bundle](reference_backends.md), regenerate
the independent comparison with:

```bash
python .local/reference/scripts/validate_sdid_inference.py
```

This requires `Rscript` and the R `jsonlite` package and downloads the pinned
reference functions using `requests`. Use `--reference-dir PATH` to reuse those
four downloaded R files without network access.

The reported interval uses an asymptotic normal approximation, not percentiles of
the draws. Validity depends on the corresponding SDID assumptions; placebo
variance in particular requires comparable treated and control error variation.
A finite-draw standard error need not equal a paper's rounded standard error
unless its resampling plan is also available. The procedures currently cover the
balanced, simultaneous-treatment SDID model without covariates.

`SyntheticDID.fit_sc(panel)` separately reproduces the no-covariate
`synthdid::sc_estimate` comparator: no donor-fit intercept, an infinitesimal donor
penalty and zero time weights. `DifferenceInDifferences` reproduces the reference
DID point estimate. The variance API above is restricted to SDID fits.

### Reproducing California's published standard error

The California paper example has an explicit resampling seed in the authors'
[replication vignette](https://github.com/synth-inference/synthdid/blob/70c1ce3eac58e28c30b67435ca377bb48baa9b8a/vignettes/paper-results.Rmd).
Use this helper when replicating Table 1:

```python
from pysynth.replication import california_paper_inference

uncertainty = california_paper_inference(panel, estimate)
print(uncertainty.standard_error)  # 8.36799308138054, rounded to 8.4 in Table 1
```

The helper validates the original 38-donor alphabetical order, California as the
single treated state, and the 1970–2000 window with 1989 onset. It starts an
isolated `Rscript --vanilla` process using R's Mersenne-Twister generator,
Inversion normal generation and Rejection sampling, with R seed 12345.
The local bundle must supply its RNG helper; base R suffices without extra R
packages. All 200 placebo
fits then run in Python with the original SDID settings. Source, RNG and seed
are recorded in `uncertainty.metadata["paper_replication"]`.

This reproduces the paper's **8.4** after rounding. NumPy's seed-0 plan yields
**10.27674583455605** instead; replaying that exact plan in R gives the same
number. The difference is finite-resample variation, not a different variance
algorithm. The paper-plan standard error and every underlying draw have been
checked independently against R; the academic integration test requires an
absolute standard-error error below `1e-9`.

## Generic in-space placebo diagnostic

```python
from pysynth import SyntheticControl, placebo_test

diagnostic = placebo_test(SyntheticControl(), panel, max_pre_rmse_ratio=5)
print(diagnostic.p_value)
print(diagnostic.table)
```

Each original donor becomes pseudo-treated in turn. The actually treated
unit is excluded from every placebo donor pool, as is the pseudo-treated
unit itself. The estimator is refitted and any failure is raised. The same
treatment time is used. Models with a fixed custom Dataprep are not suitable
for this generic helper; use a method that can fit each new Panel.

The statistic is post-RMSPE divided by pre-RMSPE. The optional cutoff excludes
placebos whose pre-RMSE exceeds the given multiple of the treated pre-RMSE.
The treated fit is always retained. With $B$ eligible placebo statistics,
the reported rank is

$$p=\frac{1+\#\{\text{placebo statistic}\geq\text{treated statistic}\}}{1+B}.$$

Ties count in the tail. A zero pre-gap with a nonzero post-gap gives an
infinite statistic; both gaps exactly zero gives zero. No fixed cutoff in
outcome units is used, so changing the measurement scale does not change
the diagnostic. RMSPE is calculated with scaled residuals to avoid underflow
or overflow when squaring. Inspect the table when fit is effectively
perfect or when no placebo passes the cutoff (then the rank is one).

The actual treated fit has $N$ donors, whereas each placebo fit has $N-1$
because the actual treated unit is excluded. Thus the fitting procedure is
not symmetric across assignments: exchangeability alone does not make this
rank an exact randomization p-value. Treat it as an in-space diagnostic and
assess donor eligibility and the research design separately. The cutoff changes the reference set and should be chosen transparently.

This generic placebo diagnostic is distinct from the SDID placebo variance
estimator above. Conformal inference as a common API, time-block resampling and
survey-design variance are not implemented. Selected backend methods expose
additional procedures on their native fitted object; consult their assumptions
and documentation.

`IndividualSC(n_bootstrap=..., random_state=...)` and
`TwoStageIndividualSC(n_neighbors=..., n_bootstrap=..., random_state=...)`
return donor-resampled ATT draws in metadata and average the fitted paths.
ISC resamples its full eligible pool; 2SISC resamples the selected nearest
donors without repeating screening. Those draws measure conditional donor
sensitivity and omit selection uncertainty. They are not confidence
intervals, and shared donors can induce dependence between individual fits.
