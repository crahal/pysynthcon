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
    panel, estimate=estimate, method="placebo", replications=200, random_state=0,
    max_iter=1_000_000, require_convergence=True,
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
When you pass an existing fit, the library verifies its sample, labels,
simplex weights, and the counterfactual path and level correction implied by
those weights before using its ATT to center an interval.

### Handling iteration-limit warnings

The original reference algorithm can reach its 10,000-iteration cap before
meeting its objective-decrease stopping rule. This is a genuine numerical
limit, not a failed resampling plan. R returns those fits silently; PySynthCon
reports the number of affected draws and which weight fits reached the cap.

Use the **inference keyword** `max_iter=1_000_000`, as in the example above,
to extend the final-stage budget. This leaves the objective, penalties,
tolerance, initial weights and 100-step sparsification stage unchanged.
It accepts only an increase from the supplied estimate's budget. When an
existing `estimate` is supplied, constructing `SyntheticDID(max_iter=...)`
alone does not change its inherited inference settings; the explicit keyword
avoids that ambiguity. Jackknife has no iterative refits and rejects the override.

`require_convergence=True` raises if any resampled weight fit still reaches
the cap. It never discards a draw or returns a partial standard error. The
default remains `False` for compatibility with the authors' finite-iteration
reference output. A million iterations is a budget, not a guarantee: inspect
the result or increase it further if the strict check fails. Meeting the
original objective-decrease rule is not a certificate of global optimality.

The result records `original_max_iter`, `max_iter`, `capped_draws` and
`capped_solvers`. Each draw's two solver records include iterations, objective,
last objective decrease, threshold, dual gap and termination reason. Export
them with `pandas.json_normalize(uncertainty.metadata["solver_diagnostics"])`.

For a numerical sensitivity comparison, replay the exact `resamples` from
the original-cap run with the larger budget. The point estimate remains
fixed; the strict check concerns resampled weights. A higher-budget point
fit is a separate `SyntheticDID(max_iter=...).fit(panel)` call. The notebooks
show original-cap and extended-cap standard errors side by side and require
all 200 draws of the extended run to meet the original stopping criterion.
The extended result is explicitly separate from exact finite-iteration
replication of the paper software.

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

`california_paper_inference` deliberately retains the authors' default cap.
For a convergence check, pass its `resamples` to `SyntheticDID().inference`
with `max_iter=1_000_000` and `require_convergence=True`, while retaining the
original result for the exact reference assertion. Do not replace the
reference target or widen its tolerance to accommodate the extended run.

## In-space placebo paths and ranks

For a study with one treated unit, `placebo_test` repeats the fit with every unit assigned to treatment in turn. Use `donor_pool="reassign"` for the California design of Abadie, Diamond and Hainmueller. Each fit uses the same number of donors, including California when another state receives the hypothetical assignment. The default `"untreated"` design excludes the originally treated unit from all placebo donor pools. It is useful when its observed post-treatment outcomes would contaminate those counterfactual paths. [California study, Figures 4–8](https://economics.mit.edu/sites/default/files/publications/Synthetic%20Control%20Methods.pdf#page=9)

```python
from functools import partial
from pysynth import placebo_test, plot_placebo_gaps
from pysynth.datasets import DATASETS, load_dataset
from pysynth.replication import california_tobacco_estimator

data = load_dataset("california_tobacco")
panel = DATASETS["california_tobacco"].panel(data)
states = placebo_test(partial(california_tobacco_estimator, data, backend="r-synth"), panel,
                      donor_pool="reassign", n_jobs=4)
print(states.rank, len(states.table))
screened = states.with_pre_mspe_cutoff(5)
fig, ax = plot_placebo_gaps(screened, xlabel="Year", ylabel="Gap in packs per person")
```

The above example requires the optional local R Synth bridge. Choose `backend="python"` to run the checked Python predictor optimizer. [Setup](reference_backends.md#r-synth-for-the-california-reassignment-study)

An estimator instance is copied for each assignment. A factory that accepts the reassigned panel can rebuild its predictor specification for each hypothetical treated unit. Each assignment gets a complete fit. The default `n_jobs=1` runs serially. Increasing it distributes independent fits across processes and requires serializable estimators and results. Assignment order is preserved.

The rank counts eligible assignments with a post/pre MSPE ratio at least as large as the treated unit’s, including the treated unit and all ties. Divide by the number of eligible assignments to obtain `p_value`. Squaring the RMSPE ratio preserves its ordering. `table` exposes both scales. Zero pretreatment error gives an infinite ratio for nonzero post-treatment error and zero when both errors are zero. A zero cutoff denominator in a relative screen admits only other perfect pretreatment fits.

The California paper screens using mean **squared** gaps. `max_pre_mspe_ratio=5` allows five times the treated pre-MSPE. The older `max_pre_rmse_ratio=5` permits up to twenty-five times that MSPE. Choose one scale. `with_pre_mspe_cutoff(None)` restores every assignment. Changing the screen reuses the stored fits, and `gaps` always includes every unit.

An unfiltered reassignment rank has a randomization interpretation under a sharp null, uniform assignment among the eligible units and an assignment-symmetric fitting rule. In observational studies the assignment mechanism needs substantive justification. A treated-relative screen changes the comparison set and returns a descriptive rank. The default untreated-only scheme also gives the treated and placebo fits different pool sizes. SDID’s placebo *variance* estimator, described above, samples its original controls to estimate a standard error.

## Conformal inference over time

Chernozhukov, Wüthrich and Zhu test a **sharp path** for a single treated unit. Specify an effect for every post-treatment date. A scalar repeats the same hypothesis at each date. A vector or a date-indexed Series can describe a delayed or changing effect. The implementation uses their outcome-only, simplex-constrained synthetic control with no intercept. [Method and validity conditions](https://doi.org/10.1080/01621459.2021.1920957) · [Authors’ implementation](https://github.com/kwuthrich/scinference/blob/567c6889ce0a1d269a62d415b88aa6baf723a3fe/R/conformal.R)

```python
import numpy as np
from pysynth import conformal_confidence_set, conformal_test, plot_conformal_set
from pysynth.datasets import DATASETS, load_dataset

ri = DATASETS["rhode_island"].panel(load_dataset("rhode_island"))
joint = conformal_test(ri, null_effect=0)
print(joint.p_value, joint.rotations)  # Includes the identity shift

grid = conformal_confidence_set(ri, np.arange(-500, 201) / 100, alpha=0.10)
fig, ax = plot_conformal_set(grid)
```

1. Subtract the proposed effect path from the observed post-treatment outcomes. Estimate donor weights by least squares across **all** dates, with the null imposed. A conventional pretreatment estimate cannot supply these weights.
2. Form the observed score from the absolute residuals over the post-treatment window. By default the score is their sum. The API also accepts an L-q norm for `q >= 1`, including the maximum for `q=np.inf`. The authors’ replication code reports the unnormalised score. A common scale factor leaves the ranks unchanged.
3. Move the window through all cyclic shifts of the observed calendar, wrapping at the end. The fraction with score at least as large as the observed score is the p-value. `rotations` records each window, score and tail indicator. The result also retains the null-adjusted residuals and fitted weights.

With 25 dates the smallest joint p-value is 1/25. Invariance of the null residual distribution under the shifts gives finite-sample validity. With serial dependence the paper establishes approximate validity under additional conditions on dependence and counterfactual estimation. Enumerating all shifts removes simulation noise from the rank. It does not establish those conditions for a particular application.

### Inverting a test at a single date

For each post-treatment date, combine that date alone with the complete pretreatment history. For each supplied candidate effect, impose the candidate, refit SCM and keep it when its p-value exceeds `alpha`. The result records every tested candidate, including rejected values. Different dates can have disconnected or empty accepted sets. Acceptance at a grid boundary leaves the extent beyond that boundary unresolved. A finer or wider grid requires further fits. These are pointwise sets with no guarantee of simultaneous coverage across the post-period.

Rhode Island’s authors used 701 candidates from −5.00 to 2.00 log points for each of six years. Each single-date test uses 20 possible positions, giving p-values in steps of 0.05. The joint test uses all six years together and answers a different question. [Study, Figure 5 and Tables 1–2](https://arxiv.org/pdf/1712.09089#page=39) · [Replication materials](https://doi.org/10.6084/m9.figshare.14473870.v1)

## Donor resampling for individual controls

`IndividualSC(n_bootstrap=..., random_state=...)` and
`TwoStageIndividualSC(backend="mscmt", n_neighbors=..., n_bootstrap=..., random_state=...)`
return donor-resampled ATT draws in metadata and average the fitted paths.
ISC resamples its full eligible pool; 2SISC resamples the selected nearest
donors without repeating screening. Those draws measure conditional donor
sensitivity and omit selection uncertainty. They are not confidence
intervals, and shared donors can induce dependence between individual fits.
Default author-style 2SISC performs no within-donor resampling. The care paper's
between-person bootstrap is a separate population-inference procedure and is
not supplied by these donor-sensitivity draws.
