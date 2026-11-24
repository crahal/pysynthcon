# Estimator and replication audit

This audit checks the statistical specifications, reference implementations,
empirical comparisons, numerical failure handling and reproducibility of the
library. It distinguishes reproduction of a paper's study from reproduction of
its authors' software and from applying that method to a new dataset.

## Current method catalog

The Basque, California, Germany and Kansas estimator comparisons each include eleven specifications. Predictor-based SCM
uses means over three pretreatment blocks. Ridge-augmented SCM, penalized SCM,
robust SCM and differenced SCM have separate interfaces and paper citations.
The comparison counts and backend-specific changes recorded in the dated
audits below describe the earlier notebook lineup. [Methods](methods.md)

## Re-audit of fits and inference designs

- **A distant donor could hide an incorrect solution.** A treated series with pretreatment values `[0.9, 0]` can be matched exactly by weights 0.1 and 0.9 on donors with histories `[0, 0]` and `[1, 0]`. Adding an unused donor with history `[0, 1000]` caused the scaled solver to stop near equal weights. For post-treatment donor outcomes 0, 20 and 0 and a treated outcome of 18, that produced a spurious effect of +8. The solver now checks whether a feasible transfer between donors materially improves the original objective on the scale of the fitted donors. The exact match recovers an effect of zero. Transfers share the original iteration budget and appear in the diagnostics.
- **A text configuration could enable look-ahead.** `allow_post_predictors="False"` previously evaluated as true. Predictor adapters now require a boolean for this switch. Germany’s documented historical exception uses the explicit boolean `True`.
- **Downstream estimates were checked.** The four full-sample outcome-SCM ATTs stayed the same. Excluding Austria increased the revised West Germany ATT by $3.24 per person relative to the previous leave-Austria-out fit. The largest change to an individual simulated ATT was 0.000020 points across the original 800 paired-design rows. Kansas pooled SCM validation RMSE moved from 0.026644 to 0.026633 log points.
- **An exact interpolation should give a tied conformal rank.** In a deterministic panel a convex blend fits the entire null-adjusted series exactly. Floating-point residuals alone produced a p-value of 1/60. The implementation checks interpolation at row-specific numerical precision and verifies feasibility for near-perfect fits. The regression example now gives a p-value of 1 under rescaling and common level shifts.
- **Two kinds of placebo require different evidence.** The tobacco paper assigns each state the intervention with every other state available as a donor. Our original helper always excluded the real treated state from the placebo donor pools. Both designs now have explicit names. In the source predictor and state order, R Synth and the Python optimizer both rank California first of 39, but retain 27 and 26 placebo states at the 5× cutoff. Figure 6 reports 29. The complete bridge run also agrees with an independent native R data preparation and fit for every assigned state. The time-permutation analysis separately checks the Rhode Island paper’s four SCM p-values and its authors’ software-generated endpoints. [Replications and sources](replication.md)

## November 24 follow-up audit

We tested the public estimator, comparison, inference and plotting interfaces
with analytic panels, malformed results, shared prediction buffers and the
pinned datasets. Separately executed R checks covered SCUL and the SDID point
and variance estimators. The following issues affected what a researcher
could conclude from the output.

| Reproducible finding | Correction |
| --- | --- |
| Adding a common level of 100,000 to an exact two-donor blend moved the fitted weights from 0.6/0.4 to roughly 0.5/0.5 even though pysyncon's solver reported success. | Check the attainable improvement along a feasible simplex direction. Recover the same inner quadratic if that improvement exceeds its tolerance. In the Basque common-sample all-lag predictor SCM this changes the ATT from −0.973575 to −0.993457 thousand 1986 US dollars per capita. |
| A zero ridge penalty on a singular AugSynth design could produce weights summing to −14. A different, valid fit could crash while reporting a ratio with a zero denominator. | Check the rank required by the reference inverse. Retain completed weights when only the descriptive scaled imbalance is undefined. |
| A missing final counterfactual could silently drop a period from the ATT and produce a placebo rank. An estimator reusing the same output buffer could overwrite earlier donor-exclusion fits. | Check labels, coverage and finite outcomes before scoring. Snapshot each sensitivity fit and placebo fit. Test explicit donor labels and extreme ratios. |
| With `CA`, `C` and `A` in the data, `treated_units="CA"` could select two treated units and move California into the donor pool. | Reject bare strings in arguments for collections of unit IDs. Both `treated_unit="CA"` and `treated_units=["CA"]` identify California correctly. |
| Supplied SDID results could contain a counterfactual that disagreed with the stored weights. | Reconstruct the offset and counterfactual from the supplied panel and weights before using a fitted estimate for inference. |
| A plausible average could pass a replication check on a different calendar or conceal quarterly errors in Kansas. | Require the historical time index and treatment boundary. Check the Kansas ATT and all sixteen quarterly effects at their respective printed precisions. See the [authors' vignette](https://github.com/ebenmichael/augsynth/blob/7e70072232fe75057aa8b5d7af08c7e1aef53ae8/vignettes/singlesynth-vignette.md#L217-L241). |
| The Basque figure reported a numerical difference between a qualitative paper headline and a calculated percentage mean. | Show the Figure 1 headline beside the 1980–1997 mean calculated using the Figure 2 denominator. Reserve a numerical difference for the matched 1975–1997 original-R comparison. See [Abadie and Gardeazabal, pp. 117–118](https://economics.mit.edu/sites/default/files/publications/The%20Economic%20Costs%20of%20Conflict.pdf#page=5). |

The high-level replication checks use source precision or the recorded
software tolerances. Optimizer checks establish feasibility or detect a
specific improvement. The nested predictor-weight problem can still have
multiple solutions.

Among the 48 common-sample rows, only the Basque all-lag predictor SCM fit changed
at the precision of the exported estimates. Its pretreatment RMSE fell from
0.064482 to 0.064345 thousand 1986 US dollars per capita. The number of donor
weights above the reporting threshold of `1e-6` fell from five to three. The separately
specified historical Basque SCM still reproduces the published donor weights.
In one Kansas validation fold, the same five selected 2SISC donors admitted
a feasible inner-loss decrease of `1.44e-6`, exceeding the solver tolerance
of `1e-6`. Recovering that solve changed Missouri's weight from 0.645 to
0.833. The candidate's pooled held-out RMSE fell from 0.0490735 to
0.0490474 log points, preserving its rank of seventh among eight candidates.
The other 71 candidate-fold results and all donor-exclusion results matched
the prior exports. Across the four examples, all 1,600 original- and extended-budget
SDID draw values and their stopping diagnostics matched the previous run.
The source-comparison CSVs preserve numerical precision alongside
the shorter labels printed on the figures.

## Errors corrected

| Finding | Why it mattered | Correction and evidence |
|---|---|---|
| The Basque donor pool included Spain's national aggregate. | The aggregate contains the treated region; the paper specifies the other **16 regions**. A zero weight in one SCM fit does not make it eligible for other estimators. | All Basque comparisons now exclude it. Source: Abadie–Gardeazabal (2003), p. 116, and the authors' Synth example. |
| The Basque historical fit used five education headcounts. | The authors combine the highest two categories and use four annual percentage shares averaged over 1964–1969. Matching a headline ATT did not verify those inputs. | Corrected preprocessing, without changing downloaded data. Independent R Synth checks now cover the four shares, donor weights and full counterfactual path. |
| Native simplex SCM was scaled by outcome levels and an absolute floor. | Large common level shifts or small outcome units could cause premature convergence to incorrect weights. | Optimize the equivalent donor-minus-treated discrepancy objective, with data-relative scaling. Analytic mixture and invariance tests verify the solution. |
| The SyntheticControlMethods adapter discarded lags below an absolute standard-deviation cutoff. | Changing measurement units could remove informative predictors. | Retain the requested lags and handle exactly constant predictors without an outcome-unit cutoff. Full ten-start fitting remains enabled. |
| Custom pysyncon predictors could use post-treatment observations silently. | Validated fitting dates alone did not protect predictor construction. | Reject post-treatment predictor windows by default. Germany's documented historical 1990 convention requires an explicit opt-in, recorded in metadata. |
| Upstream pysyncon discarded unsuccessful optimizer status. | California/Germany ASCM returned failed inner SLSQP iterates; Germany outcome-predictor SCM could silently stop at its outer iteration cap. | Check both stages, recover the same inner quadratic with CLARABEL and default outer search with Powell; retain diagnostics and raise if recovery fails. |
| Robust SCM inherited upstream missingness and truncation errors. | Genuine observed zeros were counted as missing; threshold selection could omit the final singular value. | Use the complete Panel's observation fraction of one, evaluate all singular values and predict from the denoised donor matrix, following Amjad–Shah–Shen Algorithm 1. |
| SCUL screened only exactly constant donors in its first training window. | Original setup also excludes rare-value columns in the full series and every training window. | Reproduce caret's frequency/uniqueness rule. Direct comparison checks cover rare binary columns, later-window exclusions and threshold boundaries. |
| SCUL rejected every single-validation-origin design. | The original median and minimum selectors can use one origin; only its sample-SD selector needs two. | Allow one origin for median/minimum, preserving the original visible warning about an unused undefined SD candidate. |
| Generic placebo diagnostics used an absolute `1e-12` cutoff and directly squared residuals. | Rescaling the outcome could change ranks; extreme units could underflow or overflow RMSPE. | Use exact-zero conventions and scaled RMSPE calculations. Tests verify unit invariance and genuine zero-gap cases. |
| SDID inference trusted the positional order of supplied time weights. | A reordered mutable result could silently use weights for the wrong periods. | Validate time labels, finite nonnegative weights and simplex normalization before inference. |
| Source distributions could include restored ignored helper files. | setuptools inclusion patterns do not honor Git ignore rules. | Explicitly exclude old local helper paths, private reference resources and executed-notebook copies; inspect built archives. |
| Local notebook diagnostics ended with a literal backslash-n. | The exported file was not valid JSON. | Write an actual newline; generated-code tests compile every notebook and parse the diagnostic output. |
| The notebook runner retained completed kernels. | Supplying an external kernel manager disables nbclient's default cleanup, leaving earlier notebooks' processes and channels alive during a long run. | Explicitly request cleanup; real-kernel tests verify shutdown and connection-resource removal after both success and failure. |
| Documentation called 2SISC an unpublished extension and its default combined screening with the separate MSCMT replication workflow. | Petrillo et al. (2025) already describe the approach; their two-stage Python code fits temporal-mean predictors without resampling donors. | Credit the paper and make that fitting specification the default. Direct KDTree/pysyncon comparison tests verify multivariate fits. Preserve the MSCMT/resampling variant explicitly. |
| SDID inference warned about capped draws without a direct way to extend an existing fit's inherited budget. | Setting a larger budget on a new estimator object did not affect inference supplied with the old estimate; the resampled fits could remain capped. | Add an explicit inference budget override, actionable warnings and a strict check. Re-run identical plans with the same algorithm, tolerance and penalties; independently verify extended fits against original R. |
| Repeated reference tables occupied much of each figure above the plotted data. | Source comparisons were visually separated from the matching estimates, and unrelated source specifications appeared above single-method paths. | Use direct curve labels and matched source callouts. Add separate historical reference rows/panels when needed, preserve incompatible units/windows as text, and retain full audit tables in expandable notebook sections. |

The independent Basque run uses R Synth 1.1-10 and the authors' documented
paper Table 3 preparation. Its 1975–1997 mean effect is **−0.6995627076** thousand 1986 US
dollars. The Python fit gives **−0.6995652172**, with Catalonia/Madrid weights
**0.850814/0.149186**; the paper reports **0.8508/0.1492**. The numeric fixture
records its software versions, source hash and reference path. The absolute
comparison tolerance is `1e-4` for paths, weights and ATT. These software
figures are not attributed to the paper at invented precision. The later Synth
manual normalizes temporally averaged counts instead; its schooling predictors
differ slightly from Table 3. This audit tested both conventions and chose the
paper’s annual-share means for the primary replication.

The Basque correction changes the donor pool for all twelve comparison rows.
Those rows are new applications of their respective methods; the historical
predictor fit is shown separately. See the executed notebook for the regenerated
estimates and the [replication guide](replication.md) for matching windows.

Selected changes from the earlier audit are shown below.
These are average post-treatment effects using each notebook's stated window;
they are applications of the estimators, not additional published targets.

| Example and estimator | Before audit | After audit | Main reason |
|---|---:|---:|---|
| Basque, ISC | −0.816493 | −0.859736 | Remove Spain's aggregate from eligible donors. |
| Basque, augmented SCM | −1.226881 | −1.289182 | Use the corrected regional donor pool. |
| Germany, outcome SCM | −1,304.70 | −1,297.47 | Correct the scale-dependent numerical objective. |
| Germany, all-lag predictor SCM | −1,448.65 | −1,355.82 | Recover the previously unsuccessful outer optimization. |
| Germany, augmented SCM | −1,495.09 | −1,333.11 | Recover unsuccessful inner quadratic solves. |

Basque units are thousands of 1986 US dollars per capita; Germany units are
PPP current US dollars per capita. California's SDID estimate remains
−15.603828 cigarette packs per capita. Small numerical differences can also
arise when equivalent objective scaling changes a nonconvex optimizer's path.

## Verified conventions retained

- **SDID:** independent execution of the pinned R source agrees with Python's
  SC, DID and SDID point estimates and draw-by-draw bootstrap, jackknife and
  placebo calculations. California's published point estimate and its exact
  seed-12345 placebo standard error remain the reference targets.
- **SDID paper versus software:** equation (2.2) uses a population noise
  denominator, whereas the authors' R code uses a sample denominator.
  Algorithm 3 and appendix equation (11.3) center jackknife deletions on the
  original estimate, whereas R centers on the deletion mean. The default
  intentionally reproduces R. For the independent test panel, the variances
  are respectively 1.2480391989 and 1.2465762419. The difference is documented,
  and the result metadata identifies the centering convention.
- **Ridge ASCM:** the authors' `ridge_lambda.R` omits the final possible
  holdout block. The pinned pysyncon implementation follows that convention;
  changing it would break software parity. Its lambda grid and one-standard-
  error selection were checked against the authors' implementation.
- **ISC:** costofcare's 50 replacement donor samples, predictor-series
  groups, duplicate-donor aggregation, sampling/optimization seeds and case
  aggregation agree with the audited source. The original standalone ISC
  demonstration uses ten draws; this remains an explicit option. These checks
  also cover the explicit MSCMT variant of 2SISC. The default now follows
  the separate [two-stage Python fitting specification](methods.md#two-stage-individual-synthetic-control-2sisc).
- **Germany:** including 1990 in historical predictor/validation windows is
  stated in the paper. It remains an explicit historical exception. Units
  remain PPP current US dollars, consistent with the authors' 2025 erratum.

## Limits that remain explicit

The generic in-space placebo rank is a diagnostic. Its actual fit has one
more donor than each placebo fit; exchangeability alone does not make that
asymmetric fitting procedure an exact randomization test. It is distinct
from SDID's published placebo variance procedure.

SCUL's original full-series admissibility screen can depend on post-treatment
values, including the treated outcome. Its regression and penalty selection
use pretreatment outcomes. This distinction is now stated rather than claiming
that every preprocessing decision is independent of treated post-outcomes.

The aggregate examples do not reproduce the original SCUL scanner-data study
or ISC motherhood/care studies. Those require the study-specific data,
eligibility rules, covariates and population-inference design. ISC donor draws
are sensitivity outputs, not the published study's full population-variance
estimator. The 2SISC aggregate examples have no numerical target in the care
study. Matching the core fitting specification does not reproduce its survey
eligibility, event windows, covariates or population inference. PySynthCon counts
donors explicitly; the source's neighbor query also counts the treated case.
Penalized and robust SCM notebook tuning values remain explicitly illustrative.

Nested SCM remains nonconvex; full optimization and successful numerical checks
do not prove a global optimum. Original SDID iteration-cap warnings remain
visible where the reference algorithm itself reaches its cap. No predictors,
starts, resamples or iteration budgets are removed to accelerate the notebooks.

R resources and auxiliary exports remain local-only. Executed notebooks retain
their figures; tests retain small independent numeric fixtures. A fresh clone
requires the separately supplied local bundle for SCUL/ISC and California's
exact R sampling plan.

## Validation

At the time of the estimator audit the provisioned suite passed **461 tests**, including online academic
references and local original-R integration. Six warnings record expected
SDID iteration caps in original-budget tests, and one comes from Matplotlib's
use of a deprecated pyparsing API. Separately, 154 panel, results, placebo,
validation, sensitivity and evaluation-plotting tests passed in a second
environment with NumPy 2.5.4 and SciPy 1.16.2. Independent SDID reruns match all
inference draws within `2.76e-14`; SCUL screening matches 1,840 direct caret
column/window decisions. All 16 canonical pysyncon adapter fits pass checked
termination. Across the four notebooks, all 80 starts for SyntheticControlMethods
SCM and differenced SCM ended with a successful solve. The optimizer diagnostics
retain each attempted solver and the reason for any recovery.
Notebook source checks compile every code cell and parse generated
diagnostic JSON. All four empirical notebooks were re-executed successfully:
48 comparison fits, twenty embedded figures, ten starts per SyntheticControlMethods
fit, 50 draws per ISC fit, the authors' non-resampling default 2SISC specification,
and two complete 200-draw SDID inference runs per example: the original
10,000-iteration cap and the identical plan with a 1,000,000-iteration cap.

The extended-budget implementation also matches six independently computed
original-R bootstrap/placebo draws within `8.9e-15`, with identical stopping
iterations, including a time-weight fit requiring 123,992 iterations. The
original point estimate, penalties, tolerance, initialization and sparsification
remain fixed in this comparison. Strict convergence concerns resampled weight
fits and does not certify global optimality.

All 800 extended-budget draws meet the original stopping rule for both weight
fits. The largest observed iteration count is 553,329 in Kansas. The original
cap results remain available alongside the extended results:

| Example (outcome units) | Capped draws, original / extended | Original-cap SE | Extended-cap SE |
| --- | ---: | ---: | ---: |
| Basque (thousand 1986 US dollars) | 83 / 0 | 0.3500574651 | 0.3500762154 |
| California (packs per capita) | 109 / 0 | 8.3679930814 | 8.3689443673 |
| Germany (PPP current US dollars) | 114 / 0 | 1488.2162255643 | 1488.6588309995 |
| Kansas (log GDP per capita) | 156 / 0 | 0.0337620348 | 0.0337703860 |

Each row uses 200 identical original/extended resamples. California retains
the authors' R sampling plan; its extended SE still rounds to the published
8.4. Every extended California draw also matches a separate verification run.

The new evaluation sections compare eight candidates using 72 complete fold
fits and four fresh selected-model fits. Basque, California and Germany each
have two held-out observations per candidate; Kansas has 45 across three
folds. These sample sizes are displayed on the figures. The sensitivity sections
perform 357 donor-exclusion refits plus twelve full-pool baselines, covering
every eligible donor for outcome SCM, 2SISC and SDID. Independently recomputed
fold errors, pooled scores, ATT changes and scenario extrema agree with the
exported tables. Tests cover target masking, labels, fresh instances, full
refits, immutable diagnostics and contextual failures. Original SDID
iteration-cap warnings remain visible in validation as well as point fits.

All twenty figures now use direct annotations. Visual review checks source
associations, evaluation windows, units and label placement; regression tests
cover historical reference panels, dense multiline comparisons, callout
containment and annotation targets. Full specifications and numerical audit
tables remain in expandable notebook sections after their figures.

Strict Sphinx, Ruff and distribution builds pass; archive
inspection finds no R code, private helpers or generated empirical exports.

## Primary sources

- [Abadie and Gardeazabal (2003), pp. 116–118](https://economics.mit.edu/sites/default/files/publications/The%20Economic%20Costs%20of%20Conflict.pdf#page=4)
  and the [authors' Synth preparation example](https://cran.r-project.org/web/packages/Synth/refman/Synth.html#dataprep).
- [SCUL setup and preprocessing source](https://github.com/hollina/scul/tree/121b588cf8e4eebcea68518b57a7c754040153bb/R).
- [costofcare replication](https://github.com/centre-for-care/costofcare/tree/6e7537d67b86fb08709d35b035c9ee9f45e83e93/replication_attempt/src).
- [Petrillo et al. (2025), two-stage individual synthetic control](https://eprints.whiterose.ac.uk/id/eprint/224118/1/Binder2025004.pdf)
  and the [two-stage Python source](https://github.com/centre-for-care/costofcare/blob/6e7537d67b86fb08709d35b035c9ee9f45e83e93/src/utils.py).
- [SDID paper](https://arxiv.org/pdf/1812.09970v4),
  [point estimator](https://github.com/synth-inference/synthdid/blob/70c1ce3eac58e28c30b67435ca377bb48baa9b8a/R/synthdid.R)
  and [variance implementation](https://github.com/synth-inference/synthdid/blob/70c1ce3eac58e28c30b67435ca377bb48baa9b8a/R/vcov.R).
- [Authors' ASCM ridge validation](https://github.com/ebenmichael/augsynth/blob/7e70072232fe75057aa8b5d7af08c7e1aef53ae8/R/ridge_lambda.R).
- [Amjad, Shah and Shen (2018), Algorithm 1](https://jmlr.org/papers/v19/17-777.html).
- [ADH (2015)](https://economics.mit.edu/sites/default/files/publications/Comparative%20Politics%20and%20the%20Synthetic%20Control.pdf)
  and the [2025 erratum](https://economics.mit.edu/sites/default/files/2025-09/Comparative%20Politics%20and%20the%20Synthetic%20Control%20Method%20erratum.pdf).
