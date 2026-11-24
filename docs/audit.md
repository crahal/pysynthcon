# Estimator and replication audit

This audit checks the statistical specifications, reference implementations,
empirical comparisons, numerical failure handling and reproducibility of the
library. It distinguishes reproduction of a paper's study from reproduction of
its authors' software and from applying that method to a new dataset.

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

The independent Basque run uses R Synth 1.1-10 and the authors' documented
paper Table 3 preparation. Its 1975–1997 mean effect is **−0.6995627076** thousand 1986 US
dollars. The Python fit gives **−0.6995651251**, with Catalonia/Madrid weights
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

Selected changes in the regenerated comparison estimates are shown below.
These are average post-treatment effects using each notebook's stated window;
they are applications of the estimators, not additional published targets.

| Example and estimator | Before audit | After audit | Main reason |
|---|---:|---:|---|
| Basque, ISC | −0.816493 | −0.859736 | Remove Spain's aggregate from eligible donors. |
| Basque, augmented SCM | −1.226881 | −1.289182 | Use the corrected regional donor pool. |
| Germany, outcome SCM | −1,304.70 | −1,297.47 | Correct the scale-dependent numerical objective. |
| Germany, pysyncon SCM | −1,448.65 | −1,355.82 | Recover the previously unsuccessful outer optimization. |
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
- **ISC and 2SISC:** costofcare's 50 replacement donor samples, predictor-series
  groups, duplicate-donor aggregation, sampling/optimization seeds and case
  aggregation agree with the audited source. The original standalone ISC
  demonstration uses ten draws; this remains an explicit option. 2SISC adds
  the requested nearest-neighbor screen before the same ISC procedure.
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
estimator. 2SISC is a requested extension, with no published numerical target.
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

The complete provisioned suite passes **245 tests**, including online academic
references and local original-R integration. Independent SDID reruns match all
inference draws within `2.76e-14`; SCUL screening matches 1,840 direct caret
column/window decisions. All 16 canonical pysyncon adapter fits pass checked
termination. Notebook source checks compile every code cell and parse generated
diagnostic JSON. All four empirical notebooks were re-executed successfully:
48 comparison fits, twelve embedded figures, ten starts per SyntheticControlMethods
fit, 50 draws per ISC/2SISC fit and 200 SDID inference draws per example.
Strict Sphinx, Ruff and distribution builds pass; archive
inspection finds no R code, private helpers or generated empirical exports.

## Primary sources

- [Abadie and Gardeazabal (2003), pp. 116–118](https://economics.mit.edu/sites/default/files/publications/The%20Economic%20Costs%20of%20Conflict.pdf#page=4)
  and the [authors' Synth preparation example](https://cran.r-project.org/web/packages/Synth/refman/Synth.html#dataprep).
- [SCUL setup and preprocessing source](https://github.com/hollina/scul/tree/121b588cf8e4eebcea68518b57a7c754040153bb/R).
- [costofcare replication](https://github.com/centre-for-care/costofcare/tree/6e7537d67b86fb08709d35b035c9ee9f45e83e93/replication_attempt/src).
- [SDID paper](https://arxiv.org/pdf/1812.09970v4),
  [point estimator](https://github.com/synth-inference/synthdid/blob/70c1ce3eac58e28c30b67435ca377bb48baa9b8a/R/synthdid.R)
  and [variance implementation](https://github.com/synth-inference/synthdid/blob/70c1ce3eac58e28c30b67435ca377bb48baa9b8a/R/vcov.R).
- [Authors' ASCM ridge validation](https://github.com/ebenmichael/augsynth/blob/7e70072232fe75057aa8b5d7af08c7e1aef53ae8/R/ridge_lambda.R).
- [Amjad, Shah and Shen (2018), Algorithm 1](https://jmlr.org/papers/v19/17-777.html).
- [ADH (2015)](https://economics.mit.edu/sites/default/files/publications/Comparative%20Politics%20and%20the%20Synthetic%20Control.pdf)
  and the [2025 erratum](https://economics.mit.edu/sites/default/files/2025-09/Comparative%20Politics%20and%20the%20Synthetic%20Control%20Method%20erratum.pdf).
