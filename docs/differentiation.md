# What PySynthCon adds

PySynthCon is a framework for comparing synthetic-control specifications across
estimator families, checking their sensitivity and tracing results back to
academic reference implementations. Its contribution is the shared research
workflow: one panel definition, one effect convention, comparable diagnostics
and publication figures for SCM, SCUL, ISC/2SISC and SDID. Implementing these
methods together does not establish a new identification result or a universally
better estimator.

## Relationship to existing software

The following comparison covers the linked public documentation and source
reviewed on 24 November 2026. It describes scope, rather than claiming that a
feature cannot be implemented with another package.

| Project | Existing strengths | PySynthCon's additional role |
|---|---|---|
| [pysyncon](https://sdfordham.github.io/pysyncon/) | Classical, augmented, robust and penalized SCM; placebo tests, conformal intervals and academic examples. | Integrates those estimators with SCUL, MSCMT-based individual controls and SDID in a common result and comparison workflow. |
| [SyntheticControlMethods](https://github.com/OscarEngelbrektson/SyntheticControlMethods) | SCM and differenced SCM, predictor balance, effect plots and in-space/in-time placebos. | Applies the same study definition and diagnostics to this backend and the other estimator families, with checked optimizer recovery. |
| [synthdid](https://synth-inference.github.io/synthdid/reference/index.html) | SC/DID/SDID estimation, bootstrap/jackknife/placebo variance, weight diagnostics and multi-estimate plots. | Supplies validated Python point estimation and inference alongside the other methods, including explicit paper-versus-software conventions. |
| [CausalPy](https://causalpy.readthedocs.io/en/latest/notebooks/sensitivity_checks.html) | A broad causal-analysis framework with donor deletion, convex-hull checks and placebo diagnostics. | Focuses on comparisons across the specific classical, lasso, individual and SDID reference algorithms supported here. Donor deletion and model diagnostics themselves are established tools. |
| [costofcare](https://github.com/centre-for-care/costofcare) | Individual-control study code and nearest-neighbor donor selection for the care application. | Exposes reusable case-specific panels, the authors' core two-stage fitting specification, common estimator comparisons and donor-pool sensitivity outside that study. |

Python SDID implementations also exist in
[synthdid.py](https://github.com/d2cml-ai/synthdid.py) and
[mlsynth](https://mlsynth.readthedocs.io/en/latest/sdid.html).
PySynthCon does not claim to be the first Python SDID implementation, the first
comparison package or the first package with sensitivity checks. The independent
numeric fixtures and [audit record](audit.md) establish the narrower claims made
about this implementation.

## Compare predictive performance before treatment

The common rolling-validation workflow reserves later pretreatment periods for
evaluation, refits each candidate on the earlier periods, and compares held-out
errors on identical folds. Candidates can represent estimator families or
prespecified tuning choices, such as the number of neighbors in 2SISC. Actual
post-treatment rows are excluded, and held-out treated outcomes are masked in
the panel passed to fitting. The genuine outcomes remain available only to the
scoring step.

```python
from pysynth import SyntheticControl, SyntheticDID, TwoStageIndividualSC, backtest
from pysynth.datasets import DATASETS, load_dataset

panel = DATASETS["california"].panel(load_dataset("california"))

estimators = {
    "SCM": SyntheticControl,
    "SDID": SyntheticDID,
    **{
        f"2SISC(k={k})": lambda k=k: TwoStageIndividualSC(n_neighbors=k)
        for k in (5, 10, 15)
    },
}
validation = backtest(panel, estimators, initial_window=10, horizon=3)
print(validation.summary)
print(validation.best_model)
```

`scores` records the individual fits and `predictions` retains held-out paths.
`select_estimator` runs this comparison and fits its selected candidate on the
complete panel, returning both the validation results and final estimate.
The 2SISC candidates require the optional pysyncon backend. Each value of $k$
is a research specification, not a reduction of the optimizer's search budget.

Validation-period donor outcomes remain available because synthetic control
estimates a contemporaneous untreated trajectory; this is a retrospective
counterfactual prediction exercise, not a forecast made without future donor
observations. Candidate preprocessing and external covariates must obey the same
fold boundaries. Masking also means that SCUL's original full-series
admissibility check sees placeholders during this exercise; the validation
protocol is deliberately separate from an unmasked historical replication.

A lower validation error supports a predictive comparison on those folds.
It does not establish causal identification, prove superiority on other panels,
or account for model-selection uncertainty in a subsequent confidence interval.
Candidates and folds should be chosen before examining treatment effects.

## Check dependence on donors and donor groups

The common sensitivity workflow removes each donor, or a named group of donors,
and fully refits every specified estimator. It reports changes in ATT and
pretreatment fit on the same outcome scale. A group can represent a region,
industry or a prespecified potential spillover set. Re-estimation is necessary:
zeroing and renormalizing existing weights would miss changes in predictor
weights, regularization and donor screening.

```python
from pysynth import donor_sensitivity

# California example: declare the alternative pools before inspecting effects.
sensitivity = donor_sensitivity(
    panel,
    {"SCM": SyntheticControl, "SDID": SyntheticDID},
    exclusions={"without Nevada and Utah": ["Nevada", "Utah"]},
)
print(sensitivity.table)
print(sensitivity.summary)
```

Omit `exclusions` to remove each donor in turn. Baseline estimates and every
scenario's refitted estimates remain available for inspecting their full paths
and optimizer metadata, rather than only the summary changes.

These changes measure dependence on the chosen donor pool. They are not
confidence intervals, a test of whether a donor is contaminated, or a license
to choose exclusions that produce a preferred effect. The original fitting
budgets remain in force, so a full sensitivity exercise can be expensive.

## Methodological attribution

Nearest-neighbor screening followed by individual synthetic control is described
by [Petrillo et al. (2025), section 2.3](https://eprints.whiterose.ac.uk/id/eprint/224118/1/Binder2025004.pdf).
PySynthCon's 2SISC name credits that approach; the default and optional variants
are documented in the [method guide](methods.md#two-stage-individual-synthetic-control-2sisc).
Related donor-locality approaches include
[Abadie and L'Hour's penalized SCM](https://doi.org/10.1080/01621459.2021.1971535)
and [ClusterSC](https://proceedings.mlr.press/v258/rho25a.html).
These are distinct algorithms, and none of their theoretical guarantees is
automatically transferred to PySynthCon's nearest-neighbor variant.

The contribution here is reusable, inspectable software around these methods.
A claim of methodological novelty or improved causal performance would require
additional theory and comparative evidence; the canonical notebooks are
replications and applications, not that evidence.
