# Validation, selection and donor sensitivity

PySynthCon evaluates different estimator families through the same `Panel` and
`Estimate` interfaces. Three questions are kept separate: how well a model
predicts withheld pretreatment outcomes, which prespecified candidate to fit,
and how the final estimate changes when donor eligibility changes. These
workflows add no identifying assumption or new confidence-interval guarantee.

## Compare models on withheld pretreatment periods

```python
from functools import partial
from pysynth import (
    SyntheticControl, SyntheticDID, TwoStageIndividualSC,
    backtest, plot_backtest, save_figure,
)

# Each call constructs a new estimator with its complete fitting settings.
candidates = {
    "Outcome SCM": SyntheticControl,
    "SDID": SyntheticDID,
    "2SISC k=5": partial(TwoStageIndividualSC, n_neighbors=5),
    "2SISC k=10": partial(TwoStageIndividualSC, n_neighbors=10),
}
validation = backtest(panel, candidates, initial_window=10, horizon=3)
validation.summary       # pooled RMSE, MAE, bias and prediction/fold counts
validation.scores        # one row per model and fold, including date boundaries
validation.predictions   # every held-out observation, prediction and error
fig, axes = plot_backtest(validation, units="GDP per capita")
save_figure(fig, "outputs/custom/validation")
```

Choose windows suitable for your panel; the example needs at least 13
pretreatment observations. Origins start at `initial_window` and advance by
`step` (default: `horizon`). Only complete blocks of `horizon` observations
strictly before the actual intervention are scored. Training expands at each
origin. Folds use observation positions, so for irregular time indices a
three-observation horizon need not represent three calendar years.

The fitting boundary is deliberate:

1. All actual post-treatment rows are removed from the fold panel.
2. Each treated unit's held-out outcomes are replaced by its last training
   outcome before any estimator receives the panel.
3. Validation-period donor outcomes remain available. Synthetic controls
   estimate contemporaneous counterfactuals from observed donors; this is
   retrospective validation, **not forecasting future donor outcomes**.
4. True held-out treated outcomes are used only outside fitting to calculate
   errors. A fresh estimator is constructed for every fold.

The masking rule protects against a fit accidentally reading the supplied
held-out treated outcomes. It cannot prevent a factory from closing over
external data. Custom covariates, preprocessing and `Dataprep` objects must be
constructed using the current fold's eligible data. A `Dataprep` for the
original full panel fails the adapter's alignment checks; do not reuse it.

For additional time-varying 2SISC covariates, a small estimator wrapper can
select their training rows and eligible columns inside `fit`. Here each
`covariate_frames[name]` is a time-by-unit DataFrame with the original labels:

```python
class TwoStageWithCovariates:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, fold):
        predictors = {
            name: frame.loc[fold.times[:fold.n_pre], list(fold.units)].to_numpy()
            for name, frame in covariate_frames.items()
        }
        return TwoStageIndividualSC(n_neighbors=self.n_neighbors).fit(
            fold, predictors=predictors,
        )

covariate_candidates = {
    "2SISC k=5": partial(TwoStageWithCovariates, 5),
    "2SISC k=10": partial(TwoStageWithCovariates, 10),
}
```

The same wrapper works for donor exclusions because it selects the current
panel's unit labels. Any estimated transformations must also be fitted within
the training window; slicing a transformation already learned from the full
series would not remove its leakage.

SCUL's original admissibility screen examines the full series supplied to it.
During validation this means the masked series, not the held-out truth.
Consequently this evaluation protocol deliberately differs from running raw
SCUL on an unmasked validation panel. Its complete regression/CV procedure
and the settings supplied by the factory are retained.

All candidates receive the same folds. Overall RMSE is calculated by pooling
all held-out prediction instances; it is not the average of fold RMSEs.
If `step < horizon`, overlapping periods count once per prediction instance.
For simultaneous treatment, the target is the equal-weight treated-unit mean,
consistent with `Estimate`. Candidate ties follow mapping insertion order.
The fold metadata retains each estimator's fitting diagnostics. Failures stop
the comparison with the candidate and fold identified; no model is dropped.

## Select a specification without using actual post-treatment outcomes

```python
from pysynth import select_estimator

selection = select_estimator(panel, candidates, initial_window=10, horizon=3)
selection.selected_name
selection.validation.summary
selection.estimate.att   # fresh winning specification fitted to the full panel
```

Selection minimizes pooled pretreatment validation RMSE, then refits the
winner using all original pretreatment observations. The final fit receives
the full panel, following that estimator's documented procedure. Factories
should use fixed random seeds for comparable stochastic fits. No optimizer
starts, donor resamples or convergence settings are reduced by the workflow.

This makes 2SISC donor-count selection inspectable: compare a prespecified
grid of `n_neighbors`, retain the fold errors, and refit the selected count.
The original two-stage method and the MSCMT variant remain separate choices;
use `partial(TwoStageIndividualSC, backend="mscmt", n_bootstrap=50, ... )`
explicitly for the latter. The temporal selection procedure is a software
extension of the reference workflow, not a claim that nearest-neighbor
screening or cross-validation is newly invented.

The winning validation score is optimistic after selecting among candidates.
It is not an unbiased performance estimate, proof of causal identification,
or a basis for post-selection confidence intervals. Keep the full candidate
table; prespecify the grid and folds, and reserve additional evaluation data
when making predictive performance claims. Short pretreatment series may
support too few folds to choose reliably.

## Refit after donor exclusions

```python
from pysynth import donor_sensitivity, plot_donor_sensitivity

# By default, remove each eligible donor in turn, including zero-weight donors.
sensitivity = donor_sensitivity(panel, candidates)
sensitivity.table      # each model/scenario ATT, change, fit and sign reversal
sensitivity.summary    # scenario range and largest absolute ATT change

fig, axes = plot_donor_sensitivity(sensitivity, units="GDP per capita")
save_figure(fig, "outputs/custom/donor-sensitivity")

# Study-defined groups can represent shared exposure or questionable eligibility.
groups = {"exclude adjacent regions": ["Region A", "Region B"]}
group_check = donor_sensitivity(panel, candidates, exclusions=groups)
```

Group labels must name actual controls. A scenario cannot remove all controls
or remove a treated unit. The baseline and every exclusion are complete,
fresh fits. Removing a zero-weight donor can still change tuning, donor
screening, resampling or the optimizer's solution. Negative weights are not
discarded. For 2SISC, the nearest-neighbor screen runs again on the remaining
eligible pool. This is more informative than deleting a coefficient and
renormalizing the previous weights.

`baseline[model]` and `estimates[scenario][model]` preserve full paths,
weights and diagnostics. `sign_changed` means a strict positive-to-negative
or negative-to-positive reversal; zero-to-nonzero changes are not reversals.
Summary minima and maxima refer to exclusion scenarios; the baseline is
reported separately. Scenario ranges are descriptive, not confidence
intervals or tests. A chosen exclusion grid is not a probability distribution.
Repeated stochastic fits should use fixed seeds, although changing the donor
pool can still change the realized random samples.

## Cost and reproducibility

With `M` candidates, `F` folds and `S` exclusions, validation performs `M × F`
full fits, selection performs one further full fit, and donor sensitivity
performs `M × (S + 1)` full fits. These costs are explicit. The functions do
not cache fitted estimators, subsample donors, truncate optimizations, or
suppress estimator warnings. Tables can be exported with `to_csv`; generated
outputs remain local-only under the repository's existing ignore rules.

All four empirical notebooks demonstrate temporal validation and complete
single-donor exclusions. They are new analyses of the canonical datasets,
separate from their published-reference replications.

## A known-effect simulation

The [donor-spillover notebook](https://github.com/crahal/pysynthcon/blob/main/simulated_examples/donor_spillovers.ipynb)
uses a regional policy with a known effect on the treated region. In each
replication it generates two versions of the same panel. Both have identical
pretreatment histories. In one version, four nearby donors also experience
a post-policy increase of three outcome units.

For SCM fitted on pretreatment outcomes, that increase raises the estimated
counterfactual by three times the total fitted weight on nearby donors.
The estimated treatment effect falls by the same amount. Validation sees
identical data in the two versions, so it cannot detect this change. A named
group exclusion refits the estimator on distant donors. The known treatment
effect lets the notebook measure both the predictive cost of that exclusion
before treatment and its effect on bias after treatment.

Across 200 seeded replications, the full donor pool gives SCM a lower
average held-out RMSE (1.20 points) than the distant-only pool (3.26).
With spillovers, the distant-only pool estimates the treatment effect more
accurately. The root mean squared ATT error is 2.84 points with all donors and 1.95
with distant donors. The full-pool estimate shifts by −2.78 points on
average, with a Monte Carlo standard error of 0.013. That standard error
measures simulation precision across replications.

Run `python scripts/execute_notebooks.py --examples donor_spillovers` to
regenerate the figures and Monte Carlo tables under
`outputs/simulated_examples/donor_spillovers/`.
