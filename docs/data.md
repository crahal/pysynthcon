# Data and treatment design

## Balanced panels

`Panel.from_frame` requires one numeric, finite outcome per selected unit and
time. It sorts times, keeps donors in the requested order, and places treated
units last. It rejects duplicate unit-time rows, absent requested units,
nonfinite outcomes, and holes in the selected rectangular panel. At least two
pre-periods, one post-period, and one donor are required. Times may be numeric
or sorted date labels; the treatment label must be an observed time.

```python
from pysynth import Panel

panel = Panel.from_frame(
    df, unit="id", time="date", outcome="y",
    treated_unit="treated", treatment_time=2020,
    controls=["donor_a", "donor_b"],
)
```

Omitting `controls` uses all other observed units. Choose donors based on the
research design: avoid units affected by the intervention, spillovers, or
anticipation. PySynth cannot establish donor eligibility from outcome values.
Missingness is not silently imputed and calendar gaps shared by every unit
are not automatically filled. A period is an observed row: use a regular
frequency when interpreting rolling horizons or first differences.

For matrix data, `Panel(y, n_control, n_pre, times=..., units=...)` uses **time
in rows**, controls in the first columns, and treated units in the remaining
columns. This is transposed relative to R synthdid's outcome matrix. The
matrix is copied and made read-only to prevent accidental input mutation.

## Simultaneous treatment

Pass `treated_units=[...]` to `from_frame` for a simultaneous treated block.
SDID and DID implement this design directly. Outcome SCM fits the treated
group's equal-weight average trajectory. SCUL and upstream adapters require
one treated unit. Staggered treatment is not automatically interpreted as a
simultaneous block.

## Individual synthetic controls

Prepare a separate panel for each individual, with donors eligible at that
individual's treatment date. Use the same time frequency in every panel.

```python
from pysynth import IndividualSC, TwoStageIndividualSC, Panel

panels = {
    case: Panel.from_frame(
        eligible_frames[case], unit="person", time="year", outcome="earnings",
        treated_unit=case, treatment_time=onset[case],
        controls=eligible_donors[case],
    )
    for case in onset
}
vanilla = IndividualSC().fit_many(panels)
result = TwoStageIndividualSC(n_neighbors=50).fit_many(panels)
print(vanilla.att)      # ISC: full eligible donor pool for each individual
print(result.att)       # equal-person mean of individual post-period means
print(result.summary)   # event-time effect, contributing n, weight_sum
```

Calendar eligibility, age matching, survey weights and censoring are explicit
caller decisions. `fit_many(..., person_weights=...)` supports strictly
positive person weights. An event-time mean uses cases observed at that
event time, so the composition can change across the curve. Counts and
weight totals are always returned.

For one case, either estimator's `fit(panel, features=matrix)` can use stacked
pretreatment outcomes and covariates. Columns must match `panel.units`.
Those features replace the pre-outcome fitting matrix; in 2SISC they also
determine nearest donors. Scale disparate variables deliberately. For example,
`TwoStageIndividualSC(n_neighbors=50).fit(panel, features=matrix)` screens
and fits using those features, then predicts using donor outcome paths.
The library cannot infer when externally supplied features were measured.
`fit_many` uses each panel's pretreatment outcomes; fit individual panels
separately when supplying custom feature matrices.

## Academic downloads

`load_dataset("california" | "basque" | "germany" | "kansas")` uses requests,
an immutable source commit, explicit HTTP timeout, status checking, and
SHA256 verification. An optional `cache_dir` stores verified bytes; corrupt
cached bytes raise an error. `DATASETS[name]` exposes the URL, hash, citation,
outcome labels and treatment convention. Historical covariate missingness
is retained in the DataFrame; the chosen outcome panel must be complete.
