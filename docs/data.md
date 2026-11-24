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
For a single string identifier use `treated_unit="CA"` or
`treated_units=["CA"]`. Both `treated_units` and `controls` require a
collection of identifiers. A bare string can be confused with a sequence
of one-character unit names.
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

For one case, use `fit(panel, predictors={"age": age, "education": education})`.
Each array has `panel.n_pre` rows and columns in `panel.units` order. MSCMT
uses each series as a predictor group, alongside the pretreatment outcome,
and learns predictor importance using its original optimization. Default 2SISC
screens on the stacked, unscaled predictor series, then matches their temporal
means through the authors' pysyncon specification. Scaling disparate variables
is an explicit departure from the raw-distance reference. Only pretreatment
measurements should be supplied.

For several cases, pass a mapping of case identifiers to these predictor
dictionaries through `fit_many(panels, predictors=case_predictors)`. Its keys
must match the panel mapping. The earlier generic `features` matrix remains
available only with explicitly selected `backend="fixed_metric"`; that
adaptation is not the default original ISC procedure. Install the
[R reference dependencies](reference_backends.md) for default ISC; default
2SISC requires the Python `backends` extra instead.

## Academic downloads

`load_dataset("california" | "basque" | "germany" | "kansas")` uses requests,
an immutable source commit, explicit HTTP timeout, status checking, and
SHA256 verification. By default the download stays in memory. All empirical
notebooks use in-memory downloads, and the simulation generates its panel in
memory on each run. Library callers can explicitly request a disk cache with
`cache_dir`. Corrupt cached bytes raise an error. `DATASETS[name]` exposes the URL, hash, citation,
outcome labels and treatment convention. Historical covariate missingness
is retained in the DataFrame; the chosen outcome panel must be complete.

The Basque dataset also contains Spain’s national aggregate. `Dataset.panel`
excludes it: the original paper’s donor pool consists of the other 16 regions,
and the aggregate contains the treated Basque Country. The source download is
retained unchanged.
