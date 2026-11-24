# PySynth

**Synthetic control methods, comparable results, reproducible academic examples.**

PySynth provides a common Python interface to classical synthetic control,
SCUL, individual synthetic control, and synthetic difference-in-differences.
It builds on [pysyncon](https://github.com/sdfordham/pysyncon) and
[SyntheticControlMethods](https://pypi.org/project/SyntheticControlMethods/),
with the original R fitting procedures for SCUL and ISC, and a validated
Python implementation of the original SDID algorithms.

Beyond fitting individual estimators, PySynth provides a common workflow for
pretreatment model validation, donor and donor-group sensitivity, and direct
comparisons with academic reference results. These tools retain each model's
full fitting settings and make the specification choices visible. See
[what PySynth adds](docs/differentiation.md) for its relationship to pysyncon,
SyntheticControlMethods and other causal-analysis packages.

This is a research-oriented **0.1.0** release. It supports balanced panels,
explicit donor pools, one treatment onset per panel, and case-specific panels
for individual synthetic control. Effects are always observed minus estimated
untreated outcomes. There is no automatic causal interpretation of a good fit.

## Install

Python 3.10 or newer. Install this repository; the distribution name is
`pysynth-control` and the import name is `pysynth`. Do not install the unrelated
`pysynth` project from PyPI.

```bash
git clone https://github.com/crahal/pysynthcon.git
cd pysynthcon
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e '.[backends,notebooks]'
```

For the tested academic replication environment, use Python 3.11 or 3.12:

```bash
python -m pip install -r requirements-replication.txt
python -m pip install -e '.[docs,dev]'
```

SCUL and ISC require a **separately supplied local reference bundle**, R, and
the pinned reference packages. R source, setup utilities, and generated exports
are excluded from this repository and distributions. A fresh clone alone cannot
run these two reference backends or re-execute all empirical notebooks. Follow
the [local reference setup guide](docs/reference_backends.md) to provision them.
SCM, DID and SDID estimation and inference remain available in Python; the
California paper's exact R-generated sampling plan also needs the local bundle.
The `backends` extra pins pysyncon replication fixes.

## A complete example

```python
from pysynth import (
    SyntheticControl, TwoStageIndividualSC, SyntheticDID, compare,
)
from pysynth.datasets import DATASETS, load_dataset

data = load_dataset("california")  # requests download; pinned URL + SHA256
panel = DATASETS["california"].panel(data)

models = {
    "SCM": SyntheticControl(),
    "2SISC": TwoStageIndividualSC(n_neighbors=10),
    "SDID": SyntheticDID(),
}
fits = {name: model.fit(panel) for name, model in models.items()}
print(compare(fits))
print(fits["SDID"].att)  # approximately -15.60382787 packs per capita
fits["SDID"].plot()
```

With the local R bundle installed, include the original SCUL and ISC fits in
the same comparison.

```python
from pysynth import SCUL, IndividualSC

fits["SCUL"] = SCUL(initial_window=5, horizon=12).fit(panel)
fits["ISC"] = IndividualSC().fit(panel)
print(compare(fits))
```

SDID's reference stopping rule reaches its iteration cap for one California
weight fit; the warning and solver diagnostics remain visible. The resulting
estimate and weights match the pinned R reference implementation.

With your own long-form data:

```python
from pysynth import Panel, SyntheticDID

panel = Panel.from_frame(
    df, unit="country", time="year", outcome="gdp",
    treated_unit="West Germany", treatment_time=1990,
    controls=["USA", "UK", "Austria", "France"],
)
result = SyntheticDID().fit(panel)
result.att                   # average post-treatment effect
result.weights               # labelled donor weights
result.time_weights          # SDID pre-period weights
result.to_frame()            # observed, counterfactual, effect, post
```

`treatment_time` is the **first affected observed period**. Duplicate rows,
missing outcomes, unbalanced selected panels, and invalid timing raise errors.
For simultaneous treatment, pass `treated_units=[...]` to `Panel.from_frame`.
For individual-specific timing, build one eligible panel per case and use
`IndividualSC.fit_many` or `TwoStageIndividualSC.fit_many` ([guide](docs/data.md)).

## Available methods

| Interface | Method | Notes |
|---|---|---|
| `SyntheticControl()` | Convex outcome-matching SCM | Equal importance for every pre-period; optional ridge penalty |
| `SCUL()` | Synthetic Control Using Lasso | Rolling-origin selection; signed weights and intercept |
| `IndividualSC()` | Individual Synthetic Control (ISC) | Full eligible donor pool; donor resampling and event-time aggregation |
| `TwoStageIndividualSC(n_neighbors=10)` | Two-Stage Individual Synthetic Control (2SISC) | Nearest-neighbor screening and the authors' mean-predictor SCM fit |
| `SyntheticDID()` | Synthetic difference-in-differences | Native R-reference algorithm; simultaneous treated blocks |
| `DifferenceInDifferences()` | DID comparison | Uniform unit and time weights |
| `PySyncon("synth")` | Classical predictor-based SCM | Optional `pysyncon.Dataprep` for published specifications |
| `PySyncon("augsynth")` | Augmented SCM | Ridge augmentation and upstream cross-validation |
| `PySyncon("pensynth")` | Penalized SCM | Explicit penalty in `options` |
| `PySyncon("robust")` | Robust SCM | Explicit rank/threshold and ridge in `options` |
| `SCM("synth")` | SyntheticControlMethods SCM | All pre-lags; ten starts; checked solver recovery |
| `SCM("diffsynth")` | Differenced SCM | Local pandas compatibility fix and level correction |

See [method definitions](docs/methods.md) and [backend details](docs/backends.md).
Vanilla ISC uses every eligible donor in each individual's panel. 2SISC first
selects up to `n_neighbors` donors using pretreatment Euclidean distance, then
fits the synthetic-control weights within that group. ISC uses MSCMT and 50
donor resamples; 2SISC follows the authors' pysyncon mean-predictor specification
without donor resampling. Both support event-time aggregation. The two-stage
approach is credited to
[Petrillo et al. (2025)](https://eprints.whiterose.ac.uk/id/eprint/224118/1/Binder2025004.pdf);
the method guide documents the donor-count convention and remaining study
replication limits. The earlier 2SISC variant remains available through
`backend="mscmt", n_bootstrap=50`. To migrate the initial API,
replace `IndividualSC(n_neighbors=k)` with `TwoStageIndividualSC(n_neighbors=k)`.
The [individual-method guide](docs/methods.md#individual-synthetic-control-isc)
describes the reference algorithms and the survey-specific choices needed
for a study replication.

## Select and stress-test a specification

Compare candidates on common held-out pretreatment periods, then refit the
selected candidate using the complete panel. Candidate factories create fresh
models for each fit. This example runs entirely in Python:

```python
from pysynth import SyntheticControl, SyntheticDID, select_estimator, donor_sensitivity

estimators = {"SCM": SyntheticControl, "SDID": SyntheticDID}
selection = select_estimator(panel, estimators, initial_window=12, horizon=4)
print(selection.validation.summary)
print(selection.selected_name, selection.estimate.att)

sensitivity = donor_sensitivity(panel, estimators)  # full donor-deletion refits
print(sensitivity.summary)
```

Actual post-treatment observations are excluded from selection, and held-out
treated outcomes are masked before fitting. Named donor-group exclusions are
also supported. Lower prediction error and stable donor sensitivity are useful
diagnostics; they do not establish causal validity. See the
[comparison workflows](docs/differentiation.md) for 2SISC neighbor tuning,
interpretation and related software.

## Empirical notebooks

Each notebook downloads data with `requests`, checks the source hash, runs a
reference specification with numerical assertions, and compares **all twelve**
applicable interfaces on a common outcome panel.

| Notebook | Academic setting | Reference check |
|---|---|---|
| [California](empirical_examples/california.ipynb) | Tobacco control, ADH (2010); SDID (2021) | R SDID ATT −15.60382787 |
| [Basque Country](empirical_examples/basque.ipynb) | Abadie & Gardeazabal (2003) | Catalonia/Madrid weights and 1975–1997 reference ATT |
| [Germany](empirical_examples/germany.ipynb) | ADH (2015) | Historical two-stage specification and reference ATT |
| [Kansas](empirical_examples/kansas.ipynb) | Ben-Michael et al. (2021) | Authors' Ridge ASCM vignette ATT −0.0401 |

```bash
python scripts/execute_notebooks.py --examples basque california germany kansas
```

These notebooks distinguish published specifications, upstream numerical
references, and new applications. They do **not** claim to reproduce SCUL's
original scanner-data study or the ISC/costofcare survey study. The [replication
guide](docs/replication.md) states exact benchmarks, tolerances and limitations.

Full fitting can still be quick on these small aggregate panels. With only
the outcome as one predictor group, MSCMT itself has no nontrivial predictor-
weight search; fitting many individuals with additional covariates is more
costly. The notebooks report elapsed times and full fitting settings so the
computational work is inspectable.

## Simulation notebook

[Donor spillovers](simulated_examples/donor_spillovers.ipynb) asks whether a
good pretreatment fit protects an estimate when nearby comparison regions are
also affected by a policy. The simulation generates paired regional panels
with a known treatment effect. Each pair shares the same underlying outcomes.
Only the nearby donors' post-treatment outcomes change. The notebook compares
pretreatment validation scores with actual estimation errors, then uses named
donor-group exclusions to refit SCM and DID. It runs 200 replications in Python
and writes both the draw-level estimates (`paired-estimates.csv`) and the
averages with Monte Carlo standard errors (`monte-carlo-summary.csv`).

```bash
python -m pip install -e '.[notebooks]'
python scripts/execute_notebooks.py --examples donor_spillovers
```

Run `python scripts/execute_notebooks.py` for all five notebooks. Empirical
inputs download into memory on each run, and the simulation generates its
inputs from a recorded random seed. Local figures, result tables, and numerical
diagnostics go under `outputs/empirical_examples/<study>/` and
`outputs/simulated_examples/donor_spillovers/`. The notebooks embed their
figures for reading on GitHub.

## Publication figures

Compare fitted estimators with shared-scale effect panels and an aligned
ATT/pretreatment-fit summary. The library supplies readable typography,
explicit treatment timing and outcome units; all figures remain editable
Matplotlib objects.

```python
from pysynth import plot_comparison, plot_effects, save_figure

fig, axes = plot_comparison(fits, units="packs per capita")
save_figure(fig, "outputs/custom/california-comparison")

fig, axes = plot_effects(fits, ylabel="Effect on cigarette sales (packs per capita)")
save_figure(fig, "outputs/custom/california-effects")
```

See the embedded figures in the [California notebook](empirical_examples/california.ipynb).
Each empirical notebook displays three reference/comparison figures with **source /
reproduction / difference** annotations directly beside the matching curves,
panels and ATT rows. Separate historical fits receive their own reference row
or panel. Two further figures label pretreatment validation and donor-exclusion
sensitivity results directly. Expandable notes retain full specifications
and audit tables. Local export paths follow each figure. Use
`reference_style="strip"` for the earlier source-strip layout.
Rerunning a notebook
creates local PDF, SVG and PNG exports, CSVs of source comparisons and estimates,
and fitting diagnostics. Generated files stay outside version control.
Each empirical notebook also runs the original SDID placebo variance procedure with 200
resamples, then repeats that exact plan with a larger iteration budget and
requires every resampled weight fit to meet the original stopping rule.
Both standard errors and convergence counts are shown; exact published-reference
assertions retain the original cap. See [inference convergence](docs/inference.md#handling-iteration-limit-warnings)
for the explicit budget override and strict check.
The [plotting guide](docs/plotting.md) explains local regeneration.
Dots are point estimates, without confidence intervals; shaded time regions
identify the post-period.

## Documentation and development

- [User guide and data preparation](docs/data.md)
- [What PySynth adds and related software](docs/differentiation.md)
- [Pretreatment validation, selection and donor sensitivity](docs/evaluation.md)
- [Estimator equations and assumptions](docs/methods.md)
- [Placebo diagnostics and inference](docs/inference.md)
- [Publication plotting and export](docs/plotting.md)
- [Backend compatibility](docs/backends.md)
- [Replication provenance](docs/replication.md)
- [Estimator audit, corrections and remaining limits](docs/audit.md)
- [API reference source](docs/api.rst)
- [Contributor guide](CONTRIBUTING.md), [changelog](CHANGELOG.md), [citations](docs/references.md)

```bash
python -m pytest                         # offline tests
python -m pytest -m network              # academic reference checks; internet required
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m build
```

Open `docs/_build/html/index.html` for the full navigable documentation.
Licensed under [GPL-3.0](LICENSE); see [third-party notices](THIRD_PARTY_NOTICES.md).
