# PySynth

**Synthetic control methods, comparable results, reproducible academic examples.**

PySynth provides a common Python interface to classical synthetic control,
SCUL, individual synthetic control, and synthetic difference-in-differences.
It builds on [pysyncon](https://github.com/sdfordham/pysyncon) and
[SyntheticControlMethods](https://pypi.org/project/SyntheticControlMethods/),
with native implementations of the three additional approaches requested here.

This is a research-oriented **0.1.0** release. It supports balanced panels,
explicit donor pools, one treatment onset per panel, and case-specific panels
for individual synthetic control. Effects are always observed minus estimated
untreated outcomes. There is no automatic causal interpretation of a good fit.

## Install

Python 3.10 or newer. Install this repository; the distribution name is
`pysynth-control` and the import name is `pysynth`. Do not install the unrelated
`pysynth` project from PyPI.

```bash
git clone https://github.com/crahal/PySynth.git
cd PySynth
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e '.[backends,notebooks]'
```

For the tested academic replication environment, use Python 3.11 or 3.12:

```bash
python -m pip install -r requirements-replication.txt
python -m pip install -e '.[docs,dev]'
```

Core estimators need neither R nor the optional upstream packages. The
`backends` extra pins a pysyncon revision containing replication fixes.

## A complete example

```python
from pysynth import SyntheticControl, SCUL, IndividualSC, SyntheticDID, compare
from pysynth.datasets import DATASETS, load_dataset

data = load_dataset("california")  # requests download; pinned URL + SHA256
panel = DATASETS["california"].panel(data)

models = {
    "SCM": SyntheticControl(),
    "SCUL": SCUL(initial_window=10, horizon=2),
    "Individual SC": IndividualSC(n_neighbors=10),
    "SDID": SyntheticDID(),
}
fits = {name: model.fit(panel) for name, model in models.items()}
print(compare(fits))
print(fits["SDID"].att)  # approximately -15.60382787 packs per capita
fits["SDID"].plot()
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
`IndividualSC.fit_many` ([guide](docs/data.md)).

## Available methods

| Interface | Method | Notes |
|---|---|---|
| `SyntheticControl()` | Convex outcome-matching SCM | Equal importance for every pre-period; optional ridge penalty |
| `SCUL()` | Synthetic Control Using Lasso | Rolling-origin selection; signed weights and intercept |
| `IndividualSC()` | Costofcare-style individual SC | Optional nearest donors, donor resampling, event-time aggregation |
| `SyntheticDID()` | Synthetic difference-in-differences | Native R-reference algorithm; simultaneous treated blocks |
| `DifferenceInDifferences()` | DID comparison | Uniform unit and time weights |
| `PySyncon("synth")` | Classical predictor-based SCM | Optional `pysyncon.Dataprep` for published specifications |
| `PySyncon("augsynth")` | Augmented SCM | Ridge augmentation and upstream cross-validation |
| `PySyncon("pensynth")` | Penalized SCM | Explicit penalty in `options` |
| `PySyncon("robust")` | Robust SCM | Explicit rank/threshold and ridge in `options` |
| `SCM("synth")` | SyntheticControlMethods SCM | Corrected predictor objective; deterministic single start |
| `SCM("diffsynth")` | Differenced SCM | Local pandas compatibility fix and level correction |

See [method definitions](docs/methods.md) and [backend details](docs/backends.md).
The costofcare link represents the individual synthetic control design of
Vagni and Breen (2021), not a distinct lasso estimator.

## Empirical notebooks

Each notebook downloads data with `requests`, checks the source hash, runs a
reference specification with numerical assertions, and compares **all eleven**
applicable interfaces on a common outcome panel.

| Notebook | Academic setting | Reference check |
|---|---|---|
| [California](empirical_examples/california.ipynb) | Tobacco control, ADH (2010); SDID (2021) | R SDID ATT −15.60382787 |
| [Basque Country](empirical_examples/basque.ipynb) | Abadie & Gardeazabal (2003) | Catalonia/Madrid weights and 1975–1997 reference ATT |
| [Germany](empirical_examples/germany.ipynb) | ADH (2015) | Historical two-stage specification and reference ATT |
| [Kansas](empirical_examples/kansas.ipynb) | Ben-Michael et al. (2021) | Authors' Ridge ASCM vignette ATT −0.0401 |

```bash
python scripts/execute_notebooks.py
```

These notebooks distinguish published specifications, upstream numerical
references, and new applications. They do **not** claim to reproduce SCUL's
original scanner-data study or the ISC/costofcare survey study. The [replication
guide](docs/replication.md) states exact benchmarks, tolerances and limitations.

## Documentation and development

- [User guide and data preparation](docs/data.md)
- [Estimator equations and assumptions](docs/methods.md)
- [Placebo diagnostics and inference](docs/inference.md)
- [Backend compatibility](docs/backends.md)
- [Replication provenance](docs/replication.md)
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
