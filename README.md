<h1 align="center">
  <a href="empirical_examples/california.ipynb">
    <picture>
      <source media="(max-width: 600px)" srcset="docs/assets/pysynthcon-header-compact.svg">
      <img src="docs/assets/pysynthcon-header.svg" width="100%" alt="PySynthCon — Synthetic control in Python. After California’s tobacco tax, cigarette sales averaged 15.6 fewer packs per person per year than its synthetic counterfactual, 1989–2000.">
    </picture>
  </a>
</h1>

[![Tests and documentation](https://github.com/crahal/pysynthcon/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/crahal/pysynthcon/actions/workflows/ci.yml)
[![Python 3.10 and newer](https://img.shields.io/badge/Python-3.10%2B-3776AB)](pyproject.toml)
[![License GPL-3.0-only](https://img.shields.io/badge/License-GPL--3.0--only-blue.svg)](LICENSE)
[![Purpose Research](https://img.shields.io/badge/Purpose-Research-yellow.svg)](docs/replication.md)

**[Quick start](#quick-start) · [Five worked examples](#five-worked-examples) · [Estimators](#estimators) · [Documentation](#documentation)**

PySynthCon lets you fit several synthetic controls to the same panel and see how the estimated effect changes with the method or donor pool. Compare held-out pretreatment predictions, refit after excluding donors, and put published benchmarks beside the estimates that reproduce them. The library builds on [pysyncon](https://github.com/sdfordham/pysyncon) and [SyntheticControlMethods](https://pypi.org/project/SyntheticControlMethods/).

## Quick start

Install from source with Python 3.10 or newer. The distribution is `pysynth-control` and the Python import is `pysynth`.

```bash
git clone https://github.com/crahal/pysynthcon.git
cd pysynthcon
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[plot]"
```

California introduced its tobacco-control program in 1989. Use the same states and years to fit outcome-matching synthetic control (SCM) and synthetic difference-in-differences (SDID).

```python
from pysynth import SyntheticControl, SyntheticDID, compare, plot_effects, save_figure
from pysynth.datasets import DATASETS, load_dataset

panel = DATASETS["california"].panel(load_dataset("california"))
fits = {"SCM": SyntheticControl().fit(panel), "SDID": SyntheticDID().fit(panel)}

print(compare(fits)[["att", "pre_rmse"]])
fig, axes = plot_effects(fits, ylabel="Effect on sales (packs per capita)", xlabel="Year")
save_figure(fig, "outputs/custom/california-effects")
```

The SDID estimate is **−15.60 packs per capita per year**, averaged over 1989–2000. Effects are observed outcomes minus estimated untreated outcomes. The reference fit reports an iteration-limit warning for one set of weights and reproduces the original R estimate. [California notebook](empirical_examples/california.ipynb) · [SDID convergence and inference](docs/inference.md#handling-iteration-limit-warnings)

For your own data, `Panel.from_frame` takes unit, time and outcome columns, an explicit treatment date and an optional donor list. The treatment date is the first affected observation. Panels are balanced and use a common onset within each panel. ISC and 2SISC also accept case-specific panels and aggregate in event time. [Prepare a panel](docs/data.md)

## Five worked examples

| Open the notebook | What to look for |
| --- | --- |
| [California · tobacco control](empirical_examples/california.ipynb) | Reproduce the published SDID effect of −15.6 packs per capita and its placebo standard error of 8.4. |
| [Basque Country · conflict](empirical_examples/basque.ipynb) | Recover the published SCM donor weights for Catalonia (0.8508) and Madrid (0.1492), and compare the GDP path with the original R fit. |
| [West Germany · reunification](empirical_examples/germany.ipynb) | Rebuild the historical predictor specification and compare the GDP gap with the paper’s roughly US$1,600 annual loss per person (1990–2003, PPP-adjusted). |
| [Kansas · tax cuts](empirical_examples/kansas.ipynb) | Match the authors’ ridge-augmented SCM vignette at −0.0401 log points and check all 16 quarterly effects. |
| [Simulation · donor spillovers](simulated_examples/donor_spillovers.ipynb) | Across 200 economies with a known +6-point effect, spillovers shift full-pool SCM by −2.78 points on average while its pretreatment fit stays identical. |

The four empirical notebooks check the historical specification, then compare twelve estimator interfaces on a common panel. Each downloads hash-verified data into memory. The simulation generates its own data and runs entirely in Python. Figures are embedded in the notebooks. Executing them also writes PDF, SVG, PNG and numerical tables under `outputs/empirical_examples/<study>/` or `outputs/simulated_examples/donor_spillovers/`. [Reference targets and tolerances](docs/replication.md) · [Figure exports](docs/plotting.md)

```bash
python -m pip install -e ".[notebooks]"
python scripts/execute_notebooks.py --examples donor_spillovers
```

For all five notebooks, use Python 3.11 or 3.12, install the [pinned replication requirements](requirements-replication.txt) and provision the separately supplied local R reference bundle. SCUL and vanilla ISC use the original R backends. Reproducing the California paper’s exact placebo sampling plan also uses the bundle. Once provisioned, `python scripts/execute_notebooks.py` runs all five. [Local R setup](docs/reference_backends.md)

## Estimators

| Family | Python interface | Specification |
| --- | --- | --- |
| Outcome SCM | `SyntheticControl()` | Convex donor weights matching pretreatment outcomes, with optional ridge regularization. |
| SCUL | `SCUL()` | Lasso with rolling-origin tuning, signed weights and an intercept. |
| Individual synthetic control | `IndividualSC()` | Vanilla ISC fits each individual against the full eligible pool using MSCMT and 50 donor resamples by default. |
| Two-stage individual synthetic control | `TwoStageIndividualSC(n_neighbors=10)` | 2SISC screens nearest neighbors on pretreatment features, then fits the authors’ mean-predictor SCM in that group. |
| SDID and DID | `SyntheticDID()`, `DifferenceInDifferences()` | Native Python fits for simultaneous treatment. SDID supports bootstrap, jackknife and placebo inference. |
| pysyncon | `PySyncon("synth")` | Predictor-based SCM. Select `"augsynth"`, `"pensynth"` or `"robust"` for the augmented, penalized or robust fit. |
| SyntheticControlMethods | `SCM("synth")`, `SCM("diffsynth")` | Classical and differenced SCM with all pre-outcome lags, ten starts and checked optimizer recovery. |

Install `python -m pip install -e ".[backends]"` for the two adapter families and the default 2SISC fit. The two-stage specification follows Petrillo et al. (2025). [Method definitions and attribution](docs/methods.md) · [Backend details](docs/backends.md)

## Compare specifications

`select_estimator` scores each candidate on the same withheld pretreatment periods and refits the selected candidate on the full panel. `donor_sensitivity` refits after each donor exclusion. Pass named groups to examine a possible spillover region, as in the simulation.

```python
from pysynth import donor_sensitivity, select_estimator

candidates = {"SCM": SyntheticControl, "SDID": SyntheticDID}
selection = select_estimator(panel, candidates, initial_window=12, horizon=4)
print(selection.validation.summary)
print(donor_sensitivity(panel, candidates).summary)
```

`plot_comparison` aligns ATT estimates with pretreatment RMSE, and `plot_effects` preserves a common outcome scale across methods. Both return editable Matplotlib figures. [Evaluation guide](docs/evaluation.md) · [Plotting guide](docs/plotting.md)

## Documentation

- [Getting started](docs/getting-started.md) · [Data preparation](docs/data.md) · [API reference](docs/api.rst)
- [Research workflows](docs/differentiation.md) · [Inference](docs/inference.md) · [Estimator and replication audit](docs/audit.md)
- [Contributing and running tests](CONTRIBUTING.md) · [Changelog](CHANGELOG.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)

When publishing, cite PySynthCon together with the method papers and upstream packages used in your analysis. [Software citation](CITATION.cff) · [Method references](docs/references.md)
