# Contributing to PySynth

Thank you for helping make synthetic control research easier to reproduce.
Report bugs or propose changes through the repository's GitHub issues and pull
requests. Include a minimal reproducible panel, versions, the complete error,
and the statistical specification. Do not upload confidential research data.

## Set up

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-replication.txt
python -m pip install -e '.[dev,docs]'
```

Use the `src/` layout and the common `Panel -> Estimate` interface. Native
methods should remain usable without optional backends, plotting or R.
Keep input arrays immutable, preserve labels, return original outcome units,
and define the ATT and all preprocessing windows explicitly.

## Statistical changes

For an estimator change, cite a primary paper and an immutable reference
implementation where available. State the objective, constraints, penalty
normalization, intercept convention, tuning procedure and assumptions. Do not
rename a generic regression as an established method without matching its
design. Never use treated post outcomes for fitting weights or tuning.
Distinguish algorithms that legitimately use donor post outcomes, such as
SDID time weighting and robust-SCM denoising.

Add meaningful tests: an analytic case or independently generated reference,
label/order checks, convergence failures, and leakage checks. For empirical
replications, record exact sample/predictor windows, outcome units, pinned
data URLs, hashes and tolerances chosen from reference precision. Do not widen
a tolerance merely to make a failing result pass. Separate published values,
upstream numerical parity and a new empirical application.

## Validate a change

```bash
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m pytest
python -m pytest -m network             # for data/reference changes
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m build
```

`scripts/create_notebooks.py` regenerates notebook sources and clears outputs.
Then `scripts/execute_notebooks.py` runs all notebooks in the current Python
environment, preserving warnings and stopping on failures. Use `--output-dir`
to inspect a new execution without replacing tracked outputs. Keep one
notebook per academic setting, not per method. Do not hand-edit numerical
outputs. Network-dependent tests carry the `network` marker and are excluded
from the default suite.

Figure changes belong in `pysynth.plotting`, with notebooks calling the public
API. Preserve observed-minus-counterfactual signs, actual calendar spacing,
common comparison windows and explicit units. Do not infer uncertainty from
variation across estimators. Test artist coordinates and summary values;
visually review the exported figures at their intended physical dimensions.
After changes, re-execute every affected notebook and inspect its PDF, SVG
and PNG exports under `empirical_examples/figures/`. PDF fonts must be
embedded, captions must fit the canvas, and common scales must retain
outlying methods. `--output-dir` also redirects linked figure files.

## Pull requests and releases

Use focused commits with descriptive messages. Explain the concrete issue,
resulting behavior, verification and limitations in the PR. Update relevant
docs and `CHANGELOG.md`; add a citation when adding a method. Preserve the
repository's GPL-3.0 license and any third-party license notices. Do not copy
noncommercial-only source code into the library.

Before a release, validate both the core and optional backends, execute the
academic reference checks and notebooks in the pinned environment, build
documentation with warnings treated as errors, and inspect the wheel/sdist.
Keep the version in `pyproject.toml`, `pysynth.__version__`, docs and citation
metadata consistent. Publication to a package registry is a separate action.
