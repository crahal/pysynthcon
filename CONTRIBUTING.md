# Contributing to PySynth

Reference-method changes must preserve the full fitting specification: no
reduced restart counts, shortened lag histories, fixed predictor metrics, or
substituted tuning rules in replication examples. Run the independent original
R checks for affected methods using the separately supplied local bundle
described in [the reference setup guide](https://github.com/crahal/pysynthcon/blob/main/docs/reference_backends.md).
Install its pinned dependencies with
`python .local/reference/scripts/install_r_references.py --library .r-library`; R, a Fortran
compiler and GLPK development headers are required. Keep numerical recovery
diagnostics and distinguish algorithm parity from replication of a study's data
and covariate design. New figure benchmarks need a source, matched window,
statistic, units, and an explicit precision or approximation label.

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

Changes to the shared evaluation workflows must preserve fresh estimator
instances, identical folds, masked held-out treated outcomes, and complete
refits under the original fitting settings. Covariate wrappers must use the
received panel's training dates and eligible units. Keep every candidate's
diagnostics; never silently drop failed fits. Donor-sensitivity ranges are
descriptive, and selection scores do not supply post-selection inference.

## Validate a change

```bash
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m pytest
python -m pytest -m network             # for data/reference changes
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m build
```

`scripts/create_notebooks.py` regenerates the four empirical notebook sources
and clears their embedded outputs. The simulation source lives directly in
`simulated_examples/donor_spillovers.ipynb`. Run `scripts/execute_notebooks.py`
with the active Python interpreter to execute all five, or use
`--examples donor_spillovers` for the simulation alone. Use `--output-dir` to
write executed notebook copies to another directory. Standalone result exports
always go under the checkout's `outputs/`. Keep one notebook per study. Download
empirical inputs into memory and generate simulation inputs from a recorded
seed. Network-dependent tests carry the `network` marker.

Figure changes belong in `pysynth.plotting` or `pysynth.evaluation_plotting`,
with notebooks calling the public API. Preserve observed-minus-counterfactual
signs, actual calendar spacing, common comparison windows and explicit units.
Do not infer uncertainty from
variation across estimators. Test artist coordinates and summary values;
visually review the exported figures at their intended physical dimensions.
After changes, re-execute every affected notebook and inspect its PDF, SVG
and PNG exports under `outputs/empirical_examples/<study>/` or
`outputs/simulated_examples/<study>/`. PDF fonts must be
embedded, captions must fit the canvas, and common scales must retain
outlying methods. Commit the executed notebooks with embedded figures;
generated figure exports, CSVs and diagnostics remain local.

Keep R source and reference setup/validation utilities under ignored
`.local/reference/`. Do not embed R source in tracked Python files. The small
numerical JSON fixtures are regression-test inputs and remain tracked.
GitHub CI runs the Python checks and builds without uploading auxiliary
artifacts; original-R integration and complete notebook execution run locally
with the reference bundle provisioned.

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
