# Local reference dependencies

SCUL and ISC default to the original R implementations. Estimation is offline:
it never installs packages, downloads executable code, or falls back to an
approximation. An unavailable or incorrectly versioned dependency produces an
actionable error.

The R source, setup/validation utilities, and dependency lock are maintained
**outside version control and package distributions**. Obtain the separately
supplied reference bundle before using these backends; cloning this repository
does not provide it. Keep the bundle in the ignored `.local/reference/` folder:

- `runtime/`: the R process adapters, shared helpers, and SCUL functions.
- `scripts/`: the local dependency installer, lock, and independent validators.
- `tests/`: the independent fixture generator.

For a bundle kept elsewhere, set `PYSYNTH_REFERENCE_DIR` to its absolute
`runtime/` path. This is separate from `PYSYNTH_R_LIBRARY`, which locates
installed R packages. The runtime directory is discovered from the checkout
root or its subdirectories. Missing resources cause an explicit error; the
library neither downloads them nor silently substitutes a different estimator.

Once the local bundle is in place, install R, C/Fortran development tools and GLPK
headers. On Debian or Ubuntu:

```sh
sudo apt-get install r-base r-base-dev gfortran libglpk-dev
python .local/reference/scripts/install_r_references.py --library .r-library
```

On macOS, install R and its matching Fortran compiler, and GLPK (for example,
`brew install glpk`). Windows needs Rtools and a GLPK development installation
visible to the compiler. The installer does not change system packages; the
commands above require the usual administrator rights for your system.

The Python installer checks SHA-256 hashes against
`.local/reference/scripts/r-reference-lock.json` before installing the numerical packages. It
pins glmnet **4.0-2**, MSCMT **1.4.0**, and their numerical dependencies. ggplot2
is needed by MSCMT's package namespace; if absent, its normal CRAN installation
also installs its nonnumerical dependencies. Record the printed `sessionInfo()`
when archiving a complete computing environment.

The default `.r-library` is discovered when running from the checkout root.
From another working directory, select its absolute path:

```sh
export PYSYNTH_R_LIBRARY="/absolute/path/to/PySynth/.r-library"
export PYSYNTH_REFERENCE_DIR="/absolute/path/to/PySynth/.local/reference/runtime"
```

Alternatively pass `r_library="/absolute/path/to/library"` to `SCUL`,
`IndividualSC`, or `TwoStageIndividualSC`. Each fit runs an isolated R process;
Python and other fits' random-number generators are unaffected. Results record
the R and method-package versions. The adapters preserve and surface upstream
warnings. If MSCMT's `wnnlsExt` routine reports failure during its final weight
fit or its weight/predictor checks, that call's iterate is rejected: the original
routine can return finite but suboptimal weights after the warning. The adapter
solves the **same weighted simplex quadratic at that call's predictor weights**
with analytic-gradient SLSQP and checks its simplex feasibility and convex
optimality gap independently. Successful `wnnlsExt` calls return unchanged.
The outer optimizer, its compiled `DEoptC` inner routine and `wnnlsOpt` are
unchanged; this repair does not replace or certify those separate solver paths. There is no added ridge,
discarded donor, changed resample, or reduced outer search. If the repair does
not pass those checks, estimation fails. Each repaired call appears in
`estimate.metadata["inner_solver_repairs"]`, with the original status, draw,
iterations, objective and dual gap. This is a numerical compatibility repair,
not bitwise parity with an erroneous upstream iterate. The modification to
MSCMT's helper exists only inside that fit's isolated R process.
The reported dual gap bounds suboptimality of the repair solver's normalized
mean-squared objective, `mean((sqrt(V) X w / scale)**2)`, where `scale` is the
largest absolute entry of `sqrt(V) X`. The acceptance bound `1e-8` is in those
normalized objective units, not raw outcome units. A common positive scaling
changes neither the simplex problem nor its minimizing weights. The separately
reported `objective` is the weighted mean squared error before this common
scaling; it is not MSCMT's unweighted dependent-variable loss.

## What is pinned and what is changed

- The local bundle's SCUL functions follow commit
  `121b588cf8e4eebcea68518b57a7c754040153bb`. glmnet 4.0-2 is the version in the
  authors' published tutorial. Its original solver tolerances, automatic
  fold-specific paths, median rule and exact refit are retained.
- Two validation matrix slices explicitly preserve dimensions
  for a one-period horizon. This fixes an R shape error; it does not change the
  objective, path, or selection rule.
- glmnet's C progress-bar helper uses a current R language-object constructor
  instead of `SET_TYPEOF`, which recent R removed. No statistical or numerical
  solver code is changed. The installer verifies the expected source before
  applying this compatibility patch.
- ISC calls MSCMT directly with its original default optimization settings.
  The costofcare source does not record its MSCMT version; 1.4.0 was the release
  available at that repository's December 2024 revision. The adapter therefore
  promises parity with the explicitly pinned reference, not with an unknown
  historical environment.
- ISC defaults to costofcare's 50 donor resamples, R sampling seed 42, and
  MSCMT optimizer seed 1. The standalone Vagni ISC demonstration used 10;
  request that count explicitly when reproducing that demonstration.
- `fit_many` advances one R donor-sampling stream through the supplied case
  order, preserving the original `mclapply` workflow's separation between
  sampling and optimization RNGs. State is captured before MSCMT reseeds and
  transferred privately to the next case. Archive the case order together
  with the seed; reordering cases changes the donor draws. Standalone `fit`
  remains independently reproducible from its declared seed.

For numerical verification:

```sh
python -m pytest tests/test_r_references.py -m r_reference
```

These integration checks compare the Python interface against independently
generated direct R calls, including all three SCUL penalty-selection rules,
ISC's covariate groups, 50 donor-resampling draws, and the continuous sampling
stream across two cases with different donor pools. The independent generator
is `.local/reference/tests/reference_backends.R`; fixture generation never calls the
Python adapter. The native sklearn and fixed-metric alternatives remain
explicit research options, not substitutes for these reference tests.
Tests requiring the local bundle are marked `r_reference` and report an
explicit skip when it is absent. GitHub CI runs the independent numerical
fixtures in Python; the original-R integration checks run in a provisioned
local environment.
