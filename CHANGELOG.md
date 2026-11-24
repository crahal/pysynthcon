# Changelog

All notable changes are documented here. Versioning follows semantic
versioning; the API may evolve during the 0.x research releases.

## [Unreleased]

### Added

- Original synthdid bootstrap, jackknife and placebo variance procedures,
  independent draw-by-draw R checks, and the reference `sc_estimate` variant.
- Paired source/reproduction/difference annotations with matched statistics
  and periods; archived earlier-software comparisons, fitting times and
  optimizer diagnostics in every notebook.
- Pinned R reference installer and original SCUL/MSCMT integration checks.
- Sourced `LiteratureBenchmark` annotations on every empirical figure, with
  compatible mean-effect guides, linked citations and explicit source windows.
  Paper findings are distinguished from the authors' Kansas worked example.
- Publication plotting API: observed/counterfactual paths, shared-scale
  effect panels and aligned ATT/pretreatment-RMSE summaries, with explicit
  units, intervention markers and no invented uncertainty intervals.
- Vector PDF/SVG and 300 dpi PNG export, embedded PDF fonts, editable SVG
  text and scoped styling that preserves global Matplotlib settings.
- Three redesigned figures in every empirical notebook, downloadable
  exports and CSV comparison values, plus a dedicated plotting guide.
- Figure tests for calendar spacing, treatment boundaries, common scales,
  summary values, sample compatibility, font export and physical dimensions.
- Explicit `TwoStageIndividualSC` estimator, labelled **Two-Stage Individual
  Synthetic Control (2SISC)**: nearest-neighbor screening on pretreatment
  features, followed by the original ISC procedure within that group.
- Separate ISC and 2SISC rows in all four executed empirical notebooks,
  bringing each comparison to twelve interfaces.
- Donor-selection metadata and analytic checks for full-pool equivalence,
  neighbor ties, feature matching, aggregation and post-treatment leakage.

### Changed

- SCUL defaults to the original R `lambda.median` procedure with fold-specific
  glmnet paths. ISC and 2SISC default to MSCMT and 50 donor resamples.
  R dependencies are now required for these defaults; the previous native
  approximations require explicit selection.
- pysyncon SCM estimates predictor importance instead of injecting fixed V.
  SyntheticControlMethods uses all pre-outcome lags and all ten original starts.
- Nested SCM optimization now checks termination, improves inner numerical
  conditioning, differentiates the same objective through a certified inner
  optimum, and recovers failed line searches or unavailable derivatives with
  bounded Powell. Failed recovery raises rather than returning an unchecked fit.
- ISC preserves the original R donor-sampling stream across individuals.
  Failed final MSCMT WNNLS solves are repaired on the same weighted simplex
  objective, with independently checked optimality gaps and recorded diagnostics.
- Every empirical notebook runs full fits and 200 original SDID placebo refits.
- California also reproduces the paper's 8.4 SDID standard error using the
  authors' exact R resampling plan; every resample is exported for inspection.
- German figure units now state PPP-adjusted current US dollars, following
  the authors' 2025 erratum; the unit-label correction does not alter the data.
- `Estimate.plot()` now uses the publication style and accepts plot labels
  and captions while retaining its Axes return value.
- Notebook generation uses stable cell IDs and public plotting functions;
  custom execution directories include their linked figure exports.
- `IndividualSC()` now unambiguously denotes vanilla ISC over the full
  eligible donor pool. Replace the initial API's
  `IndividualSC(n_neighbors=k)` with `TwoStageIndividualSC(n_neighbors=k)`.
- Both individual estimators take keyword-only options and return method
  labels `ISC` and `2SISC`. Donor resampling remains a conditional sensitivity
  diagnostic; 2SISC keeps its stage-one donor group fixed during resampling.

## [0.1.0] — 2026-11-23

### Added

- Installable `pysynth-control` distribution with the `pysynth` import namespace.
- Validated balanced panels, labelled estimates, comparison tables and plots.
- Native convex outcome SCM and DID baselines.
- SCUL with rolling-origin lasso selection, fold-local preprocessing and
  explicit common-grid selection rules.
- Individual synthetic controls with nearest-donor screening, conditional
  donor resampling, case-specific panels and event-time aggregation.
- Native SDID with unit/time weights, treated blocks, reference Frank–Wolfe
  optimization and convergence diagnostics.
- Optional adapters for pysyncon SCM, augmented, penalized and robust methods,
  and SyntheticControlMethods SCM and differenced SCM.
- In-space placebo diagnostics with explicit donor exclusion and rank rules.
- requests-based immutable, hash-verified academic dataset downloads.
- Four empirical notebooks: California, Basque Country, Germany and Kansas.
- Independent R fixtures, analytic tests, academic reference checks, CI,
  full Sphinx documentation, citation metadata and contributor guidance.

### Compatibility

- Pinned pysyncon revision for corrected augmented-SCM replications.
- Local SyntheticControlMethods fixes for pandas differencing alignment and
  string unit IDs, predictor residual broadcasting and predictor scaling;
  one optimization start avoids its SciPy initialization shape issue.

### Known scope limits

- No automatic staggered-adoption estimator or missing-outcome imputation.
- No common confidence-interval API or synthdid variance implementations.
- SCUL median-grid option differs from glmnet's automatic default paths.
- IndividualSC does not reproduce MSCMT predictor weighting or the complete
  original survey/scanner-data studies. New-method applications in the
  notebooks are labelled separately from validated reference specifications.
