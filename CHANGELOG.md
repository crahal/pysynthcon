# Changelog

All notable changes are documented here. Versioning follows semantic
versioning; the API may evolve during the 0.x research releases.

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
