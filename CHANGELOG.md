# Changelog

All notable changes are documented here. Versioning follows semantic
versioning; the API may evolve during the 0.x research releases.

## [Unreleased]

### Inference and numerical audit

- Add sharp-path conformal tests for outcome-matching SCM and pointwise inversion over a supplied effect grid. Return every rotation and score, null-imposed weights and residuals. Plot accepted grid candidates with optional independently sourced endpoints.
- Reproduce four Rhode Island p-values and six annual pointwise sets in a new empirical notebook. The existing paired spillover simulation now tests the known effect path under clean and exposed donor pools.
- Add full in-space treatment reassignment, panel-specific estimator factories, optional parallel refits, squared-error screens, and placebo gap plots. The California notebook uses the original smoking covariates and all 39 assignments in both the local R Synth package and the checked Python predictor optimizer. The untreated-only pool remains the default.
- Correct false convergence in outcome SCM when a remote donor sets the numerical scale. Record any improving donor transfers and enforce the iteration budget. Require a boolean for the historical post-treatment predictor override.
- Read the Stata tobacco predictors and the Rhode Island supplement directly from checksum-verified response bytes.

### Estimator attribution

- Give predictor-based, ridge-augmented, penalized and robust SCM their own
  public classes and cite the original methodology for each. Expose the
  differenced specification under its own name and identify its working-paper
  source. Existing backend selectors remain available for compatibility.
- Compare each named specification once in the empirical notebooks. Classical
  predictor SCM uses three prespecified pretreatment outcome means. The
  historical fits keep their original predictor designs. Germany’s figures
  cite the paper’s approximate annual GDP effect.

### Documentation and figures

- Place the fitted mean ATT in the lower-left corner of effect panels and
  remove its arrow. The dotted line shows the mean over the treated periods.
- Keep each estimator in one equally sized effect panel. The Basque example
  shows all twelve fits in a three-column grid, with the paper and R Synth
  comparisons printed inside the historical SCM panel.
- Keep the ATT and pretreatment RMSE dots on evenly spaced rows when a fit
  has several reference comparisons. Numbered keys beside the estimator
  lead to the values and evaluation windows in the figure’s citation lines.
- Set figure titles in STIX Two Text and labels, ticks and estimates in
  Source Sans 3. Bundle the fonts for consistent publication exports and
  use the same pairing in the desktop and mobile README headers.
- Measure multiline captions and stacked published estimates with the active
  renderer so their labels fit across Matplotlib versions.
- Use the PySynthCon name in the README, citation, documentation and exported
  figure credits. The README header shows the California SDID estimate with
  separate layouts for desktop and mobile screens.

### Simulation and notebook outputs

- Add a seeded, paired regional-policy simulation comparing outcome SCM and
  DID under donor spillovers. It uses expanding-window validation, a named
  donor-group exclusion and known potential outcomes to measure estimation error.
- Write standalone figures, tables and diagnostics from every notebook under
  the checkout's `outputs/` directory. Export complete empirical donor weights,
  selection results, replication checks, warnings and software versions.
  Empirical source data download into memory. Simulation panels are generated
  in memory from the recorded seed.
- `scripts/execute_notebooks.py` runs the four empirical studies and the
  simulation. `--examples` selects studies, and `--output-dir` selects the
  destination of executed notebook copies.

### Estimator and replication audit

- Check successful pysyncon simplex solves for a feasible step that improves
  the objective beyond the solver tolerance. Recover the same quadratic when
  this check fails. Handle undefined scaled imbalance after a completed
  AugSynth fit and reject singular unregularized or residualized designs.
- Validate complete, finite, aligned paths before computing effects, placebo
  ranks or validation scores. Preserve refit results when a custom estimator
  reuses a prediction buffer. Check an existing SDID fit against the
  counterfactual and intercept implied by its weights.
- Reject bare strings passed as collections of unit IDs. This prevents
  `treated_units="CA"` from silently assigning treatment to units `C` and `A`.
  In placebo output, `at_least_treated_ratio` records the rank comparison
  even when the displayed post/pre RMSPE ratio exceeds floating-point range.
- Correct numeric scaling in SCUL's optional Python backend, 2SISC neighbor
  ordering and extreme placebo ratios. Permit a fixed SCUL penalty with one
  post-treatment observation in the original R backend.
- Validate historical calendars and Kansas's sixteen quarterly effects.
  Distinguish the Basque paper's qualitative headline from its Figure 2
  percentage series. Direct labels show a bound for differences smaller
  than display precision. CSVs retain the full numerical difference.
  Put grouped solver warnings below notebook figures. The Kansas donor
  sensitivity chart labels states using the study's FIPS-to-name mapping.
- Limit notebook execution to the four published examples, and report a
  missing notebook as an error. Include only those notebooks and the two
  development scripts in their respective source-distribution directories.
  Check explicitly supplied R library directories before a reference fit.

### Direct figure annotations

- Label observed and counterfactual curves at their endpoints; place sourced
  comparisons, reproduced values and signed differences beside their matching
  paths, effect panels and ATT rows. Direct annotation is the default; the
  earlier reference table remains available with `reference_style="strip"`.
- Give historical fits absent from the comparison their own labelled rows or
  panels. Keep percentage and subset-window comparisons distinct from
  full-period, level-scale ATT markers.
- Label validation minima, baseline ATTs and the donor exclusions responsible
  for the largest sensitivity changes directly on the evaluation figures.
- Move full specifications and audit tables into expandable notebook sections
  below the figures, with concise source citations and local export paths.

### SDID inference convergence

- Add an explicit inference iteration-budget override, including when a
  fitted estimate is supplied. Preserve original penalties, tolerance,
  starting weights, sparsification and resampling plans.
- Add a strict convergence option that rejects unresolved resampled fits;
  report capped draws by weight type and retain objective-decrease thresholds,
  terminal decreases, dual gaps and iteration budgets in diagnostics.
- Make iteration-limit warnings actionable. Keep exact original-cap
  replication and compare it with a complete extended-budget run in every
  notebook, showing standard-error changes and exporting draw-level diagnostics.

### Comparative research workflows

- Add common expanding-window pretreatment validation across estimator
  families and tuning candidates, masking held-out treated outcomes and
  excluding all actual post-treatment rows. Preserve fold predictions,
  pooled scoring, solver diagnostics and fresh estimator instances.
- Add validation-based specification selection and full-panel refitting,
  including prespecified 2SISC neighbor grids. Document the optimistic selected
  score and absence of post-selection inference guarantees.
- Add complete donor-deletion and named-group exclusion refits, with baseline
  paths, ATT changes, sign reversals and diagnostic tables. Rerun 2SISC
  screening within every eligible pool; retain zero- and negative-weight donors.
- Add publication plots for validation and donor sensitivity, and demonstrate
  both workflows in all four empirical notebooks.
- Credit Petrillo et al. (2025) for two-stage ISC. Make the authors' core Python
  specification the 2SISC default: raw nearest-neighbor screening followed by
  temporal-mean predictor SCM, without donor resampling. Preserve the earlier
  variant through `backend="mscmt", n_bootstrap=50`. Document the source's
  treated-inclusive neighbor-count convention and study-specific limitations.
- Document differentiation from pysyncon and related tools, crediting existing
  methods and diagnostics without claiming new causal identification.

### Audit corrections

- Correct Basque eligibility to 16 regional donors and reproduce paper Table 3
  using mean annual education shares. Add independent original-R Synth checks
  of predictors, weights and the full path at absolute tolerance `1e-4`.
- Restore original SCUL near-zero-variance screening across full series and
  every training fold; allow one-origin median/minimum selection.
- Preserve SCM predictions across measurement-unit and common-level changes;
  retain informative lags regardless of absolute outcome scale.
- Check pysyncon inner/outer termination and recover failed solves without
  changing objectives. Reject silent post-treatment predictor use and mismatched
  outcome data; record Germany’s explicit historical exception.
- Correct robust SCM handling of genuine zeros, singular-value thresholds and
  denoised prediction; validate against the published SVD/ridge equations.
- Make RMSPE and generic placebo ranks scale invariant; validate supplied SDID
  time-weight alignment and simplex weights. Clarify the descriptive placebo
  rank and the original SDID paper/software finite-sample differences.
- Fix invalid local diagnostic JSON, validate generated notebook code, and
  clean up notebook kernels after success or failure. Regenerate empirical
  comparisons after the audit. R and auxiliary exports
  remain local-only; source distributions explicitly exclude restored ignored
  helpers and executed notebook copies. See `docs/audit.md` for sources, evidence and limits.

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
  features, followed by the authors' mean-predictor fit within that group;
  the separate MSCMT/resampling variant remains available explicitly.
- Separate ISC and 2SISC rows in all four executed empirical notebooks,
  bringing each comparison to twelve interfaces.
- Donor-selection metadata and analytic checks for full-pool equivalence,
  neighbor ties, feature matching, aggregation and post-treatment leakage.

### Changed

- Generated figure exports, diagnostic tables and prior-output archives are
  local-only; executed notebooks retain their embedded figures and results.
- R source and reference setup/validation utilities are local-only dependencies,
  selected through `PYSYNTH_REFERENCE_DIR`. They are excluded from the repository
  and distributions. Full SCUL/ISC fits and California's exact paper sampling
  plan require the separately supplied bundle. GitHub CI runs Python checks;
  original-R integration remains available locally.
- SCUL defaults to the original R `lambda.median` procedure with fold-specific
  glmnet paths. ISC defaults to MSCMT and 50 donor resamples; the explicit
  MSCMT variant of 2SISC retains that workflow.
  R dependencies are required for these fits; the previous native
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
  diagnostic; the MSCMT variant of 2SISC keeps its stage-one donor group fixed
  during resampling.

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
