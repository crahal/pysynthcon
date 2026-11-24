# Publication figures

The plotting functions compare fitted `Estimate` objects on the same observed
outcome path and treatment window. They return Matplotlib objects, so you can
adjust the design for a journal without rebuilding the figure from scratch.
Install plotting support with `pip install -e '.[plot]'` in a source checkout.

Three complementary figures answer different questions:

| Function | What it shows | Returns |
|---|---|---|
| `plot_paths(result)` | Observed outcome and a labelled counterfactual | Matplotlib `Axes` |
| `plot_effects(results)` | Effect paths in small multiples on common scales | `Figure`, two-dimensional `Axes` array |
| `plot_comparison(results)` | Aligned ATT and pretreatment-RMSE dot plots | `Figure`, two-element `Axes` array |

`Estimate.plot()` delegates to `plot_paths` and retains its existing Axes
return value. With `effects=True`, it draws a single observed-minus-synthetic
path with a zero reference line.

## Compare estimators

```python
from pysynth import (
    SyntheticControl, SCUL, IndividualSC, TwoStageIndividualSC, SyntheticDID,
    plot_comparison, plot_effects, save_figure,
)
from pysynth.datasets import DATASETS, load_dataset

panel = DATASETS["california"].panel(load_dataset("california"))
models = {
    "Outcome SCM": SyntheticControl(),
    "SCUL": SCUL(initial_window=5, horizon=12),
    "ISC": IndividualSC(),
    "2SISC": TwoStageIndividualSC(n_neighbors=10),
    "SDID": SyntheticDID(),
}
fits = {name: model.fit(panel) for name, model in models.items()}

fig, axes = plot_comparison(
    fits,
    title="California Proposition 99",
    subtitle="1989–2000 post-treatment; identical outcomes and donor eligibility",
    units="packs per capita",
)
save_figure(fig, "outputs/custom/california-comparison")

fig, axes = plot_effects(
    fits, ncols=3,
    title="California Proposition 99",
    ylabel="Effect on cigarette sales (packs per capita)",
    xlabel="Year",
)
save_figure(fig, "outputs/custom/california-effects")
```

The figures preserve dictionary order; they do not rank estimators by effect
size or fit. Use `labels={"original name": "short display name"}` for long
method names. `colors={"original name": "#286A8C"}` can assign specific colors;
the default uses a single accessible dark blue, with labels carrying method
identity. Neither option changes the fitted results or their mapping keys.

All estimates must have identical observed outcomes, ordered time indexes
and treatment boundaries. Invalid comparisons fail before a figure is
created. The functions do not align mismatched studies by dropping rows,
interpolate missing periods, or silently fit unfitted estimators.

## Read the figures correctly

- Effect paths use **observed minus counterfactual**, including any fitted
  intercept or SDID level correction already present in the result.
- In the default effect panels, the lower-left `Mean ATT` label gives the
  average post-treatment gap. A dotted line at that value spans the treated
  periods. Panels paired with a published comparison show its source and
  evaluation window.
- The dashed vertical line is exactly the first affected observation,
  `result.observed.index[result.n_pre]`. The pale background marks the
  post-period. It is not a confidence band.
- Small multiples share x and y axes by default. Their vertical limits
  include every plotted effect and zero; an outlying method is not clipped.
  `sharey=False` explicitly permits different vertical scales. `ylim=(low,
  high)` is an explicit user-selected view and can crop data; disclose that
  choice in a caption.
- The summary uses each result's ATT and pretreatment RMSE over the common
  sample. ATT is the mean post-treatment gap. Lower RMSE describes better
  pretreatment fit, not stronger causal identification.
- Dots are point estimates. The API does not invent confidence intervals,
  significance markers or uncertainty ribbons from between-method variation.

Numeric years and decimal quarters retain their actual spacing. Datetime
indexes use calendar coordinates and date ticks; `PeriodIndex` values use
period starts with period labels. Timedeltas use days. Other labels use
ordinal positions with explicit category labels. No monthly, quarterly or
annual frequency is inferred from arbitrary numbers.

## A reference counterfactual

```python
from pysynth import plot_paths

ax = plot_paths(
    fits["SDID"],
    title="California Proposition 99",
    subtitle="Observed consumption and no-covariate SDID counterfactual",
    ylabel="Cigarette sales (packs per capita)", xlabel="Year",
    observed_label="Observed California",
    counterfactual_label="SDID counterfactual",
    intervention_label="1989",
    note="Shading marks the post-treatment period. No uncertainty interval is shown.",
)
save_figure(ax.figure, "outputs/custom/california-reference")
```

Observed and counterfactual paths differ by both line style and color, so
their meaning remains clear in grayscale. Labels connect directly to the
path endpoints. Set `reference_style="strip"` to use the earlier legend and
source-strip layout instead.
`subtitle` and `note` reserve measured caption space in a new figure.
To embed a plot into your own figure, pass `ax=...` and place figure-level
captions on that parent figure yourself.

## Export at manuscript dimensions

```python
fig, axes = plot_comparison(fits, figsize=(7.2, 4.2), units="packs per capita")

# All three files, at the same physical dimensions:
paths = save_figure(fig, "outputs/custom/main-result")

# One explicit format, or a custom selection:
save_figure(fig, "outputs/custom/main-result.pdf")
save_figure(fig, "outputs/custom/main-result", formats=("svg", "png"), dpi=600)
```

The empirical notebooks export figures at 7.2 inches wide. For a journal
submission, set the figure width to its final printed size before saving.
Titles use **STIX Two Text**. Axis labels, estimates and direct annotations
use **Source Sans 3**, whose equal-width digits keep neighbouring values easy
to compare. Both families ship with PySynthCon.

PDF embeds the typefaces. SVG keeps the labels editable and includes the
glyphs used in the figure. Both formats retain vector lines at any scale.
PNG uses the same typefaces and defaults to 300 dpi.

The save helper preserves physical dimensions rather than cropping to a
content-dependent bounding box. It creates parent directories, replaces
existing destinations, returns their paths, and leaves the figure open.
Captions are wrapped using actual font measurements. After adjusting a
figure's size, use `save_figure` again to refresh caption wrapping.

```python
from pysynth import publication_style, publication_title_font
import matplotlib.pyplot as plt

with publication_style():
    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")
    fits["SDID"].plot(ax=ax, effects=True, ylabel="Effect (packs per capita)")
    fig.suptitle("California Proposition 99", fontproperties=publication_title_font(), fontsize=15)
    save_figure(fig, "outputs/custom/single-column.pdf")
```

Styling is scoped: global Matplotlib settings are restored after each call.
The library does not change your backend, call `show()`, or close figures.
Importing `pysynth` does not require Matplotlib until plotting is requested.

## Compare a published result with its reproduction

Use `ReferenceComparison` to pair a published result with the **same fitted
specification**. By default, `reference_style="direct"` places source values,
reproductions and signed differences, `Δ = reproduced − source`, next to the
corresponding curves, panels or ATT rows. Short numbered citations appear in
the figure footer; callouts link to their sources in vector
exports. The calculation uses the actual fitted effect path; it never takes
a manually entered reproduction.

```python
from pysynth import ReferenceComparison, reference_comparison_table

reference = ReferenceComparison(
    label="SDID",
    estimate=fits["SDID"],
    reference=-15.6,
    window=(1989, 2000),
    units="packs per capita",
    citation="Arkhangelsky et al. (2021), Table 1",
    source_url="https://arxiv.org/pdf/1812.09970v4#page=8",
    context="No covariates. Source rounded to one decimal; Δ includes rounding.",
    reference_precision=1,
    precision=3,
)

fig, axes = plot_comparison(fits, references=[reference], units="packs per capita")
fig, axes = plot_effects(fits, references=[reference], ylabel="Effect (packs per capita)")
ax = plot_paths(fits["SDID"], references=[reference])
reference_comparison_table([reference]).to_csv("reference-checks.csv", index=False)

# Optional earlier layout, with the complete source text above the chart:
fig, axes = plot_comparison(fits, references=[reference], reference_style="strip")
```

The source remains at its printed precision. `precision` controls the display
of the reproduced result and difference. Exact agreement displays an unsigned
zero. A nonzero difference smaller than one unit of the displayed precision
appears as a bound, for example `|Δ| < 0.0001`. The table retains the full
numeric `difference` alongside the figure's `difference_label`. For an
approximate headline, set `approximate=True` to mark the source and displayed
point differences with `≈`. Differences from rounded or approximate
headlines are descriptive, not optimization errors or replication tolerances.

A hollow ochre diamond identifies a compatible source mean on the matching
estimator's ATT row. A single comparison sits beside the estimate. For
several comparisons, numbered keys follow the estimator name and the linked
citation lines below the axes give each source value, reproduction and
window. Both plots keep their ordinary row spacing. The full-period ATT
stays distinct from a percentage headline or a check over a shorter window.
An ochre dotted segment marks a source mean over its evaluation window when
the units match the effect axis.
Each fit occupies one cell in the effect grid. When several sources refer to
the same fit, the panel shows a short linked line for each source and its
reproduced value. The labels sit together in a clear corner, with the
evaluation windows and units in the numbered citations. In the Basque
notebook, the paper’s percentage headline and the R Synth level comparison
both fit below the historical curve. If a panel is too small to hold its
labels clear of the data, keys beside its title lead to the full comparisons
in the figure caption.
Outcome paths carry textual callouts, and RMSE never receives an ATT guide.

Matching uses the exact `Estimate` object, not similar labels or numerical
values. In direct mode, a historical fit absent from the supplied mapping is
automatically added as a separate `Reference · …` row or panel. References
sharing the same estimate object use one row or panel. The supplied mapping
and fitted values remain unchanged. This keeps historical specifications
visible without attaching their benchmark to a different estimator. Strip
mode retains the original mapping and reports any separate historical fit
in the source strip instead.

A reference window can cover a subset of the post-period. Its reproduced
value is calculated on that subset, and its guide on effect paths covers
only those dates. On a full-period ATT row, the subset comparison is text
only; it does not become an incompatible numeric marker.
Invalid, missing, reversed or pretreatment endpoints raise an error. Every
reference must share the plotted case's observed outcome and intervention
boundary.

For percentage comparisons, `statistic="mean_percent_effect"` computes
the mean of `100 * (observed - counterfactual) / observed`; its difference is
reported in percentage points. The denominator is the observed outcome,
and averaging occurs **after** division. This statistic is used for the
Basque paper's Figure 2 convention. Figure 1 gives an approximate visual
summary without defining an average. For this case `comparable=False` shows
the headline alongside our Figure 2 calculation and leaves the difference
empty in the audit table and CSV. The percentage appears as text when
the main figure uses dollar units; no percentage guide is drawn on those
axes. Zero observed denominators raise an error.

The empirical notebooks call
`pysynth.replication.reference_comparisons(name, reference)` to bind their
historical fits to sourced results. Basque and German annotations distinguish
approximate paper headlines from precise upstream software targets. The
California helper also accepts `additional={"SC": sc, "DID": did}` for the
paper's own SC and DID specifications. Kansas cites the authors' vignette and
prints the quarterly evaluation window. See [replication provenance](replication.md)
for source locators and specification details. Full context and numerical
precision remain available in `reference_comparison_table`; the notebooks
show these tables in expandable audit sections below the figures and export
them as CSV.

## Annotate a general literature benchmark

The earlier `LiteratureBenchmark` interface remains available for descriptive
context without a paired reproduction. Pass it to any of the three plotting
functions. By default its headline appears in a callout inside the chart,
with its citation and specification context in the footer. A numeric callout
points to its source guide on effect and ATT axes. A text-only callout has
no leader. The source link is attached to the callout. Use `reference_style="strip"` for the
earlier warm-toned source strip with its full specification context. Prefer
`ReferenceComparison` when a source result belongs to one exact fitted
specification.

```python
from pysynth import LiteratureBenchmark

published = LiteratureBenchmark(
    headline="SDID mean effect: −15.6 packs per capita",
    citation="Arkhangelsky et al. (2021), Table 1 · 1989–2000",
    source_url="https://arxiv.org/pdf/1812.09970v4#page=8",
    context="No-covariate SDID; other specifications need not match this value.",
    value=-15.6,
    window=(1989, 2000),
)
fig, axes = plot_comparison(fits, benchmark=published, units="packs per capita")
save_figure(fig, "outputs/custom/comparison-with-literature")
```

A numeric `value` adds a dotted guide to the ATT axis and to the post-period
of effect paths. It is the source's **mean effect**, not an uncertainty
interval or a published time-varying trajectory. The same annotation on an
observed/counterfactual figure is textual: an ATT does not belong on an
outcome-level axis. The RMSE axis never receives an ATT guide. Automatic
effect and ATT limits include the benchmark, even when it falls outside the
new estimates; explicit `ylim` remains a user-selected view.

The caller must establish compatible outcome units, transformation and
estimand before setting `value`. The endpoints in `window` must exactly match
the plotted post-treatment sample; incompatible windows raise an error before
plotting. Different predictor specifications can still be compared, but must
be identified in `context`. These annotations do not certify replication.

Omit `value` and `window` for a result in different units or over a different
window. For example, the Basque paper's approximately 10% GDP loss is a text
annotation on figures measured in thousands of dollars. It is not converted
to a dollar ATT or drawn at −10. `label="Authors’ reference"` distinguishes a
worked software example from a number printed in a journal article.

`pysynth.replication.LITERATURE_BENCHMARKS` contains the original sourced annotations.
The notebooks now use the paired `reference_comparisons` helper above. Use
either `references` or `benchmark` on a figure; supplying both raises an error.
See the [benchmark provenance](replication.md)
for page/table locators, the pinned Kansas vignette, and the German erratum.
Strip mode allows extra room for the source strip; explicit `figsize` values
are respected. With a caller-supplied `ax`, place literature annotations on
the parent figure instead. Review label placement after changing dimensions
or adding long custom text; Matplotlib artists remain editable.

## Reproduce the notebook figures

Each academic notebook saves a reference path, an ATT/RMSE comparison, an
effect-panel figure, a validation figure and a donor-sensitivity figure,
plus comparison and source-check values as CSV under
`outputs/empirical_examples/<example>/`. The donor-spillover simulation writes
its plots and tables to `outputs/simulated_examples/donor_spillovers/`.
These generated files stay local and
are excluded from version control and source distributions. The committed
notebooks embed all five figures, with local export paths and literature links immediately
after each figure. Full setup notes, specification details and audit tables
can be expanded below the charts. The notebooks also group emitted fit
warnings by category and message and report their counts below the relevant
figure. The inference section prints the number of original-cap SDID draws
that stopped at the limit. Rerun
the notebooks to create PDF, SVG and PNG files. Historical reference fits
remain explicitly labelled when shown alongside common-comparison methods.

`plot_backtest(validation)` displays pooled held-out RMSE beside all fold
scores on a common color scale, directly labelling the lowest pooled RMSE
(including exact ties). `plot_donor_sensitivity(sensitivity)` labels the baseline and largest
absolute ATT change beside the relevant estimates, including the donor
exclusion responsible. It also shows every exclusion estimate and the
scenario range. Its optional `unit_labels` maps donor IDs to names used
in single-donor annotations. For example, the Kansas notebook uses the
source data's state names for FIPS codes. The fitted panels and exported
scenario tables retain their original IDs. Both accept `title`, `units`, `labels`, `figsize` and
`note`; both return a Figure and two editable Axes. Neither invents confidence
intervals or attributes these new diagnostics to a paper's published results.
See the [evaluation guide](evaluation.md) for complete examples.

```bash
python scripts/execute_notebooks.py
# Save executed notebook copies elsewhere. Result files still go under ./outputs/.
python scripts/execute_notebooks.py --output-dir /tmp/pysynth-rendered
```

See [replication conventions](replication.md) for the source data, time windows
and limits of the estimator comparisons. Better figures do not resolve those
estimation or identification limits.
