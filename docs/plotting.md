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
    "SCUL": SCUL(),
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
save_figure(fig, "figures/california-comparison")

fig, axes = plot_effects(
    fits, ncols=3,
    title="California Proposition 99",
    ylabel="Effect on cigarette sales (packs per capita)",
    xlabel="Year",
)
save_figure(fig, "figures/california-effects")
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
save_figure(ax.figure, "figures/california-reference")
```

Observed and counterfactual paths differ by both line style and color, so
their meaning remains clear in grayscale. The legend sits above the data.
`subtitle` and `note` reserve measured caption space in a new figure.
To embed a plot into your own figure, pass `ax=...` and place figure-level
captions on that parent figure yourself.

## Export at manuscript dimensions

```python
fig, axes = plot_comparison(fits, figsize=(7.2, 4.2), units="packs per capita")

# All three files, at the same physical dimensions:
paths = save_figure(fig, "figures/main-result")

# One explicit format, or a custom selection:
save_figure(fig, "figures/main-result.pdf")
save_figure(fig, "figures/main-result", formats=("svg", "png"), dpi=600)
```

Figure sizes are in inches. Notebook exports use a 7.2-inch width, appropriate
for many two-column manuscript layouts; adjust to the target journal. PNG
defaults to 300 dpi. PDF and SVG retain vector lines and text, independently
of that raster resolution. PDF embeds TrueType fonts; SVG keeps editable
text. The bundled DejaVu Sans font avoids a system-font or TeX dependency.

The save helper preserves physical dimensions rather than cropping to a
content-dependent bounding box. It creates parent directories, replaces
existing destinations, returns their paths, and leaves the figure open.
Captions are wrapped using actual font measurements. After adjusting a
figure's size, use `save_figure` again to refresh caption wrapping.

```python
from pysynth import publication_style
import matplotlib.pyplot as plt

with publication_style():
    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")
    fits["SDID"].plot(ax=ax, effects=True, ylabel="Effect (packs per capita)")
    fig.savefig("figures/single-column.pdf")
```

Styling is scoped: global Matplotlib settings are restored after each call.
The library does not change your backend, call `show()`, or close figures.
Importing `pysynth` does not require Matplotlib until plotting is requested.

## Annotate a literature benchmark

Pass a `LiteratureBenchmark` to any of the three plotting functions. The
result appears in a separate, warm-toned source strip, with its reported
precision, citation and specification context. Citation text links to the
source in vector exports; notebooks also print a clickable source link.

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
save_figure(fig, "figures/comparison-with-literature")
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

`pysynth.replication.LITERATURE_BENCHMARKS` contains the sourced annotations
used by the four notebooks. See the [benchmark provenance](replication.md)
for page/table locators, the pinned Kansas vignette, and the German erratum.
Default figure heights allow extra room for the source strip; explicit
`figsize` values are respected. With a caller-supplied `ax`, place literature
annotations on the parent figure instead.

## Reproduce the notebook figures

Each academic notebook saves a reference path, an ATT/RMSE comparison, an
effect-panel figure, and the comparison values as CSV under
`empirical_examples/figures/<example>/`. The notebook contains download links
to its PDF, SVG and PNG exports. The historical reference figure is explicitly
separate from the common-comparison specifications.

```bash
python scripts/execute_notebooks.py
# Write executed notebooks and their linked figures to another directory:
python scripts/execute_notebooks.py --output-dir /tmp/pysynth-rendered
```

See [replication conventions](replication.md) for the source data, time windows
and limits of the estimator comparisons. Better figures do not resolve those
estimation or identification limits.
