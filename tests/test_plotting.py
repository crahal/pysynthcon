"""Figure correctness: samples, coordinates, scales, summaries and real exports."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pysynth import (
    Estimate,
    SyntheticControl,
    plot_comparison,
    plot_effects,
    plot_paths,
    save_figure,
)

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def example(times):
    observed = pd.Series([2, 3, 5, 7], index=times)
    return Estimate(
        "Example",
        observed,
        observed - np.array([0.2, -0.2, 1, 3]),
        pd.Series([0.5, 0.5], index=["donor A", "donor B"]),
        n_pre=2,
    )


@pytest.mark.parametrize(
    "times",
    [
        pd.Index([2000, 2001, 2004, 2010]),
        pd.Index([2000.0, 2000.25, 2001.0, 2001.75]),
        pd.DatetimeIndex(["2000-01-01", "2000-01-03", "2000-02-01", "2001-01-01"]),
        pd.date_range("2020-01-01", periods=4, freq="MS", tz="Europe/London"),
        pd.period_range("2000Q1", periods=4, freq="Q"),
        pd.to_timedelta([0, 1, 4, 8], unit="D"),
        pd.Index(["period A", "period B", "period C", "period D"]),
    ],
)
def test_time_coordinates_and_first_affected_period(times):
    from matplotlib.dates import date2num

    result = example(times)
    ax = plot_paths(result)
    ax.figure.canvas.draw()
    if isinstance(times, pd.PeriodIndex):
        expected = date2num(times.to_timestamp().to_pydatetime())
    elif isinstance(times, pd.DatetimeIndex):
        expected = date2num(times.to_pydatetime())
    elif isinstance(times, pd.TimedeltaIndex):
        expected = times.total_seconds().to_numpy() / 86400
    elif pd.api.types.is_numeric_dtype(times.dtype):
        expected = times.to_numpy()
    else:
        expected = np.arange(4)
        assert [t.get_text() for t in ax.get_xticklabels()] == times.tolist()
    lines = {line.get_label(): line for line in ax.lines}
    np.testing.assert_array_equal(lines["Observed"].get_xdata(), expected)
    np.testing.assert_array_equal(lines["Observed"].get_ydata(), result.observed)
    np.testing.assert_array_equal(lines["Synthetic control"].get_ydata(), result.counterfactual)
    np.testing.assert_array_equal(lines["Intervention"].get_xdata(), [expected[2], expected[2]])


def test_effects_share_limits_and_keep_all_values_and_method_order():
    first = example(pd.Index([2000, 2001, 2004, 2010]))
    outlier = replace(first, counterfactual=first.observed - [-100, 0, 200, 300])
    mapping = {"First": first, "Extreme": outlier, "Last": first}
    fig, axes = plot_effects(mapping, ncols=2)
    fig.canvas.draw()
    assert axes.shape == (2, 2)
    assert not axes[1, 1].get_visible()
    limits = [ax.get_ylim() for ax in axes.flat if ax.get_visible()]
    assert limits[0] == limits[1] == limits[2]
    assert limits[0][0] < -100 and limits[0][1] > 300
    for ax, (label, result) in zip(axes.flat, mapping.items()):
        line = next(line for line in ax.lines if line.get_label() == label)
        np.testing.assert_array_equal(line.get_ydata(), result.effects)
        assert label in ax.get_title(loc="left")


def test_effects_all_zero_and_independent_scales_are_explicit():
    first = example(pd.Index([1, 2, 3, 4]))
    zero = replace(first, counterfactual=first.observed.copy())
    _, axes = plot_effects({"Zero": zero})
    assert axes[0, 0].get_ylim()[0] < 0 < axes[0, 0].get_ylim()[1]
    _, axes = plot_effects({"Zero": zero, "Nonzero": first}, sharey=False)
    assert axes[0, 0].get_ylim() != axes[0, 1].get_ylim()


def test_summary_dot_positions_match_att_and_rmse():
    first = example(pd.Index([1, 2, 3, 4]))
    second = replace(first, counterfactual=first.observed - [3, 4, -2, -4])
    fig, axes = plot_comparison({"First": first, "Second": second})
    fig.canvas.draw()
    np.testing.assert_allclose(axes[0].collections[0].get_offsets(), [[2, 0], [-3, 1]])
    np.testing.assert_allclose(axes[1].collections[0].get_offsets(), [[0.2, 0], [np.sqrt(12.5), 1]])
    assert axes[0].get_xlim()[0] < 0 < axes[0].get_xlim()[1]
    assert axes[1].get_xlim()[0] == 0
    assert [t.get_text() for t in axes[0].get_yticklabels()] == ["First", "Second"]
    # No uncertainty whiskers or ribbons have been invented.
    assert not axes[0].containers and not axes[1].containers
    assert len(axes[0].collections) == len(axes[1].collections) == 1


@pytest.mark.parametrize("function", [plot_effects, plot_comparison])
@pytest.mark.parametrize("problem", ["empty", "observed", "time", "boundary", "nan", "unfitted"])
def test_invalid_comparisons_rejected_before_a_figure_is_created(function, problem):
    first = example(pd.Index([1, 2, 3, 4]))
    second = deepcopy(first)
    mapping = {"First": first, "Second": second}
    if problem == "empty":
        mapping = {}
    elif problem == "observed":
        second.observed += 1
    elif problem == "time":
        second.observed.index = second.counterfactual.index = pd.Index([2, 3, 4, 5])
    elif problem == "boundary":
        second.n_pre = 3
    elif problem == "nan":
        second.counterfactual.iloc[0] = np.nan
    else:
        mapping["Second"] = SyntheticControl()
    before = plt.get_fignums()
    with pytest.raises((ValueError, TypeError)):
        function(mapping)
    assert plt.get_fignums() == before


def test_existing_estimate_plot_contract(panel):
    fit = SyntheticControl().fit(panel)
    _, supplied = plt.subplots()
    ax = fit.plot(ax=supplied, effects=True, ylabel="Effect (dollars)")
    assert ax is supplied
    assert ax.get_ylabel() == "Effect (dollars)"
    line = next(line for line in ax.lines if line.get_label() == fit.method)
    np.testing.assert_allclose(line.get_ydata(), fit.effects)


def test_plotting_and_export_preserve_rcparams(tmp_path):
    keys = [
        "font.family",
        "font.size",
        "axes.edgecolor",
        "pdf.fonttype",
        "svg.fonttype",
        "figure.dpi",
    ]
    before = {key: deepcopy(mpl.rcParams[key]) for key in keys}
    fit = example(pd.Index([1, 2, 3, 4]))
    ax = plot_paths(fit)
    plot_effects({"Example": fit})
    plot_comparison({"Example": fit})
    save_figure(ax.figure, tmp_path / "test", formats=("svg",))
    assert {key: mpl.rcParams[key] for key in keys} == before


def test_pdf_svg_png_exports_with_physical_size_and_editable_fonts(tmp_path):
    from PIL import Image

    ax = plot_paths(example(pd.Index([1, 2, 3, 4])), figsize=(7.2, 3.8))
    paths = save_figure(ax.figure, tmp_path / "figures" / "reference")
    assert [p.suffix for p in paths] == [".pdf", ".svg", ".png"]
    pdf = paths[0].read_bytes()
    assert pdf.startswith(b"%PDF-") and b"/FontFile2" in pdf
    assert b"/Subtype /Type3" not in pdf
    svg = paths[1].read_text()
    assert "<svg" in svg and "<text" in svg and "Observed" in svg
    with Image.open(paths[2]) as raster:
        assert raster.size == (2160, 1140)
    assert plt.fignum_exists(ax.figure.number)
    assert save_figure(ax.figure, tmp_path / "single.pdf") == (tmp_path / "single.pdf",)


@pytest.mark.parametrize("formats", [(), ("jpg",), ("pdf", "bogus")])
def test_unsupported_exports_fail_without_writing(tmp_path, formats):
    ax = plot_paths(example(pd.Index([1, 2, 3, 4])))
    with pytest.raises(ValueError):
        save_figure(ax.figure, tmp_path / "output", formats=formats)
    assert not list(tmp_path.iterdir())


def test_long_captions_do_not_squeeze_paths_or_clip_exports(tmp_path):
    ax = plot_paths(
        example(pd.Index([1960, 1980, 1990, 2003])),
        title="German reunification",
        subtitle="Observed outcome and reference counterfactual",
        ylabel="GDP per capita (US dollars, source units)",
        counterfactual_label="Historical SCM counterfactual",
        note=(
            "Reference: Abadie, Diamond & Hainmueller (2015). The historical specification "
            "includes 1990 in predictor training. Comparison estimators use only periods "
            "before 1990. This deliberately long caption must fit inside the exported figure."
        ),
    )
    with mpl.rc_context({"savefig.bbox": "tight"}):
        save_figure(ax.figure, tmp_path / "caption")
    assert ax.get_position().width > 0.7
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    bounds = ax.figure.bbox
    for container in ax.figure.axes:
        if not container.axison:
            for text in container.texts:
                text_bounds = text.get_window_extent(renderer)
                assert text_bounds.x0 >= bounds.x0 - 2
                assert text_bounds.x1 <= bounds.x1 + 2
                assert text_bounds.y0 >= bounds.y0 - 2
                assert text_bounds.y1 <= bounds.y1 + 2


def test_export_preserves_vector_canvas(tmp_path):
    from matplotlib.backends.backend_svg import FigureCanvasSVG

    ax = plot_paths(example(pd.Index([1, 2, 3, 4])))
    canvas = FigureCanvasSVG(ax.figure)
    save_figure(ax.figure, tmp_path / "vector.svg")
    assert ax.figure.canvas is canvas
