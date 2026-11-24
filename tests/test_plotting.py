"""Figure correctness: samples, coordinates, scales, summaries and real exports."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pysynth import (
    Estimate,
    LiteratureBenchmark,
    ReferenceComparison,
    SyntheticControl,
    plot_comparison,
    plot_effects,
    plot_paths,
    publication_style,
    publication_title_font,
    reference_comparison_table,
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


def test_publication_fonts_resolve_to_packaged_faces():
    from pathlib import Path

    from matplotlib.font_manager import FontProperties, findfont

    with publication_style():
        folder = Path(publication_title_font().get_file()).parent
        for family, weight, filename in [
            ("Source Sans 3", "normal", "SourceSans3-Regular.ttf"),
            ("Source Sans 3", "semibold", "SourceSans3-Semibold.ttf"),
            ("STIX Two Text", "semibold", "STIXTwoText-SemiBold.ttf"),
        ]:
            prop = FontProperties(family=family, weight=weight)
            assert Path(findfont(prop)).samefile(folder / filename)


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
    import re
    from base64 import b64decode
    from io import BytesIO

    from fontTools.ttLib import TTFont
    from PIL import Image

    ax = plot_paths(example(pd.Index([1, 2, 3, 4])), figsize=(7.2, 3.8))
    paths = save_figure(ax.figure, tmp_path / "figures" / "reference")
    assert [p.suffix for p in paths] == [".pdf", ".svg", ".png"]
    pdf = paths[0].read_bytes()
    assert pdf.startswith(b"%PDF-") and b"/FontFile2" in pdf
    assert b"/Subtype /Type3" not in pdf
    svg = paths[1].read_text()
    assert "<svg" in svg and "<text" in svg and "Observed" in svg
    assert b"SourceSans3-Regular" in pdf and b"STIXTwoText-SemiBold" in pdf
    assert "SIL OPEN FONT LICENSE" in svg
    faces = set()
    for data in re.findall(r"data:font/woff;base64,([A-Za-z0-9+/=]+)", svg):
        with TTFont(BytesIO(b64decode(data))) as font:
            faces.add((font["name"].getDebugName(1), font["OS/2"].usWeightClass))
            assert font["name"].getDebugName(0) and font["name"].getDebugName(13)
            assert ord("e") in font.getBestCmap()
    assert faces == {
        ("PySynthCon Figure Sans", 400),
        ("PySynthCon Figure Sans", 600),
        ("PySynthCon Figure Serif", 600),
    }
    with Image.open(paths[2]) as raster:
        assert raster.size == (2160, 1140)
    assert plt.fignum_exists(ax.figure.number)
    assert save_figure(ax.figure, tmp_path / "single.pdf") == (tmp_path / "single.pdf",)


def test_svg_preserves_custom_type_and_literal_font_names(tmp_path):
    fig = plt.figure()
    label = "Compare 'Source Sans 3' and 'STIX Two Text'."
    fig.text(0.1, 0.5, label, family="DejaVu Serif")
    (path,) = save_figure(fig, tmp_path / "custom.svg")
    svg = path.read_text()
    assert label in svg and "'DejaVu Serif'" in svg
    assert "data:font/woff" not in svg


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


def literature(**kwargs):
    return LiteratureBenchmark(
        headline="Example SCM mean effect: −20 dollars",
        citation="Example authors (2021), Table 1",
        source_url="https://example.org/paper",
        context="Same units and post-period; different specifications may give different effects.",
        **kwargs,
    )


def test_numeric_benchmark_is_only_on_effect_and_att_axes():
    fit = example(pd.Index([2000, 2001, 2004, 2010]))
    benchmark = literature(value=-20, window=(2004, 2010))
    ax = plot_paths(fit, benchmark=benchmark)
    assert not any(line.get_label() == "Literature mean ATT" for line in ax.lines)
    _, axes = plot_effects({"One": fit, "Two": fit}, benchmark=benchmark)
    for ax in axes.flat:
        line = next(line for line in ax.lines if line.get_label() == "Literature mean ATT")
        np.testing.assert_array_equal(line.get_xdata(), [2004, 2010])
        np.testing.assert_array_equal(line.get_ydata(), [-20, -20])
        assert ax.get_ylim()[0] < -20
    _, axes = plot_comparison({"One": fit}, benchmark=benchmark)
    line = next(line for line in axes[0].lines if line.get_label() == "Literature mean ATT")
    np.testing.assert_array_equal(line.get_xdata(), [-20, -20])
    assert axes[0].get_xlim()[0] < -20
    assert not any(line.get_label() == "Literature mean ATT" for line in axes[1].lines)
    # Adding a published guide must not change our fitted results or summary dots.
    np.testing.assert_allclose(axes[0].collections[0].get_offsets(), [[fit.att, 0]])


@pytest.mark.parametrize("function", [plot_paths, plot_effects, plot_comparison])
def test_different_benchmark_window_rejected_before_plotting(function):
    fit = example(pd.Index([2000, 2001, 2004, 2010]))
    benchmark = literature(value=-20, window=(2001, 2010))
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="window must match"):
        function(fit if function is plot_paths else {"One": fit}, benchmark=benchmark)
    assert plt.get_fignums() == before


def test_text_only_benchmark_and_source_survive_export(tmp_path):
    fit = example(pd.Index([2000, 2001, 2004, 2010]))
    benchmark = literature()
    fig, axes = plot_comparison(
        {"One": fit, "Two": fit}, benchmark=benchmark, reference_style="strip"
    )
    assert not any(line.get_label() == "Literature mean ATT" for ax in axes for line in ax.lines)
    paths = save_figure(fig, tmp_path / "literature")
    svg = paths[1].read_text()
    assert benchmark.source_url in svg
    assert "Published result" in svg and "Table 1" in svg
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    strip, artists, _ = fig._pysynth_benchmark_layout
    bounds = strip.get_window_extent(renderer)
    previous_bottom = bounds.y1
    for artist in artists:
        text = artist.get_window_extent(renderer)
        assert bounds.x0 <= text.x0 < text.x1 <= bounds.x1
        assert bounds.y0 - 2 <= text.y0 < text.y1 <= previous_bottom + 2
        previous_bottom = text.y0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"value": float("nan"), "window": (2004, 2010)},
        {"value": -1},
        {"window": (2004, 2010)},
    ],
)
def test_invalid_benchmark_metadata(kwargs):
    with pytest.raises(ValueError):
        literature(**kwargs)


def test_calendar_benchmark_uses_calendar_coordinates():
    from matplotlib.dates import date2num

    times = pd.date_range("2000-01-01", periods=4, freq="MS")
    fit = example(times)
    benchmark = literature(value=-20, window=(times[2], times[3]))
    ax = plot_paths(fit, effects=True, benchmark=benchmark)
    line = next(line for line in ax.lines if line.get_label() == "Literature mean ATT")
    np.testing.assert_allclose(line.get_xdata(), date2num(times[2:].to_pydatetime()))


def matched_reference(fit, **kwargs):
    fields = dict(
        label="Historical SCM",
        estimate=fit,
        reference=1.8,
        window=(2004, 2010),
        units="dollars per person",
        citation="Authors (2021), Table 1",
        source_url="https://example.org/paper",
        context="Same historical specification.",
    )
    fields.update(kwargs)
    return ReferenceComparison(**fields)


def test_reference_values_use_source_window_and_percentage_denominator():
    fit = example(pd.Index([2000, 2001, 2004, 2010]))
    whole = matched_reference(fit)
    subset = matched_reference(fit, window=(2010, 2010))
    percent = matched_reference(fit, statistic="mean_percent_effect", reference=30)
    table = reference_comparison_table([whole, subset, percent])
    np.testing.assert_allclose(table.reproduced, [2, 3, (100 / 5 + 300 / 7) / 2])
    np.testing.assert_allclose(table.difference, table.reproduced - table.reference)
    assert table.loc[2, "difference_units"] == "percentage points"
    # Averaging ratios is deliberately different from dividing aggregate means.
    assert percent.reproduced != 100 * fit.att / fit.observed.iloc[2:].mean()
    assert table.loc[1, "window_start"] == 2010


def test_matched_guides_do_not_spread_to_other_methods_or_incompatible_axes():
    fit = example(pd.Index([2000, 2001, 2004, 2010]))
    other = replace(fit, method="Another specification")
    ref = matched_reference(fit)
    _, axes = plot_effects({"Historical SCM": fit, "Other": other}, references=[ref])
    matching = [
        line for line in axes[0, 0].lines if line.get_label() == "Published: Historical SCM"
    ]
    assert len(matching) == 1
    np.testing.assert_allclose(matching[0].get_ydata(), [1.8, 1.8])
    assert not any(line.get_label().startswith("Published:") for line in axes[0, 1].lines)
    _, axes = plot_comparison({"Historical SCM": fit, "Other": other}, references=[ref])
    assert len(axes[0].collections) == 2
    np.testing.assert_allclose(axes[0].collections[1].get_offsets(), [[1.8, 0]])
    assert len(axes[1].collections) == 1
    ax = plot_paths(fit, references=[ref])
    assert not any(line.get_label().startswith("Published:") for line in ax.lines)


def test_subset_and_percentage_references_never_become_full_period_att_markers():
    fit = example(pd.Index([2000, 2001, 2004, 2010]))
    subset = matched_reference(fit, window=(2010, 2010))
    percent = matched_reference(fit, statistic="mean_percent_effect", reference=30)
    fig, axes = plot_comparison({"Historical SCM": fit}, references=[subset, percent])
    assert len(axes[0].collections) == len(axes[1].collections) == 1
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panel_bottom = min(ax.get_tightbbox(renderer).y0 for ax in axes)
    for text, _ in fig._pysynth_comparison_notes:
        box = text.get_window_extent(renderer)
        assert fig.bbox.y0 <= box.y0 < box.y1 < panel_bottom
    ax = plot_paths(fit, effects=True, references=[subset, percent])
    guides = [line for line in ax.lines if line.get_label().startswith("Published:")]
    assert len(guides) == 1
    np.testing.assert_allclose(guides[0].get_xdata(), [2010, 2010])


@pytest.mark.parametrize("function", [plot_paths, plot_effects, plot_comparison])
def test_reference_comparisons_reject_a_different_observed_case_before_plotting(function):
    fit = example(pd.Index([2000, 2001, 2004, 2010]))
    different = replace(fit, observed=fit.observed + 1)
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        function(
            fit if function is plot_paths else {"SCM": fit},
            references=[matched_reference(different)],
        )
    assert plt.get_fignums() == before


@pytest.mark.parametrize("window", [(2001, 2010), (2004, 2008), (2010, 2004)])
def test_reference_windows_must_be_observed_ordered_post_periods(window):
    with pytest.raises(ValueError, match="reference window"):
        matched_reference(example(pd.Index([2000, 2001, 2004, 2010])), window=window)


def test_paired_reference_strip_exports_signed_differences_without_overlaps(tmp_path):
    fit = example(pd.Index([2000, 2001, 2004, 2010]))
    refs = [
        matched_reference(fit, label="Historical synthetic control with covariates"),
        matched_reference(
            fit,
            reference=30,
            approximate=True,
            statistic="mean_percent_effect",
            units="% of observed GDP",
            precision=2,
            reference_precision=0,
        ),
        matched_reference(
            fit,
            reference=2.00001,
            precision=3,
            reference_precision=5,
            source_label="Upstream software",
        ),
    ]
    fig, _ = plot_comparison({"SCM": fit}, references=refs, reference_style="strip")
    paths = save_figure(fig, tmp_path / "paired")
    svg = paths[1].read_text()
    for label in [
        "REFERENCE CHECK",
        "Reproduced",
        "+0.200",
        "≈ 30",
        " pp",
        "|Δ| &lt; 0.001",
        "https://example.org/paper",
    ]:
        assert label in svg
    assert "−0.000" not in svg and "+0.000" not in svg
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    strip, heading, headings, rows = fig._pysynth_reference_layout
    bounds = strip.get_window_extent(renderer)
    artists = [heading, *headings]
    for label, _, values, source, _ in rows:
        artists.extend([label, *values, source])
        assert label.get_window_extent(renderer).x1 < values[0].get_window_extent(renderer).x0
        for previous, following in zip(values, values[1:]):
            assert (
                previous.get_window_extent(renderer).x1 < following.get_window_extent(renderer).x0
            )
    for artist in artists:
        text = artist.get_window_extent(renderer)
        assert bounds.x0 - 2 <= text.x0 < text.x1 <= bounds.x1 + 2
        assert bounds.y0 - 2 <= text.y0 < text.y1 <= bounds.y1 + 2
    for previous, following in zip(rows, rows[1:]):
        assert (
            previous[3].get_window_extent(renderer).y0 > following[0].get_window_extent(renderer).y1
        )


def test_shared_effect_label_is_centered_on_panels_after_reference_strip_and_resize(tmp_path):
    fit = example(pd.Index([2000, 2001, 2004, 2010]))
    refs = [matched_reference(fit) for _ in range(3)]
    fig, axes = plot_effects(
        {"SCM": fit, "Another": replace(fit)}, references=refs, reference_style="strip"
    )
    for height in (fig.get_figheight(), fig.get_figheight() + 1):
        fig.set_figheight(height)
        save_figure(fig, tmp_path / "label.svg")
        positions = [ax.get_position() for ax in axes.flat if ax.get_visible()]
        expected = (min(box.y0 for box in positions) + max(box.y1 for box in positions)) / 2
        assert fig._supylabel.get_position()[1] == pytest.approx(expected, abs=0.002)


@pytest.mark.parametrize("function", [plot_paths, plot_effects, plot_comparison])
def test_direct_reference_annotations_export_links_without_top_strip(function, tmp_path):
    fit = example(pd.Index([2000, 2001, 2004, 2010]))
    ref = matched_reference(fit)
    output = function(fit if function is plot_paths else {"SCM": fit}, references=[ref])
    fig = output.figure if function is plot_paths else output[0]
    assert not hasattr(fig, "_pysynth_reference_layout")
    save_figure(fig, tmp_path / f"{function.__name__}.svg")
    svg = (tmp_path / f"{function.__name__}.svg").read_text()
    assert "Paper 1.8" in svg and "+0.200" in svg
    assert ref.source_url in svg
    assert "REFERENCE CHECK" not in svg
    assert "Δ = fit" in svg


def test_unplotted_historical_fit_gets_its_own_row_without_changing_input():
    fit = example(pd.Index([2000, 2001, 2004, 2010]))
    historical = replace(fit, counterfactual=fit.observed - [1, 1, -4, -8])
    ref = matched_reference(historical, reference=-6)
    mapping = {"Comparison SCM": fit}
    _, axes = plot_comparison(mapping, references=[ref])
    assert list(mapping) == ["Comparison SCM"]
    assert [t.get_text() for t in axes[0].get_yticklabels()] == [
        "Comparison SCM",
        "Reference · Historical SCM",
    ]
    np.testing.assert_allclose(axes[0].collections[0].get_offsets(), [[2, 0], [-6, 1]])
    np.testing.assert_allclose(axes[0].collections[1].get_offsets(), [[-6, 1]])
    assert not any("Paper" in text.get_text() and text.xy[1] == 0 for text in axes[0].texts)


def test_unplotted_reference_panel_includes_historical_effects_in_shared_limits():
    fit = example(pd.Index([2000, 2001, 2004, 2010]))
    historical = replace(fit, counterfactual=fit.observed - [0, 0, -20, -30])
    ref = matched_reference(historical, reference=-25)
    fig, axes = plot_effects({"Comparison": fit}, references=[ref])
    visible = [ax for ax in axes.flat if ax.get_visible()]
    assert len(visible) == 2
    assert axes.shape == (1, 2)
    assert "Reference · Historical SCM" in visible[1].get_title(loc="left").replace("\n", " ")
    assert visible[0].get_ylim() == visible[1].get_ylim()
    assert visible[0].get_ylim()[0] < -30
    guide = next(
        line for line in visible[1].lines if line.get_label() == "Published: Historical SCM"
    )
    np.testing.assert_array_equal(guide.get_ydata(), [-25, -25])
    assert not any(line.get_label().startswith("Published:") for line in visible[0].lines)


def test_mean_callout_points_to_mean_not_final_effect_and_path_ticks_remain_observed():
    fit = example(pd.Index([2000, 2001, 2004, 2010]))
    fig, axes = plot_effects({"SCM": fit}, colors={"SCM": "red"})
    callout = fig._pysynth_direct_annotations[0]
    assert callout.xy[1] == fit.att
    assert callout.xy[1] != fit.effects.iloc[-1]
    assert callout.get_color() == "red"
    guide = next(line for line in axes[0, 0].lines if line.get_label() == "Fitted mean ATT")
    np.testing.assert_array_equal(guide.get_xdata(), [2004, 2010])
    assert guide.get_color() == "red"
    ax = plot_paths(fit, color="red", intervention_label="Treatment")
    assert ax.get_legend() is None
    assert all(2000 <= tick <= 2010 for tick in ax.get_xticks())
    assert {"Observed", "Synthetic control"}.issubset({t.get_text() for t in ax.texts})
    assert next(t for t in ax.texts if t.get_text() == "Synthetic control").get_color() == "red"
    onset = next(t for t in ax.texts if t.get_text().strip() == "Treatment")
    assert onset.get_position() == (2004, 1.02)
    assert onset.get_va() == "bottom"


@pytest.mark.parametrize("function", [plot_paths, plot_effects, plot_comparison])
def test_invalid_reference_layout_is_rejected_before_creating_figure(function):
    fit = example(pd.Index([2000, 2001, 2004, 2010]))
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="reference_style"):
        function(fit if function is plot_paths else {"SCM": fit}, reference_style="hidden")
    assert plt.get_fignums() == before


@pytest.fixture
def basque_reference_fits():
    from pysynth.replication import reference_comparisons

    # Reproduce the notebook's eleven estimators and its two source windows
    # using an in-memory series. All panels share the observed GDP trajectory.
    years = pd.Index(range(1960, 1998), name="Year")
    observed = pd.Series(np.linspace(5, 9, len(years)), index=years)
    gap = np.where(years < 1970, 0.01, -0.6995627075801111)
    historical = Estimate(
        "Historical SCM", observed, observed - gap, pd.Series([1.0], index=["donor"]), n_pre=10
    )
    names = (
        "Outcome SCM",
        "Predictor SCM · 3 blocks",
        "Ridge-augmented SCM",
        "Penalized SCM · λ = 0.01",
        "Robust SCM · rank 3",
        "Differenced SCM",
        "SCUL",
        "ISC · full donor pool",
        "2SISC · nearest 10",
        "Synthetic DID",
        "DID",
    )
    fits = {
        name: replace(historical, counterfactual=observed - gap * (1 + i / 20))
        for i, name in enumerate(names)
    }
    return fits, historical, reference_comparisons("basque", historical)


def assert_basque_source_links(path, headline, numerical):
    from xml.etree import ElementTree

    # Check the exported callouts themselves retain the correct source
    # links and windows after the renderer reflows the text for this size.
    links = ElementTree.parse(path).findall(".//{http://www.w3.org/2000/svg}a")
    for reference, label, window in (
        (headline, "Fig. 1 headline", "1980–1997"),
        (numerical, "R Synth", "1975–1997"),
    ):
        text = " ".join(
            word
            for link in links
            if link.get("{http://www.w3.org/1999/xlink}href") == reference.source_url
            for word in " ".join(link.itertext()).split()
        )
        assert label in text and window in text


def test_basque_historical_effects_and_citations_stay_separate_in_export(
    tmp_path, basque_reference_fits
):
    fits, historical, (headline, numerical) = basque_reference_fits
    fig, axes = plot_effects(fits, references=[headline, numerical])
    assert axes.ndim == 2 and axes.shape[1] == 3
    visible = [ax for ax in axes.flat if ax.get_visible()]
    names = [*fits, "Reference · Historical SCM"]
    assert len(visible) == len(names)
    for ax, name in zip(visible, names):
        assert name in " ".join(ax.get_title(loc="left").split())
    historical_ax = visible[-1]
    curve = next(line for line in historical_ax.lines if line.get_label() == names[-1])
    np.testing.assert_allclose(curve.get_ydata(), historical.effects)

    notes = {a.get_url(): a for a in fig._pysynth_direct_annotations if a.get_url()}
    descriptive = notes[headline.source_url]
    numeric = notes[numerical.source_url]
    for width in (7.2, 6.0):
        fig.set_figwidth(width)
        (path,) = save_figure(fig, tmp_path / f"basque-{width}.svg")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [
            note.get_bbox_patch().get_window_extent(renderer) for note in (descriptive, numeric)
        ]
        assert not boxes[0].overlaps(boxes[1])
        for note, box in zip((descriptive, numeric), boxes):
            for bounds in (fig.bbox, note.axes.get_window_extent(renderer)):
                assert bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)
            assert all(not box.overlaps(ax.get_tightbbox(renderer)) for ax in visible)

        assert descriptive.arrow_patch is None
        assert numerical.window[0] <= numeric.xy[0] <= numerical.window[1]
        assert numeric.xy[1] == numerical.reference
        guide_point = historical_ax.transData.transform(numeric.xy)
        leader = numeric.arrow_patch
        leader_tip = leader.get_path().transformed(leader.get_transform()).vertices[-1]
        assert np.linalg.norm(guide_point - leader_tip) <= renderer.points_to_pixels(4)

        assert_basque_source_links(path, headline, numerical)


def test_basque_dot_plot_preserves_row_spacing_and_linked_comparisons(
    tmp_path, basque_reference_fits
):
    fits, historical, (headline, numerical) = basque_reference_fits
    name = "Reference · Historical SCM"
    plain, plain_axes = plot_comparison({**fits, name: historical})
    margin = plain_axes[0].get_ylim()[0] - plain_axes[0].get_yticks()[-1]
    plt.close(plain)
    fig, axes = plot_comparison(fits, references=[headline, numerical])
    labels = [" ".join(t.get_text().split()) for t in axes[0].get_yticklabels()]
    assert labels[:-1] == list(fits)
    assert name in labels[-1]
    assert len(axes) == 2

    for width in (7.2, 6.0):
        fig.set_figwidth(width)
        (path,) = save_figure(fig, tmp_path / f"dot-plot-{width}.svg")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        row_centers = []
        for ax in axes:
            offsets = ax.collections[0].get_offsets()
            assert len(offsets) == len(fits) + 1
            row_centers.append(ax.transData.transform(offsets)[:, 1])
            steps = np.diff(row_centers[-1])
            np.testing.assert_allclose(steps, steps[0])
            assert ax.get_ylim()[0] - offsets[-1, 1] == pytest.approx(margin)
        np.testing.assert_allclose(*row_centers, atol=0.5)
        assert all(key in labels[-1] for key in ("1", "2"))

        notes = {artist.get_url(): artist for artist, _ in fig._pysynth_comparison_notes}
        notes = [notes[ref.source_url] for ref in (headline, numerical)]
        assert "% of observed GDP" in " ".join(notes[0].get_text().split())
        boxes = [artist.get_window_extent(renderer) for artist in notes]
        assert not boxes[0].overlaps(boxes[1])
        caption_boxes = [
            t.get_window_extent(renderer) for t in notes[0].axes.texts if t not in notes
        ]
        for box in boxes:
            assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)
            assert all(not box.overlaps(ax.get_tightbbox(renderer)) for ax in axes)
            assert all(not box.overlaps(caption) for caption in caption_boxes)
        assert_basque_source_links(path, headline, numerical)


def test_dense_historical_annotations_use_empty_facet_space_and_keep_both_source_windows():
    fit = example(pd.Index([2000, 2001, 2004, 2010]))
    historical = replace(fit, counterfactual=fit.observed - [0, 0, -4, -8])
    refs = [
        matched_reference(
            historical,
            statistic="mean_percent_effect",
            reference=-100,
            units="% of observed GDP",
            approximate=True,
        ),
        matched_reference(
            historical, window=(2010, 2010), reference=-8, source_label="Original R software"
        ),
    ]
    fits = {f"Model {i}": fit for i in range(12)}
    fig, axes = plot_effects(fits, references=refs)
    historical_panel = axes[-1, 0].get_position()
    source_gutter = fig._pysynth_direct_annotations[-1].axes.get_position()
    assert historical_panel.width > axes[0, 0].get_position().width
    assert source_gutter.x1 - historical_panel.x0 > 2.5 * axes[0, 0].get_position().width
    assert not axes[-1, 1].get_visible() and not axes[-1, 2].get_visible()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Leave the outer padding as well as the text inside its source gutter.
    for artist in fig._pysynth_direct_annotations[-2:]:
        box = artist.get_bbox_patch().get_window_extent(renderer)
        bounds = artist.axes.get_window_extent(renderer)
        assert bounds.x0 <= box.x0 < box.x1 <= bounds.x1
        assert bounds.y0 <= box.y0 < box.y1 <= bounds.y1
    # Distinct checks in the same source may share a numbered citation and URL.
    fig, _ = plot_comparison(fits, references=refs)
    text = " ".join(
        artist.get_text()
        for artist, _ in fig._pysynth_comparison_notes
        if artist.get_url() == refs[0].source_url
    )
    for detail in ("2004–2010", "2010–2010", "% of observed GDP"):
        assert detail in " ".join(text.split())


def test_unpaired_benchmark_leaders_identify_the_source_guide():
    fit = example(pd.Index([2000, 2001, 2004, 2010]))
    benchmark = literature(value=-20, window=(2004, 2010))
    ax = plot_paths(fit, benchmark=benchmark)
    assert ax.figure._pysynth_direct_annotations[-1].arrow_patch is None

    fig, _ = plot_effects({"One": fit, "Two": fit}, benchmark=benchmark)
    callouts = [a for a in fig._pysynth_direct_annotations if a.get_url()]
    assert len(callouts) == 1
    assert callouts[0].xy == (2007, -20)
    fig, _ = plot_comparison({"One": fit, "Two": fit}, benchmark=benchmark)
    assert fig._pysynth_direct_annotations[0].xy[0] == -20
    fig, _ = plot_comparison({"One": fit}, benchmark=literature())
    assert fig._pysynth_direct_annotations[0].arrow_patch is None


def test_single_path_cites_only_sources_for_its_plotted_fit():
    fit = example(pd.Index([2000, 2001, 2004, 2010]))
    reference = matched_reference(replace(fit), citation="Unplotted study")
    ax = plot_paths(fit, references=[reference])
    text = [t.get_text() for axis in ax.figure.axes for t in axis.texts]
    assert all("Unplotted study" not in value and "fit − source" not in value for value in text)


def test_extreme_source_percentage_and_level_means_are_finite():
    fit = example(pd.Index([2000, 2001, 2004, 2010]))
    observed = pd.Series(1.7e308, index=fit.observed.index)
    fit = replace(fit, observed=observed, counterfactual=0 * observed)
    level = matched_reference(fit)
    percentage = matched_reference(fit, statistic="mean_percent_effect")
    assert level.reproduced == 1.7e308
    assert percentage.reproduced == 100


def test_descriptive_headline_is_unpaired_in_both_layouts():
    fit = example(pd.Index([2000, 2001, 2004, 2010]))
    ref = matched_reference(fit, comparable=False, approximate=True, source_label="Paper headline")
    direct, axes = plot_effects({"Model": fit}, references=[ref])
    callout = direct._pysynth_direct_annotations[0]
    assert "Paper headline" in callout.get_text() and "mean" in callout.get_text()
    assert "Δ" not in callout.get_text()
    assert not any(line.get_label().startswith("Paper headline:") for line in axes[0, 0].lines)
    strip, _ = plot_comparison({"Model": fit}, references=[ref], reference_style="strip")
    assert any(artist.get_text() == "—" for artist in strip._pysynth_reference_layout[0].texts)
    assert ref.difference is None


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (2.00001, "|Δ| < 0.001"),
        (1.99999, "|Δ| < 0.001"),
        (2.0, "Δ 0.000"),
        (1.8, "Δ +0.200"),
        (2.2, "Δ −0.200"),
    ],
)
def test_direct_reference_labels_and_csv_preserve_small_discrepancies(source, expected):
    fit = example(pd.Index([2000, 2001, 2004, 2010]))
    ref = matched_reference(fit, reference=source, precision=3)
    _, axes = plot_comparison({"SCM": fit}, references=[ref])
    assert any(expected in artist.get_text() for artist in axes[0].texts)
    table = reference_comparison_table([ref])
    assert table.at[0, "difference_label"] == expected
    assert table.at[0, "difference"] == 2.0 - source
