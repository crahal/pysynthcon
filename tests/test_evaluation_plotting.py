"""Plot values, visibility, and export for the common evaluation workflows."""

import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from pysynth import (  # noqa: E402
    DifferenceInDifferences,
    Panel,
    SyntheticControl,
    backtest,
    donor_sensitivity,
    plot_backtest,
    plot_donor_sensitivity,
    save_figure,
)


@pytest.fixture
def comparison_panel():
    rng = np.random.default_rng(183)
    donors = rng.normal(size=(14, 4)).cumsum(axis=0)
    target = donors @ [0.3, 0.7, 0, 0]
    target[11:] += 2
    return Panel(np.column_stack([donors, target]), 4, 11, units=["A", "B", "C", "D", "T"])


def test_validation_plot_shows_pooled_and_each_fold_error(comparison_panel):
    result = backtest(
        comparison_panel,
        {"SCM": SyntheticControl, "DID": DifferenceInDifferences},
        initial_window=5,
        horizon=2,
    )
    before = mpl.rcParams.copy()
    fig, axes = plot_backtest(result, units="dollars")
    table = result.summary.sort_values("rank", kind="stable")
    np.testing.assert_allclose(axes[0].collections[0].get_offsets()[:, 0], table.validation_rmse)
    expected = result.scores.pivot(index="model", columns="fold", values="validation_rmse")
    np.testing.assert_allclose(axes[1].images[0].get_array(), expected.reindex(table.index))
    assert [t.get_text() for t in axes[0].get_yticklabels()] == list(table.index)
    assert axes[0].get_xlim()[0] == 0
    assert "dollars" in axes[0].get_xlabel()
    assert "6 held-out predictions per candidate" in axes[0].get_xlabel()
    annotations = [t for t in axes[0].texts if t.get_text() == "Lowest validation RMSE"]
    assert len(annotations) == 1
    assert annotations[0].xy == (table.validation_rmse.iloc[0], 0)
    assert mpl.rcParams == before
    plt.close(fig)


def test_donor_plot_preserves_every_scenario_and_baseline(comparison_panel):
    result = donor_sensitivity(
        comparison_panel, {"SCM": SyntheticControl, "DID": DifferenceInDifferences}
    )
    fig, axes = plot_donor_sensitivity(result, units="dollars")
    # Each row contributes one range, a scenario scatter and a baseline diamond.
    for row, name in enumerate(result.summary.index):
        expected = result.table.loc[result.table.model == name, "att"]
        np.testing.assert_allclose(axes[0].collections[row * 3 + 1].get_offsets()[:, 0], expected)
        np.testing.assert_allclose(
            axes[0].collections[row * 3 + 2].get_offsets()[0, 0], result.baseline[name].att
        )
    np.testing.assert_allclose(
        [c.get_offsets()[0, 0] for c in axes[1].collections], result.summary.max_abs_change
    )
    for row, name in enumerate(result.summary.index):
        scenario = result.summary.loc[name, "worst_scenario"]
        label = f"Omit {scenario[5:]}"
        annotation = next(t for t in axes[1].texts if t.get_text() == label and t.xy[1] == row)
        assert annotation.xy == (result.summary.loc[name, "max_abs_change"], row)
        selected = result.table.loc[
            (result.table.model == name) & (result.table.scenario == scenario)
        ].iloc[0]
        assert abs(selected.att - result.baseline[name].att) == annotation.xy[0]
    assert axes[0].get_xlim()[0] < min(0, result.table.att.min())
    assert axes[0].get_xlim()[1] > max(0, result.table.att.max())
    footer = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "not confidence intervals" in " ".join(footer.split())
    plt.close(fig)


def test_validation_plot_labels_all_exact_ties(comparison_panel):
    result = backtest(
        comparison_panel,
        {"First": DifferenceInDifferences, "Identical": DifferenceInDifferences},
        initial_window=5,
        horizon=2,
    )
    fig, axes = plot_backtest(result)
    annotations = [t for t in axes[0].texts if "Lowest validation RMSE" in t.get_text()]
    assert len(annotations) == 2
    assert all(t.get_text() == "Lowest validation RMSE (tie)" for t in annotations)
    assert [t.xy[1] for t in annotations] == [0, 1]
    plt.close(fig)


@pytest.mark.parametrize("kind", ["validation", "sensitivity"])
@pytest.mark.parametrize("units", ["outcome units", "PPP current US$ per capita"])
def test_evaluation_figures_export_without_clipping(comparison_panel, tmp_path, kind, units):
    if kind == "validation":
        result = backtest(
            comparison_panel, {"Long method name": SyntheticControl}, initial_window=5, horizon=2
        )
        fig, _ = plot_backtest(result, units=units)
    else:
        result = donor_sensitivity(
            comparison_panel,
            {"Long method name": SyntheticControl},
            exclusions={"Exclude the northern comparison region": ["A"]},
        )
        fig, _ = plot_donor_sensitivity(result, units=units)
        assert any(
            " ".join(t.get_text().split()) == "Exclude the northern comparison region"
            for ax in fig.axes
            for t in ax.texts
        )
    paths = save_figure(fig, tmp_path / kind)
    assert {p.suffix for p in paths} == {".svg", ".pdf", ".png"}
    assert all(p.stat().st_size > 1000 for p in paths)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        artists = [*ax.texts, ax.title, ax._left_title]
        if ax.axison:
            artists += [ax.xaxis.label]
            for positions, ticks, limits in [
                (ax.get_xticks(), ax.get_xticklabels(), sorted(ax.get_xlim())),
                (ax.get_yticks(), ax.get_yticklabels(), sorted(ax.get_ylim())),
            ]:
                # Locators may propose ticks outside the visible view interval;
                # Matplotlib does not draw those artists.
                artists += [t for p, t in zip(positions, ticks) if limits[0] <= p <= limits[1]]
        for artist in artists:
            if artist.get_visible() and artist.get_text():
                bounds = artist.get_window_extent(renderer)
                assert bounds.x0 >= -1 and bounds.x1 <= fig.bbox.width + 1
                assert bounds.y0 >= -1 and bounds.y1 <= fig.bbox.height + 1
    plt.close(fig)


@pytest.mark.parametrize("plot", [plot_backtest, plot_donor_sensitivity])
def test_evaluation_plots_reject_unrelated_results(plot):
    with pytest.raises(TypeError):
        plot({})
