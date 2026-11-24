import numpy as np
import pandas as pd
import pytest

from pysynth import Panel, SyntheticControl, conformal_confidence_set
from pysynth.conformal_plotting import plot_conformal_set

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture
def grid():
    table = pd.DataFrame(
        {
            "time": [2004, 2004, 2004, 2006, 2006, 2006],
            "candidate": [-1.0, 0.0, 1.0] * 2,
            "p_value": [0.4, 0.1, 0.4, 0.1, 0.1, 0.1],
            "accepted": [True, False, True, False, False, False],
        }
    )
    table.attrs["alpha"] = 0.1
    return table


def test_disconnected_sets_empty_periods_and_aligned_reference(grid):
    fit = SyntheticControl().fit(
        Panel(
            np.column_stack([np.zeros(4), [1, 0, 2, 3]]),
            1,
            2,
            times=pd.Index([2000, 2001, 2004, 2006]),
        )
    )
    bounds = pd.DataFrame({"lower": [4, -1], "upper": [5, 1]}, index=[2006, 2004])
    fig, ax = plot_conformal_set(grid, fit, reference_bounds=bounds)
    try:
        marks, estimates, source = ax.collections
        np.testing.assert_array_equal(marks.get_offsets(), [[2004, -1], [2004, 1]])
        np.testing.assert_array_equal(estimates.get_offsets(), [[2004, 2], [2006, 3]])
        np.testing.assert_array_equal(
            source.get_offsets(),
            [
                [2004, -1],
                [2004, 1],
                [2006, 4],
                [2006, 5],
            ],
        )
        assert len(ax.lines) == 1  # Only the horizontal zero reference.
        assert any(text.get_text() == "Empty" for text in ax.texts)
    finally:
        plt.close(fig)


def test_inversion_level_survives_plot_and_csv_roundtrip_is_neutral():
    p = Panel(np.column_stack([np.zeros(4), [1, 0, 2, 3]]), 1, 3)
    table = conformal_confidence_set(p, [-1, 0, 1], alpha=0.3)
    fig, ax = plot_conformal_set(table)
    try:
        assert ax.legend_.get_texts()[0].get_text() == "70% accepted grid"
        assert ax.get_xlim()[0] < 3 < ax.get_xlim()[1]
    finally:
        plt.close(fig)
    table.attrs.clear()
    fig, ax = plot_conformal_set(table)
    try:
        assert ax.legend_.get_texts()[0].get_text() == "Accepted grid"
    finally:
        plt.close(fig)


def test_invalid_plot_grid_level_and_reference_rejected(grid):
    with pytest.raises(ValueError, match="conflicts"):
        plot_conformal_set(grid, alpha=0.05)
    with pytest.raises(ValueError, match="match p_value"):
        plot_conformal_set(grid.assign(accepted=True))
    with pytest.raises(ValueError, match="unique"):
        plot_conformal_set(pd.concat([grid, grid]))
    with pytest.raises(ValueError, match="post-period"):
        plot_conformal_set(grid, reference_bounds=pd.DataFrame({"lower": [0], "upper": [1]}))
