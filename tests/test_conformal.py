import numpy as np
import pandas as pd
import pytest

from pysynth.classic import SyntheticControl
from pysynth.conformal import conformal_confidence_set, conformal_test
from pysynth.datasets import DATASETS, load_dataset
from pysynth.panel import Panel


def one_donor():
    # Hand-computable residuals after subtracting the time-varying null (2, 4).
    return Panel(
        np.column_stack([np.zeros(5), [0, -1, 2, 5, 8]]),
        1,
        3,
        times=pd.Index(["a", "b", "c", "d", "e"]),
        units=("control", "treated"),
    )


@pytest.mark.parametrize("q", [1, 2, np.inf])
def test_sharp_path_all_cyclic_windows_and_labels(q):
    panel = one_donor()
    path = pd.Series([2, 4], index=["d", "e"])
    result = conformal_test(panel, path, q=q)
    np.testing.assert_array_equal(result.residuals, [0, -1, 2, 3, 4])
    assert result.weights.to_dict() == {"control": 1.0}
    assert result.residuals.index.equals(panel.times)
    assert result.null_effect.to_dict() == path.to_dict()
    pairs = [[3, 4], [4, 0], [0, -1], [-1, 2], [2, 3]]
    np.testing.assert_allclose(result.rotations.statistic, np.linalg.norm(pairs, ord=q, axis=1))
    assert result.rotations.window_start.tolist() == ["d", "e", "a", "b", "c"]
    assert result.rotations.window_end.tolist() == ["e", "a", "b", "c", "d"]
    assert result.rotations.wraps.tolist() == [False, True, False, False, False]
    assert result.statistic == result.rotations.loc[0, "statistic"]
    assert result.p_value == (0.4 if q == np.inf else 0.2)
    np.testing.assert_array_equal(panel.treated, [0, -1, 2, 5, 8])


def test_weights_are_refitted_after_imposing_null_on_entire_panel():
    panel = Panel(np.array([[0, 2, 0], [0, 2, 0], [0, 2, 2], [0, 2, 2]]), 2, 2)
    np.testing.assert_allclose(SyntheticControl().fit(panel).weights, [1, 0], atol=1e-12)
    tested = conformal_test(panel)
    np.testing.assert_allclose(tested.weights, [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(tested.residuals, [-1, -1, 1, 1], atol=1e-12)
    # A sharp effect of two makes donor one fit all four adjusted observations.
    adjusted = conformal_test(panel, 2)
    np.testing.assert_allclose(adjusted.weights, [1, 0], atol=1e-12)
    tied = conformal_test(Panel(np.zeros((4, 2)), 1, 2), 0)
    assert tied.p_value == 1
    assert tied.rotations.at_least_observed.all()


def test_l1_equal_window_sums_count_as_ties_with_different_maxima():
    panel = Panel(np.column_stack([np.zeros(6), [5, 5, 5, 0, 4, 11]]), 1, 3)
    result = conformal_test(panel)
    assert result.statistic == 15
    assert result.rotations.loc[3, "statistic"] == result.statistic
    assert result.p_value == 4 / 6


@pytest.mark.parametrize("weights", [[0.25, 0.75], [0.125, 0.375, 0.25, 0.25]])
@pytest.mark.parametrize("unit_scale,shift", [(1, 0), (2.0**-40, 0), (1, 2.0**30)])
def test_exact_mixture_yields_all_ties_even_with_high_post_variance(weights, unit_scale, shift):
    donors = np.random.RandomState(0).randint(-10, 11, (60, len(weights)))
    donors[-12:] *= 10
    outcomes = np.column_stack([donors, donors @ weights]) * unit_scale + shift
    panel = Panel(outcomes, len(weights), 48)
    result = conformal_test(panel)
    assert result.p_value == 1
    assert result.statistic == 0
    assert conformal_confidence_set(panel, [0]).p_value.eq(1).all()

    # Moving the treated outcomes at a scale relative to donor differences
    # rather than their common level must still produce a real discrepancy.
    outcomes[-12:, -1] += 0.25 * unit_scale
    changed = conformal_test(Panel(outcomes, len(weights), 48))
    assert changed.statistic > 0
    assert changed.p_value < 0.1


def test_pointwise_grid_inversion_refits_each_candidate_in_isolation():
    p = one_donor()
    candidates = [-1, 0, 3, 4, 5, 10]
    result = conformal_confidence_set(p, candidates, alpha=0.25)
    for t, part in result.groupby("time"):
        rows = [0, 1, 2, p.times.get_loc(t)]
        isolated = Panel(p.y[rows], p.n_control, p.n_pre, p.times[rows], p.units)
        expected = [conformal_test(isolated, v).p_value for v in candidates]
        np.testing.assert_array_equal(part.p_value, expected)
        np.testing.assert_array_equal(part.accepted, np.array(expected) > 0.25)
    assert (result.p_value == 0.25).any()
    assert not result.loc[result.p_value == 0.25, "accepted"].any()
    empty = conformal_confidence_set(p, [-1000, -999], alpha=0.25)
    assert not empty.accepted.any()


@pytest.mark.parametrize(
    "null", [True, 1j, [1], [0, np.nan], [[0, 0]], pd.Series([0, 0], index=["e", "d"])]
)
def test_rejects_ill_defined_sharp_paths(null):
    with pytest.raises(ValueError, match="null_effect"):
        conformal_test(one_donor(), null)


@pytest.mark.parametrize("q", [True, 0.9, -np.inf, np.nan, "1"])
def test_rejects_invalid_norm(q):
    with pytest.raises(ValueError, match="q must"):
        conformal_test(one_donor(), q=q)


@pytest.mark.parametrize("grid", [[], [1, 1], [1, 0], [np.inf], [[0, 1]]])
def test_rejects_ill_defined_candidate_grids(grid):
    with pytest.raises(ValueError, match="candidates"):
        conformal_confidence_set(one_donor(), grid)


def test_rejects_non_single_treated_and_invalid_level():
    grouped = Panel(np.zeros((4, 3)), 1, 2)
    with pytest.raises(ValueError, match="one treated"):
        conformal_test(grouped)
    with pytest.raises(ValueError, match="one treated"):
        conformal_confidence_set(grouped, [0])
    for alpha in (True, 0, 1, np.nan):
        with pytest.raises(ValueError, match="alpha"):
            conformal_confidence_set(one_donor(), [0], alpha=alpha)


@pytest.mark.network
def test_rhode_island_sc_tables_and_authors_grid_boundaries():
    # CWZ Table 2 and Table 1, moving-block synthetic-control column.
    panel = DATASETS["rhode_island"].panel(load_dataset("rhode_island"))
    tested = conformal_test(panel)
    assert tested.p_value == 1 / 25
    # Independent scinference 567c688, limSolve 2.0.3 type=2, raw L1 score.
    assert tested.statistic == pytest.approx(1.90746717716567, abs=1e-5)
    n = panel.n_pre
    for h, rank in [(1, 6), (2, 6), (3, 5)]:
        placebo = Panel(panel.y[:n], panel.n_control, n - h, panel.times[:n], panel.units)
        assert conformal_test(placebo).p_value == rank / 19

    # Figure 5's pointwise 90% sets. Check the author's grid endpoints plus
    # their next rejected neighbours. The example executes the entire grid.
    boundaries = [(-26, 70), (-86, -1), (-81, 17), (-116, -33), (-146, -33), (-131, -19)]
    for t, (low, high) in enumerate(boundaries, start=n):
        rows = np.r_[0:n, t]
        single = Panel(panel.y[rows], panel.n_control, n, panel.times[rows], panel.units)
        grid = np.array([low - 1, low, high, high + 1]) / 100
        checks = conformal_confidence_set(single, grid, alpha=0.1)
        assert checks.accepted.tolist() == [False, True, True, False]
