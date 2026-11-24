"""Analytic checks of the outcome-only simplex objective."""

import numpy as np
import pytest

from pysynth import Panel, SyntheticControl


@pytest.mark.parametrize("scale,shift", [(1.0, 0.0), (1e-14, 0.0), (1.0, 1e8)])
def test_simplex_fit_is_invariant_to_units_and_common_time_path(scale, shift):
    donors = np.array([[0, 2], [1, 3], [2, 4], [4, 6]], dtype=float)
    treated = donors @ [0.8, 0.2]
    treated[-1] += 3
    common = shift + np.arange(4) ** 2
    panel = Panel((np.column_stack([donors, treated]) + common[:, None]) * scale, 2, 3)
    fit = SyntheticControl().fit(panel)
    np.testing.assert_allclose(fit.weights, [0.8, 0.2], atol=1e-8)
    assert fit.att / scale == pytest.approx(3, abs=1e-7)


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
def test_ridge_has_the_documented_original_unit_penalty(scale):
    # min_b (2b-.4)^2 + 2*((1-b)^2+b^2) has b=.35.
    y = np.tile([0.0, 2.0, 0.4], (5, 1)) * scale
    fit = SyntheticControl(ridge=2 * scale**2).fit(Panel(y, 2, 4))
    np.testing.assert_allclose(fit.weights, [0.65, 0.35], atol=1e-10)
    expected = ((2 * 0.35 - 0.4) ** 2 + 2 * (0.65**2 + 0.35**2)) * scale**2
    assert fit.metadata["objective"] == pytest.approx(expected)
    assert fit.metadata["scaled_optimality_gap"] < 1e-10


def test_identical_pre_outcomes_use_feasible_uniform_solution():
    panel = Panel(np.tile(np.arange(6)[:, None], (1, 4)), 3, 4)
    fit = SyntheticControl().fit(panel)
    np.testing.assert_allclose(fit.weights, np.ones(3) / 3)
    assert fit.pre_rmse == 0


def test_success_flag_does_not_override_infeasible_weights(monkeypatch):
    from scipy.optimize import OptimizeResult

    monkeypatch.setattr(
        "pysynth._optim.minimize",
        lambda *args, **kwargs: OptimizeResult(x=np.array([0.8, 0.8]), success=True),
    )
    panel = Panel(np.tile([0.0, 2.0, 0.4], (5, 1)), 2, 4)
    with pytest.raises(RuntimeError, match="infeasible"):
        SyntheticControl().fit(panel)


def test_convex_gap_bounds_objective_error_independently_of_success(monkeypatch):
    from scipy.optimize import OptimizeResult

    monkeypatch.setattr(
        "pysynth._optim.minimize",
        lambda *args, **kwargs: OptimizeResult(x=np.array([0.5, 0.5]), success=True, nit=0),
    )
    panel = Panel(np.tile([0.0, 2.0, 0.4], (5, 1)), 2, 4)
    fit = SyntheticControl().fit(panel)
    # The exact optimum has zero error at weights (.8, .2).
    assert 0 < fit.metadata["objective"] <= fit.metadata["optimality_gap"]
    assert fit.metadata["optimality_gap"] == pytest.approx(1.2)
