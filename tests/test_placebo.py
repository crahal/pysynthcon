"""Unit-invariant RMSPE-rank diagnostics, including exact-fit edge cases."""

import numpy as np
import pytest

from pysynth import DifferenceInDifferences, Panel, placebo_test


@pytest.mark.parametrize("scale", [1e-15, 1e-200, 1e200])
def test_placebo_rank_and_rmspe_are_invariant_to_outcome_units(scale):
    rng = np.random.default_rng(82)
    outcomes = rng.normal(size=(12, 5)).cumsum(axis=0)
    outcomes[8:, -1] += 5
    original = placebo_test(
        DifferenceInDifferences(), Panel(outcomes, 4, 8), max_pre_rmse_ratio=1.1
    )
    scaled = placebo_test(
        DifferenceInDifferences(), Panel(outcomes * scale, 4, 8), max_pre_rmse_ratio=1.1
    )
    assert np.isfinite(scaled.treated.pre_rmse)
    assert scaled.treated.pre_rmse / scale == pytest.approx(original.treated.pre_rmse)
    np.testing.assert_allclose(scaled.table.rmspe_ratio, original.table.rmspe_ratio, atol=1e-13)
    np.testing.assert_array_equal(scaled.table.eligible, original.table.eligible)
    assert scaled.p_value == original.p_value


@pytest.mark.parametrize("effect", [0, 3])
def test_exact_zero_pre_fit_convention_and_rank(effect):
    outcomes = np.tile(np.arange(6, dtype=float)[:, None], (1, 4))
    outcomes[4:, -1] += effect
    diagnostic = placebo_test(DifferenceInDifferences(), Panel(outcomes, 3, 4))
    assert diagnostic.treated.pre_rmse == 0
    assert diagnostic.table.iloc[0].rmspe_ratio == (np.inf if effect else 0)
    assert (diagnostic.table.iloc[1:].rmspe_ratio == 0).all()
    assert diagnostic.p_value == (0.25 if effect else 1)
