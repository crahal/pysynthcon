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


def test_missing_placebo_prediction_cannot_count_as_an_exact_fit(panel):
    class DropsLastPlaceboPrediction:
        def fit(self, supplied):
            fit = DifferenceInDifferences().fit(supplied)
            if supplied.units[-1] != panel.units[-1]:
                fit.counterfactual = fit.counterfactual.iloc[:-1]
            return fit

    with pytest.raises(ValueError, match="counterfactual labels"):
        placebo_test(DropsLastPlaceboPrediction(), panel)


@pytest.mark.parametrize(
    "pre, pseudo_post, treated_post",
    [
        (1e-300, 1e280, 1e289),  # both displayed quotients overflow
        (1e300, 1e-280, 1e-270),  # both displayed quotients underflow
    ],
)
def test_extreme_rmspe_ratios_do_not_turn_distinct_fits_into_ties(pre, pseudo_post, treated_post):
    from pysynth.results import make_estimate

    panel = Panel(np.zeros((4, 4)), 3, 2)

    class ExtremeFit:
        def fit(self, sample):
            post = treated_post if sample.units[-1] == panel.units[-1] else pseudo_post
            return make_estimate(
                sample, "extreme", np.zeros(sample.n_control), counterfactual=[pre, pre, post, post]
            )

    result = placebo_test(ExtremeFit(), panel)
    assert result.table.rmspe_ratio.nunique() == 1
    assert result.table.at_least_treated_ratio.tolist() == [True, False, False, False]
    assert result.p_value == 0.25
