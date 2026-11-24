"""A numerical comparison must be defined over the promised estimation sample."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pysynth import DifferenceInDifferences, compare


def test_missing_observation_cannot_change_att_window_silently(panel):
    fit = DifferenceInDifferences().fit(panel)
    shortened = replace(fit, counterfactual=fit.counterfactual.iloc[:-1])
    with pytest.raises(ValueError, match="counterfactual labels"):
        compare({"shortened": shortened})
    invalid = fit.counterfactual.copy()
    invalid.iloc[-1] = np.nan
    with pytest.raises(ValueError, match="counterfactual must be finite"):
        _ = replace(fit, counterfactual=invalid).att


def test_prefitted_result_must_match_explicit_panel(panel):
    original = DifferenceInDifferences().fit(panel)
    other = replace(panel, y=panel.y + 1)
    with pytest.raises(ValueError, match="observed outcomes"):
        compare({"historical": original}, other)


def test_integer_paths_subtract_before_conversion_and_without_wraparound(panel):
    fit = DifferenceInDifferences().fit(panel)
    fit.observed = pd.Series([2**63 - 1] * 2 + [2**53 + 1])
    fit.counterfactual = pd.Series([-(2**63), 2**63 - 2, 2**53])
    fit.n_pre = 2
    np.testing.assert_array_equal(fit.effects, [float(2**64 - 1), 1.0, 1.0])
    assert fit.att == 1.0


def test_labelled_subset_of_eligible_donors_is_valid_in_comparison(panel):
    fit = DifferenceInDifferences().fit(panel)
    # A custom single-donor estimator can omit its zero weights.
    fit.weights = pd.Series([1.0], index=[panel.units[2]])
    fit.counterfactual = pd.Series(panel.control[:, 2], index=panel.times)
    table = compare({"one": fit}, panel)
    assert table.loc["one", "donors_used"] == 1
    assert table.loc["one", "att"] == fit.att
    fit.weights.index = [panel.units[-1]]
    with pytest.raises(ValueError, match="outside the fitted panel"):
        compare({"ineligible": fit}, panel)
