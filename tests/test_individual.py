"""Analytic checks of vanilla ISC and the two-stage donor restriction."""

import numpy as np
import pytest

from pysynth import IndividualSC, Panel, TwoStageIndividualSC


@pytest.fixture
def individual_panel():
    # A and B bracket the treated pre-path exactly. C is farther away.
    return Panel(
        [[0, 2, 1, 1], [0, 0, 2, 0], [0, 2, 100, 4]],
        n_control=3,
        n_pre=2,
        units=["A", "B", "C", "treated"],
    )


def test_isc_and_2sisc_have_distinct_donor_pools(individual_panel):
    vanilla = IndividualSC().fit(individual_panel)
    two_stage = TwoStageIndividualSC(n_neighbors=1).fit(individual_panel)
    assert vanilla.method == "ISC"
    assert two_stage.method == "2SISC"
    np.testing.assert_allclose(vanilla.weights, [0.5, 0.5, 0], atol=1e-6)
    np.testing.assert_array_equal(two_stage.weights, [1, 0, 0])
    assert vanilla.att == pytest.approx(3, abs=1e-5)
    assert two_stage.att == 4
    assert vanilla.metadata["selected_donors"] == ["A", "B", "C"]
    assert two_stage.metadata["selected_donors"] == ["A"]
    assert two_stage.metadata["n_eligible_donors"] == 3
    assert two_stage.metadata["n_selected_donors"] == 1
    assert two_stage.metadata["squared_distances"] == [1, 1, 4]


def test_2sisc_fits_convex_weights_inside_screened_pool(individual_panel):
    result = TwoStageIndividualSC(n_neighbors=2).fit(individual_panel)
    assert result.metadata["selected_donors"] == ["A", "B"]
    np.testing.assert_allclose(result.weights, [0.5, 0.5, 0], atol=1e-6)
    assert result.att == pytest.approx(3, abs=1e-5)


@pytest.mark.parametrize("n_neighbors", [3, 10])
@pytest.mark.parametrize("n_bootstrap", [0, 12])
def test_full_pool_2sisc_equals_isc(individual_panel, n_neighbors, n_bootstrap):
    options = {"n_bootstrap": n_bootstrap, "random_state": 17}
    vanilla = IndividualSC(**options).fit(individual_panel)
    two_stage = TwoStageIndividualSC(n_neighbors=n_neighbors, **options).fit(individual_panel)
    np.testing.assert_array_equal(vanilla.weights, two_stage.weights)
    np.testing.assert_array_equal(vanilla.counterfactual, two_stage.counterfactual)
    assert vanilla.metadata["bootstrap_atts"] == two_stage.metadata["bootstrap_atts"]
    assert two_stage.metadata["n_selected_donors"] == 3


def test_2sisc_features_control_both_stages(individual_panel):
    # C is the sole exact match on supplied features, despite poor pre-outcome fit.
    features = np.array([[0, 10, 5, 5]])
    result = TwoStageIndividualSC(n_neighbors=1).fit(individual_panel, features=features)
    assert result.metadata["selected_donors"] == ["C"]
    assert result.metadata["matching_features"] == "supplied_features"
    np.testing.assert_array_equal(result.weights, [0, 0, 1])
    assert result.att == -96


def test_2sisc_ties_follow_panel_order(individual_panel):
    permutation = [1, 0, 2, 3]
    reordered = Panel(
        individual_panel.y[:, permutation],
        3,
        2,
        units=[individual_panel.units[i] for i in permutation],
    )
    fit = TwoStageIndividualSC(n_neighbors=1).fit(reordered)
    assert fit.metadata["selected_donors"] == ["B"]


def test_2sisc_never_uses_donor_or_treated_post_outcomes_for_weights(individual_panel):
    changed = individual_panel.y.copy()
    changed[2, :] = [-1000, 2000, -3000, 4000]
    other = Panel(changed, 3, 2, units=individual_panel.units)
    model = TwoStageIndividualSC(n_neighbors=2, n_bootstrap=5, random_state=3)
    before, after = model.fit(individual_panel), model.fit(other)
    assert before.metadata["selected_donors"] == after.metadata["selected_donors"]
    np.testing.assert_array_equal(before.weights, after.weights)


def test_2sisc_fit_many_retains_screening_and_person_weights(individual_panel):
    changed = individual_panel.y.copy()
    changed[2, -1] += 2
    other = Panel(changed, 3, 2, units=individual_panel.units)
    result = TwoStageIndividualSC(n_neighbors=1).fit_many(
        {"first": individual_panel, "second": other},
        person_weights={"first": 1, "second": 3},
    )
    assert result.att == 5.5
    assert result.summary.loc[0, "effect"] == 5.5
    for estimate in result.estimates.values():
        assert estimate.method == "2SISC"
        assert estimate.metadata["selected_donors"] == ["A"]


@pytest.mark.parametrize("n_neighbors", [-1, 1.5, None, True])
def test_2sisc_rejects_invalid_neighbor_count(individual_panel, n_neighbors):
    with pytest.raises(ValueError, match="n_neighbors must be a positive integer"):
        TwoStageIndividualSC(n_neighbors=n_neighbors).fit(individual_panel)


def test_isc_rejects_old_screening_configuration():
    with pytest.raises(TypeError):
        IndividualSC(n_neighbors=3)
    # A former positional neighbor count must not silently become n_bootstrap.
    with pytest.raises(TypeError):
        IndividualSC(3)
