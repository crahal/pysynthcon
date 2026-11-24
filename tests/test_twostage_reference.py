"""Compare the two-stage backend with the authors' KDTree/Dataprep recipe."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from pysynth import Panel, TwoStageIndividualSC

pysyncon = pytest.importorskip("pysyncon")
KDTree = pytest.importorskip("sklearn.neighbors").KDTree


@pytest.fixture
def multivariate_panel():
    rng = np.random.default_rng(431)
    donors = 10 + rng.normal(size=(18, 6)).cumsum(axis=0)
    treated = 0.25 * donors[:, 0] + 0.6 * donors[:, 2] + 0.15 * donors[:, 4]
    treated[12:] += 2
    panel = Panel(
        np.column_stack([donors, treated]),
        n_control=6,
        n_pre=12,
        units=list("ABCDEFZ"),
        times=pd.date_range("2000", periods=18, freq="YS"),
    )
    predictors = {
        "z_age": rng.normal(size=(12, 7)) + np.arange(7),
        "z_education": rng.uniform(0, 5, size=(12, 7)),
    }
    return panel, predictors


def author_recipe(panel, predictors, n_neighbors):
    """Independently reproduce src/utils.py's core sc_b operations.

    Use k+1 in its KDTree query so the treated entry leaves k actual donors.
    Labels are sorted as in its pivoted frame. No PySynthCon optimizer helper is
    called: this is raw pysyncon with the authors' mean-predictor specification.
    """
    order = [panel.n_control, *range(panel.n_control)]
    design = np.vstack([panel.y[: panel.n_pre], *predictors.values()])[:, order]
    tree = KDTree(design.T, leaf_size=30, metric="euclidean")
    indices = tree.query(design.T, k=n_neighbors + 1, return_distance=False)[0]
    donor_indices = sorted(order[i] for i in indices if order[i] != panel.n_control)
    rows = []
    for unit in [*donor_indices, panel.n_control]:
        for time in range(len(panel.y)):
            row = {"unit": unit, "time": time, "outcome": panel.y[time, unit]}
            row.update(
                {
                    name: matrix[time, unit] if time < panel.n_pre else np.nan
                    for name, matrix in predictors.items()
                }
            )
            rows.append(row)
    frame = pd.DataFrame(rows)
    prep = pysyncon.Dataprep(
        foo=frame,
        predictors=sorted(["outcome", *predictors]),
        predictors_op="mean",
        dependent="outcome",
        unit_variable="unit",
        time_variable="time",
        treatment_identifier=panel.n_control,
        controls_identifier=donor_indices,
        time_predictors_prior=range(panel.n_pre),
        time_optimize_ssr=range(panel.n_pre),
    )
    reference = pysyncon.Synth()
    reference.fit(dataprep=prep)
    weights = np.zeros(panel.n_control)
    weights[donor_indices] = reference.W
    return weights, donor_indices


@pytest.mark.parametrize("n_neighbors", [3, 6])
def test_two_stage_matches_raw_author_recipe(multivariate_panel, n_neighbors):
    panel, predictors = multivariate_panel
    expected, selected = author_recipe(panel, predictors, n_neighbors)
    model = TwoStageIndividualSC(n_neighbors=n_neighbors)
    result = model.fit(panel, predictors=predictors)
    np.testing.assert_allclose(result.weights, expected, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(result.counterfactual, panel.control @ expected, atol=1e-8)
    assert result.metadata["selected_donors"] == [panel.units[i] for i in selected]
    assert result.metadata["backend"] == "pysyncon_mean_predictors"
    assert result.metadata["n_bootstrap"] == 0
    assert result.metadata["predictor_aggregation"] == "mean"
    assert len(result.metadata["predictor_weights"]) == len(predictors) + 1
    assert model.backend_.dataprep.special_predictors is None
    x0, x1 = model.backend_.dataprep.make_covariate_mats()
    np.testing.assert_allclose(
        x1, np.vstack([panel.y[:12], *predictors.values()]).reshape(3, 12, 7).mean(axis=1)[:, -1]
    )
    assert x0.shape == (3, n_neighbors)


def test_two_stage_weights_ignore_all_post_outcomes(multivariate_panel):
    panel, predictors = multivariate_panel
    changed = panel.y.copy()
    changed[panel.n_pre :] = np.arange(changed[panel.n_pre :].size).reshape(-1, 7) * 1e6
    altered = Panel(changed, panel.n_control, panel.n_pre, times=panel.times, units=panel.units)
    model = TwoStageIndividualSC(n_neighbors=3)
    before = model.fit(panel, predictors=predictors)
    after = model.fit(altered, predictors=predictors)
    np.testing.assert_array_equal(before.weights, after.weights)
    assert before.metadata["selected_donors"] == after.metadata["selected_donors"]


def test_two_stage_covariate_order_matches_authors_sorted_pivot(multivariate_panel):
    panel, predictors = multivariate_panel
    first = TwoStageIndividualSC(n_neighbors=3).fit(panel, predictors=predictors)
    second = TwoStageIndividualSC(n_neighbors=3).fit(
        panel, predictors=dict(reversed(list(predictors.items())))
    )
    np.testing.assert_array_equal(first.weights, second.weights)
    assert first.metadata["predictor_groups"] == second.metadata["predictor_groups"]


def test_default_two_stage_single_donor_needs_no_optional_backend(monkeypatch):
    monkeypatch.setattr(
        "pysynth.backends._require", lambda *args: pytest.fail("The simplex is analytic")
    )
    panel = Panel([[0, 10, 1], [1, 11, 2], [2, 12, 6]], 2, 2, units=["A", "B", "Z"])
    fit = TwoStageIndividualSC(n_neighbors=1).fit(panel)
    np.testing.assert_array_equal(fit.weights, [1, 0])
    assert fit.att == 4
    assert fit.metadata["optimization"]["solution"] == "single_donor_simplex"


@pytest.mark.parametrize("n_bootstrap", [1, 50, True, -1, 0.0])
def test_author_backend_rejects_donor_resampling(multivariate_panel, n_bootstrap):
    panel, _ = multivariate_panel
    with pytest.raises(ValueError, match="requires n_bootstrap=0"):
        TwoStageIndividualSC(n_bootstrap=n_bootstrap).fit(panel)


def test_author_backend_rejects_feature_rows_without_series_groups(multivariate_panel):
    panel, _ = multivariate_panel
    with pytest.raises(ValueError, match="named time-series"):
        TwoStageIndividualSC().fit(panel, features=panel.y[: panel.n_pre])
    with pytest.raises(ValueError, match="n_pre-by-units"):
        TwoStageIndividualSC().fit(panel, predictors={"age": panel.y})


def test_two_stage_fit_many_retains_case_predictors_and_person_weights(multivariate_panel):
    panel, predictors = multivariate_panel
    changed = panel.y.copy()
    changed[panel.n_pre :, -1] += 4
    second = Panel(changed, panel.n_control, panel.n_pre, times=panel.times, units=panel.units)
    model = TwoStageIndividualSC(n_neighbors=3)
    result = model.fit_many(
        {"a": panel, "b": second},
        predictors={"a": predictors, "b": predictors},
        person_weights={"a": 1, "b": 3},
    )
    assert result.att == pytest.approx(result.estimates["a"].att + 3)
    np.testing.assert_array_equal(result.estimates["a"].weights, result.estimates["b"].weights)


def test_explicit_optimizer_options_are_checked(multivariate_panel):
    panel, predictors = multivariate_panel
    with pytest.raises(RuntimeError, match="optimization failed"):
        TwoStageIndividualSC(options={"optim_options": {"maxiter": 1}}).fit(
            panel, predictors=predictors
        )


def test_constant_mean_predictors_raise_without_silently_changing_specification():
    panel = Panel([[0, 2, 1], [2, 0, 1], [3, 4, 7]], 2, 2)
    with pytest.raises(ValueError, match="constant predictors"):
        TwoStageIndividualSC(n_neighbors=2).fit(panel)


def test_recorded_options_do_not_alias_caller_settings(multivariate_panel):
    panel, predictors = multivariate_panel
    options = {"optim_options": {"maxiter": 1000}}
    before = deepcopy(options)
    model = TwoStageIndividualSC(n_neighbors=3, options=options)
    result = model.fit(panel, predictors=predictors)
    assert options == before
    options["optim_options"]["maxiter"] = 17
    assert result.metadata["options"] == before
    result.metadata["options"]["optim_options"]["maxiter"] = 20
    assert model.options["optim_options"]["maxiter"] == 17
