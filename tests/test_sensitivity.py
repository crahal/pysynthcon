"""Full refits under donor removal, with explicit analytic sensitivity checks."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pysynth.classic import SyntheticControl
from pysynth.individual import TwoStageIndividualSC
from pysynth.panel import Panel
from pysynth.results import make_estimate
from pysynth.sensitivity import donor_sensitivity


@pytest.fixture
def panel():
    return Panel(
        [[1, 0, 3, 1], [2, 0, 3, 2], [3, 0, 3, 3], [4, 0, 3, 6], [5, 0, 3, 7]],
        3,
        3,
        times=pd.date_range("2000", periods=5, freq="YS"),
        units=("A", "B", "C", "treated"),
    )


def test_leave_one_out_finds_influential_donor_and_retains_full_results(panel):
    result = donor_sensitivity(panel, {"SCM": SyntheticControl})
    assert list(result.baseline) == ["SCM"]
    assert list(result.estimates) == ["omit:A", "omit:B", "omit:C"]
    table = result.table.set_index("scenario")
    assert result.baseline["SCM"].att == pytest.approx(2, abs=1e-6)
    assert table.loc["omit:A", "att"] == pytest.approx(4.5, abs=1e-6)
    np.testing.assert_allclose(table.loc[["omit:B", "omit:C"], "att"], 2, atol=1e-6)
    assert table.loc["omit:A", "excluded_units"] == ("A",)
    assert (table.donor_count == 2).all()
    assert (table.baseline_pre_rmse < 1e-6).all()
    assert table.loc["omit:A", "pre_rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert not table.sign_changed.any()
    summary = result.summary.loc["SCM"]
    assert summary.baseline_att == pytest.approx(2, abs=1e-6)
    assert summary.min_att == pytest.approx(2, abs=1e-6)
    assert summary.max_att == pytest.approx(4.5, abs=1e-6)
    assert summary.max_abs_change == pytest.approx(2.5, abs=1e-6)
    assert summary.worst_scenario == "omit:A"
    assert summary.n_scenarios == 3
    fit = result.estimates["omit:A"]["SCM"]
    assert list(fit.weights.index) == ["B", "C"]
    assert fit.counterfactual.index.equals(panel.times)
    assert "optimality_gap" in fit.metadata


def test_named_group_removes_all_members_and_preserves_input_order(panel):
    result = donor_sensitivity(panel, {"SCM": SyntheticControl}, exclusions={"nearby": ["C", "A"]})
    row = result.table.iloc[0]
    assert row.excluded_units == ("A", "C")
    assert row.donor_count == 1
    assert row.att == 6.5
    assert list(result.estimates["nearby"]["SCM"].weights.index) == ["B"]


class EqualWeightEstimator:
    def fit(self, panel):
        return make_estimate(panel, "equal", np.ones(panel.n_control) / panel.n_control)


def test_all_models_and_scenarios_get_fresh_objects_and_full_protocol(panel):
    constructed, calls = [], []

    class TrackedEstimator:
        def __init__(self, budget):
            self.budget = budget
            constructed.append(self)

        def fit(self, fitted_panel):
            calls.append((self, self.budget, fitted_panel.units))
            return make_estimate(
                fitted_panel,
                "tracked",
                np.ones(fitted_panel.n_control) / fitted_panel.n_control,
                metadata={"budget": self.budget},
            )

    result = donor_sensitivity(
        panel,
        {"full": lambda: TrackedEstimator(200), "other": lambda: TrackedEstimator(73)},
    )
    assert len(constructed) == len(calls) == 8
    assert len({id(obj) for obj in constructed}) == 8
    assert [call[1] for call in calls] == [200, 73] * 4
    assert result.table.model.tolist() == ["full", "other"] * 3
    for scenario, fits in result.estimates.items():
        assert fits["full"].metadata["budget"] == 200
        assert fits["other"].metadata["budget"] == 73
        assert scenario.startswith("omit:")


def test_zero_weight_donors_are_refitted_not_deleted_and_renormalized(panel):
    class PoolDependentEstimator:
        def fit(self, fitted_panel):
            # A valid estimator can change its tuning/selection with donor pool.
            weights = np.zeros(fitted_panel.n_control)
            weights[0 if fitted_panel.n_control == 3 else -1] = 1
            return make_estimate(fitted_panel, "pool-dependent", weights)

    result = donor_sensitivity(panel, {"adaptive": PoolDependentEstimator})
    assert result.baseline["adaptive"].weights["B"] == 0
    fit = result.estimates["omit:B"]["adaptive"]
    np.testing.assert_array_equal(fit.weights, [0, 1])
    assert fit.att == 3.5
    assert result.baseline["adaptive"].att == 2


def test_fit_metadata_snapshots_survive_shared_nested_state_and_later_mutation(panel):
    shared = {"history": {"donor_pools": []}}

    class SharedDiagnosticsEstimator:
        def fit(self, fitted_panel):
            shared["history"]["donor_pools"].append(fitted_panel.units[: fitted_panel.n_control])
            return make_estimate(
                fitted_panel,
                "shared-diagnostics",
                np.ones(fitted_panel.n_control) / fitted_panel.n_control,
                metadata=shared,
            )

    result = donor_sensitivity(panel, {"shared": SharedDiagnosticsEstimator})
    fits = [result.baseline["shared"], *(x["shared"] for x in result.estimates.values())]
    assert [len(fit.metadata["history"]["donor_pools"]) for fit in fits] == [1, 2, 3, 4]
    assert fits[0].metadata["history"]["donor_pools"] == [("A", "B", "C")]
    assert fits[1].metadata["history"]["donor_pools"][-1] == ("B", "C")
    shared["history"]["donor_pools"].clear()
    fits[-1].metadata["history"]["donor_pools"].append(("external",))
    assert [len(fit.metadata["history"]["donor_pools"]) for fit in fits] == [1, 2, 3, 5]


def test_2sisc_reselects_nearest_neighbors_in_reduced_pool(panel):
    result = donor_sensitivity(
        panel,
        {
            "2SISC": lambda: TwoStageIndividualSC(
                backend="fixed_metric", n_neighbors=1, n_bootstrap=0
            )
        },
    )
    assert result.baseline["2SISC"].metadata["selected_donors"] == ["A"]
    removed = result.estimates["omit:A"]["2SISC"]
    assert removed.metadata["selected_donors"] == ["C"]
    assert removed.metadata["n_eligible_donors"] == 2
    np.testing.assert_array_equal(removed.weights, [0, 1])


def test_negative_weights_are_retained_and_excluded_in_default_analysis(panel):
    class SignedEstimator:
        def fit(self, fitted_panel):
            weights = np.zeros(fitted_panel.n_control)
            weights[0], weights[-1] = -1, 2
            return make_estimate(fitted_panel, "signed", weights)

    result = donor_sensitivity(panel, {"signed": SignedEstimator})
    assert result.baseline["signed"].weights["A"] == -1
    assert "omit:A" in result.estimates
    np.testing.assert_array_equal(result.estimates["omit:B"]["signed"].weights, [-1, 2])


def test_multiple_treated_units_and_original_panel_remain_intact(panel):
    source = Panel(
        np.column_stack([panel.y, panel.y[:, -1] + 2]),
        panel.n_control,
        panel.n_pre,
        panel.times,
        (*panel.units, "treated2"),
    )
    before = source.y.copy()
    seen = []

    class MutatingEstimator:
        def fit(self, fitted_panel):
            seen.append((fitted_panel.units, fitted_panel.n_treated))
            fitted_panel.y.flags.writeable = True
            fitted_panel.y[:, 0] += 100
            return make_estimate(
                fitted_panel, "mutating", np.ones(fitted_panel.n_control) / fitted_panel.n_control
            )

    result = donor_sensitivity(source, {"mutating": MutatingEstimator})
    np.testing.assert_array_equal(source.y, before)
    assert not source.y.flags.writeable
    assert all(units[-2:] == ("treated", "treated2") for units, _ in seen)
    assert all(n_treated == 2 for _, n_treated in seen)
    for fits in result.estimates.values():
        np.testing.assert_array_equal(fits["mutating"].observed, source.treated)


@pytest.mark.parametrize(
    "baseline_att,scenario_att,changed",
    [(1, -1, True), (-1, 1, True), (0, -1, False), (1, 0, False), (0, 0, False), (-2, -1, False)],
)
def test_sign_changes_only_count_strict_reversals(panel, baseline_att, scenario_att, changed):
    class FixedEffectEstimator:
        def fit(self, fitted_panel):
            effect = baseline_att if fitted_panel.n_control == 3 else scenario_att
            return make_estimate(
                fitted_panel,
                "fixed-effect",
                np.zeros(fitted_panel.n_control),
                counterfactual=fitted_panel.treated - effect,
            )

    result = donor_sensitivity(panel, {"fixed": FixedEffectEstimator})
    assert result.table.sign_changed.eq(changed).all()
    assert result.summary.loc["fixed", "worst_scenario"] == "omit:A"


def test_default_scenario_names_do_not_collide_for_mixed_label_types(panel):
    mixed = replace(panel, units=(1, "1", "C", "treated"))
    result = donor_sensitivity(mixed, {"equal": EqualWeightEstimator})
    assert len(result.estimates) == 3
    assert result.table.excluded_units.tolist() == [(1,), ("1",), ("C",)]


@pytest.mark.parametrize(
    "exclusions,match",
    [
        ({}, "nonempty mapping"),
        ([], "nonempty mapping"),
        ({"": ["A"]}, "names"),
        ({" ": ["A"]}, "names"),
        ({1: ["A"]}, "names"),
        ({"x": []}, "nonempty, unique"),
        ({"x": ["A", "A"]}, "nonempty, unique"),
        ({"x": ["unknown"]}, "unknown or treated"),
        ({"x": ["treated"]}, "unknown or treated"),
        ({"x": ["A", "B", "C"]}, "leave at least one"),
        ({"x": "A"}, "sequence"),
        ({"x": None}, "hashable"),
        ({"x": [["A"]]}, "hashable"),
    ],
)
def test_bad_exclusions_fail_before_fitting(panel, exclusions, match):
    def never_called():
        pytest.fail("Invalid exclusions must be rejected before fitting")

    with pytest.raises(ValueError, match=match):
        donor_sensitivity(panel, {"SCM": never_called}, exclusions=exclusions)


def test_single_donor_cannot_be_omitted(panel):
    small = Panel(panel.y[:, [0, -1]], 1, panel.n_pre)
    with pytest.raises(ValueError, match="at least two control"):
        donor_sensitivity(small, {"SCM": SyntheticControl})


@pytest.mark.parametrize("estimators", [{}, [], {"": SyntheticControl}, {3: SyntheticControl}])
def test_bad_model_specification(panel, estimators):
    with pytest.raises(ValueError):
        donor_sensitivity(panel, estimators)


def test_requires_panel_and_callable_factories(panel):
    with pytest.raises(TypeError, match="must be a Panel"):
        donor_sensitivity(None, {"SCM": SyntheticControl})
    with pytest.raises(TypeError, match="zero-argument factory"):
        donor_sensitivity(panel, {"SCM": SyntheticControl()})


def test_reused_estimators_are_rejected_with_context(panel):
    shared = EqualWeightEstimator()
    with pytest.raises(RuntimeError, match="model 'equal'.*scenario 'omit:A'.*fresh"):
        donor_sensitivity(panel, {"equal": lambda: shared})


def test_failed_factory_has_model_and_baseline_context(panel):
    def broken():
        raise OSError("backend unavailable")

    with pytest.raises(RuntimeError, match="model 'broken'.*scenario 'baseline'.*backend") as error:
        donor_sensitivity(panel, {"broken": broken})
    assert isinstance(error.value.__cause__, OSError)


def test_failed_refit_has_model_and_exclusion_context(panel):
    class FailingEstimator:
        def fit(self, fitted_panel):
            if "A" not in fitted_panel.units:
                raise ArithmeticError("did not converge")
            return EqualWeightEstimator().fit(fitted_panel)

    with pytest.raises(RuntimeError, match="model 'fails'.*scenario 'omit:A'.*converge"):
        donor_sensitivity(panel, {"fails": FailingEstimator})


@pytest.mark.parametrize(
    "corruption",
    [
        "not_estimate",
        "timing",
        "observed_labels",
        "observed_values",
        "counterfactual_labels",
        "counterfactual_nan",
        "weights_labels",
        "weights_nan",
        "time_weights_labels",
        "intercept_inf",
    ],
)
def test_malformed_fit_is_never_silently_included(panel, corruption):
    class MalformedEstimator:
        def fit(self, fitted_panel):
            fit = EqualWeightEstimator().fit(fitted_panel)
            if corruption == "not_estimate":
                return object()
            if corruption == "timing":
                fit.n_pre -= 1
            elif corruption == "observed_labels":
                fit.observed = fit.observed.iloc[::-1]
            elif corruption == "observed_values":
                fit.observed = fit.observed + 1
            elif corruption == "counterfactual_labels":
                fit.counterfactual = fit.counterfactual.iloc[::-1]
            elif corruption == "counterfactual_nan":
                fit.counterfactual.iloc[0] = np.nan
            elif corruption == "weights_labels":
                fit.weights = fit.weights.iloc[::-1]
            elif corruption == "weights_nan":
                fit.weights.iloc[0] = np.nan
            elif corruption == "time_weights_labels":
                fit.time_weights = pd.Series([0.5, 0.5], index=["wrong", "labels"])
            elif corruption == "intercept_inf":
                fit.intercept = np.inf
            return fit

    with pytest.raises(RuntimeError, match="model 'malformed'.*scenario 'baseline'"):
        donor_sensitivity(panel, {"malformed": MalformedEstimator})


def test_finite_but_overflowing_att_difference_is_rejected(panel):
    class ExtremeEstimator:
        def fit(self, fitted_panel):
            # Keep summation finite by using one post-period.
            effect = 1e308 if fitted_panel.n_control == 3 else -1e308
            counterfactual = fitted_panel.treated.copy()
            counterfactual[-1] = -effect
            return make_estimate(
                fitted_panel,
                "extreme",
                np.zeros(fitted_panel.n_control),
                counterfactual=counterfactual,
            )

    one_post = Panel(panel.y[:4], 3, 3, panel.times[:4], panel.units)
    with pytest.raises(RuntimeError, match="ATT change is nonfinite"):
        donor_sensitivity(one_post, {"extreme": ExtremeEstimator})


def test_nonfinite_att_from_finite_paths_is_rejected_with_context(panel):
    class ExtremeMeanEstimator:
        def fit(self, fitted_panel):
            counterfactual = fitted_panel.treated.copy()
            counterfactual[fitted_panel.n_pre :] = -1e308
            return make_estimate(
                fitted_panel,
                "extreme-mean",
                np.zeros(fitted_panel.n_control),
                counterfactual=counterfactual,
            )

    with np.errstate(over="ignore"):
        with pytest.raises(RuntimeError, match="model 'extreme'.*scenario 'baseline'.*finite"):
            donor_sensitivity(panel, {"extreme": ExtremeMeanEstimator})
