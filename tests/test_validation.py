"""Temporal validation must never expose held-out treated targets to fits."""

import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pysynth.classic import SyntheticControl
from pysynth.panel import Panel
from pysynth.results import make_estimate
from pysynth.validation import BacktestResult, backtest, select_estimator


@pytest.fixture
def trend_panel():
    time = np.arange(11, dtype=float)
    y = np.column_stack([time, time**2, 2 * time])
    y[8:, -1] += 5
    return Panel(y, 2, 8, pd.date_range("2000", periods=11, freq="YS"), ("A", "B", "T"))


class FixedEstimator:
    def __init__(self, weight=2.0, seen=None):
        self.weight = weight
        self.seen = seen

    def fit(self, panel):
        if self.seen is not None:
            self.seen.append(panel)
        return make_estimate(
            panel,
            "Fixed",
            [self.weight, 0],
            metadata={"settings": {"weight": self.weight}, "n_pre": panel.n_pre},
        )


def test_analytic_predictions_summary_and_complete_fold_boundaries(trend_panel):
    result = backtest(
        trend_panel,
        {"good": FixedEstimator, "bad": lambda: FixedEstimator(1)},
        initial_window=3,
        horizon=2,
    )
    assert isinstance(result, BacktestResult)
    assert result.metadata["origins"] == (3, 5)
    assert result.metadata["step"] == 2
    assert result.metadata["actual_post_treatment_used"] is False
    assert result.best_model == "good"
    summary = result.summary
    assert summary.index.tolist() == ["good", "bad"]
    assert summary["rank"].tolist() == [1, 2]
    assert summary["n_folds"].tolist() == [2, 2]
    assert summary["n_predictions"].tolist() == [4, 4]
    assert summary.loc["good", "validation_rmse"] == 0
    assert summary.loc["bad", "validation_rmse"] == pytest.approx(np.sqrt(21.5))
    assert summary.loc["bad", "mae"] == 4.5
    assert summary.loc["bad", "bias"] == 4.5
    bad = result.predictions.loc[result.predictions.model == "bad"]
    np.testing.assert_array_equal(bad.observed, [6, 8, 10, 12])
    np.testing.assert_array_equal(bad.counterfactual, [3, 4, 5, 6])
    np.testing.assert_array_equal(bad.error, [3, 4, 5, 6])
    first = result.scores.iloc[0]
    assert first.training_start == trend_panel.times[0]
    assert first.training_end == trend_panel.times[2]
    assert first.validation_start == trend_panel.times[3]
    assert first.validation_end == trend_panel.times[4]
    assert first.train_rmse == 0
    assert first.n_validation == 2
    assert len(result.metadata["fit_metadata"]) == 4
    assert result.metadata["fit_metadata"][0] == {
        "model": "good",
        "fold": 0,
        "method": "Fixed",
        "metadata": {"settings": {"weight": 2.0}, "n_pre": 3},
    }


def test_actual_post_data_cannot_change_validation(trend_panel):
    poisoned_y = trend_panel.y.copy()
    poisoned_y[trend_panel.n_pre :] = -1e100
    poisoned = replace(trend_panel, y=poisoned_y)
    factories = {"good": FixedEstimator, "bad": lambda: FixedEstimator(1)}
    original = backtest(trend_panel, factories, initial_window=2, horizon=2)
    altered = backtest(poisoned, factories, initial_window=2, horizon=2)
    pd.testing.assert_frame_equal(original.scores, altered.scores)
    pd.testing.assert_frame_equal(original.predictions, altered.predictions)
    pd.testing.assert_frame_equal(original.summary, altered.summary)


def test_each_fold_gets_only_training_and_masked_targets(trend_panel):
    seen = []

    class TargetReader:
        def fit(self, panel):
            seen.append(panel)
            return make_estimate(panel, "TargetReader", [0, 0], counterfactual=panel.treated)

    result = backtest(trend_panel, {"peeker": TargetReader}, initial_window=3, horizon=2)
    for panel, origin in zip(seen, (3, 5)):
        assert panel.n_pre == origin
        assert len(panel.times) == origin + 2
        assert panel.units == trend_panel.units
        assert panel.times.equals(trend_panel.times[: origin + 2])
        np.testing.assert_array_equal(panel.y[:origin], trend_panel.y[:origin])
        np.testing.assert_array_equal(panel.control, trend_panel.control[: origin + 2])
        np.testing.assert_array_equal(panel.y[origin:, -1], [2 * (origin - 1)] * 2)
        assert panel.times[-1] < trend_panel.times[trend_panel.n_pre]
    np.testing.assert_array_equal(result.predictions.counterfactual, [4, 4, 8, 8])
    np.testing.assert_array_equal(result.predictions.error, [2, 4, 2, 4])


def test_heldout_targets_are_used_only_for_external_scoring(trend_panel):
    changed_y = trend_panel.y.copy()
    changed_y[6:8, -1] += 100
    changed = replace(trend_panel, y=changed_y)
    original = backtest(trend_panel, {"fixed": FixedEstimator}, initial_window=6, horizon=2)
    altered = backtest(changed, {"fixed": FixedEstimator}, initial_window=6, horizon=2)
    pd.testing.assert_series_equal(
        original.predictions.counterfactual, altered.predictions.counterfactual
    )
    assert original.summary.loc["fixed", "validation_rmse"] == 0
    assert altered.summary.loc["fixed", "validation_rmse"] == 100


def test_multitreated_mask_is_per_unit_and_scoring_is_equal_average(trend_panel):
    values = np.column_stack([trend_panel.y, trend_panel.y[:, -1] + 100])
    panel = replace(trend_panel, y=values, units=("A", "B", "T1", "T2"))
    seen = []

    class AveragePredictor:
        def fit(self, panel):
            seen.append(panel)
            return make_estimate(panel, "Average", [2, 0], intercept=50)

    result = backtest(panel, {"average": AveragePredictor}, initial_window=3, horizon=2)
    assert result.metadata["n_treated"] == 2
    np.testing.assert_array_equal(seen[0].y[3:, 2:], [[4, 104], [4, 104]])
    np.testing.assert_array_equal(result.predictions.observed, [56, 58, 60, 62])
    assert result.summary.loc["average", "validation_rmse"] == 0


def test_overlapping_windows_count_prediction_instances(trend_panel):
    result = backtest(
        trend_panel, {"bad": lambda: FixedEstimator(1)}, initial_window=2, horizon=3, step=1
    )
    errors = np.array([2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7])
    np.testing.assert_array_equal(result.predictions.error, errors)
    assert result.metadata["origins"] == (2, 3, 4, 5)
    assert result.summary.loc["bad", "n_predictions"] == 12
    assert result.predictions.time.value_counts().max() == 3
    assert result.summary.loc["bad", "validation_rmse"] == pytest.approx(
        np.sqrt(np.mean(errors**2))
    )
    assert result.summary.loc["bad", "validation_rmse"] != pytest.approx(
        result.scores.validation_rmse.mean()
    )


def test_ties_preserve_factory_insertion_order(trend_panel):
    result = backtest(
        trend_panel,
        {"z-first": FixedEstimator, "a-second": FixedEstimator},
        initial_window=3,
        horizon=2,
    )
    assert result.summary.index.tolist() == ["z-first", "a-second"]
    assert result.best_model == "z-first"


def test_selection_refits_winner_with_fresh_instance_on_original_panel(trend_panel):
    instances, seen = [], []

    def factory():
        estimator = FixedEstimator(seen=seen)
        instances.append(estimator)
        return estimator

    result = select_estimator(
        trend_panel,
        {"good": factory, "bad": lambda: FixedEstimator(1)},
        initial_window=3,
        horizon=2,
    )
    assert result.selected_name == result.validation.best_model == "good"
    assert len(instances) == len({id(instance) for instance in instances}) == 3
    assert seen[-1] is not trend_panel
    np.testing.assert_array_equal(seen[-1].y, trend_panel.y)
    assert seen[-1].times.equals(trend_panel.times)
    assert seen[-1].units == trend_panel.units
    assert result.estimate.n_pre == trend_panel.n_pre
    assert result.estimate.att == 5
    assert result.estimate.metadata == {"settings": {"weight": 2}, "n_pre": 8}


def test_selected_identity_cannot_depend_on_actual_post_outcomes(trend_panel):
    poisoned_y = trend_panel.y.copy()
    poisoned_y[8:, -1] -= 42
    factories = {"good": FixedEstimator, "bad": lambda: FixedEstimator(1)}
    original = select_estimator(trend_panel, factories, initial_window=3, horizon=2)
    altered = select_estimator(
        replace(trend_panel, y=poisoned_y), factories, initial_window=3, horizon=2
    )
    assert original.selected_name == altered.selected_name
    pd.testing.assert_frame_equal(original.validation.summary, altered.validation.summary)
    assert original.estimate.att == 5
    assert altered.estimate.att == -37


def test_final_selection_fit_cannot_mutate_original_panel(trend_panel):
    before = trend_panel.y.copy()
    seen = []

    class MutatingEstimator:
        def fit(self, panel):
            seen.append(panel)
            panel.y.flags.writeable = True
            panel.y[:, 1] += 1000
            return FixedEstimator().fit(panel)

    result = select_estimator(
        trend_panel, {"mutating": MutatingEstimator}, initial_window=3, horizon=2
    )
    np.testing.assert_array_equal(trend_panel.y, before)
    assert not trend_panel.y.flags.writeable
    assert all(supplied is not trend_panel for supplied in seen)
    assert result.estimate.att == 5


def test_nonfinite_selected_att_is_rejected_with_final_fit_context(trend_panel):
    class ExtremeFinalEstimator:
        def fit(self, panel):
            estimate = FixedEstimator().fit(panel)
            if panel.n_pre == trend_panel.n_pre:
                estimate.counterfactual.iloc[panel.n_pre :] = -1e308
            return estimate

    with pytest.warns(RuntimeWarning, match="overflow"):
        with pytest.raises(RuntimeError, match="Final fit.*'extreme'.*must be finite"):
            select_estimator(
                trend_panel, {"extreme": ExtremeFinalEstimator}, initial_window=3, horizon=2
            )


def test_reused_estimator_instance_is_rejected(trend_panel):
    estimator = FixedEstimator()
    with pytest.raises(RuntimeError, match="model 'reuse', fold 1.*fresh estimator"):
        backtest(trend_panel, {"reuse": lambda: estimator}, initial_window=3, horizon=2)


def test_reused_instance_in_final_fit_is_rejected(trend_panel):
    estimator = FixedEstimator()
    with pytest.raises(RuntimeError, match="Final fit.*'reuse'.*fresh estimator"):
        select_estimator(trend_panel, {"reuse": lambda: estimator}, initial_window=6, horizon=2)


def test_metadata_snapshots_are_independent_of_later_fits(trend_panel):
    shared = {"history": []}

    class MetadataEstimator:
        def fit(self, panel):
            shared["history"].append(panel.n_pre)
            return make_estimate(panel, "SharedMetadata", [2, 0], metadata=shared)

    result = backtest(trend_panel, {"shared": MetadataEstimator}, initial_window=3, horizon=2)
    assert result.metadata["fit_metadata"][0]["metadata"]["history"] == [3]
    assert result.metadata["fit_metadata"][1]["metadata"]["history"] == [3, 5]


def test_solver_warnings_and_failed_fit_are_not_hidden(trend_panel):
    class Failure:
        def fit(self, panel):
            warnings.warn("original solver warning", RuntimeWarning)
            raise ArithmeticError("solver failed")

    with pytest.warns(RuntimeWarning, match="original solver warning"):
        with pytest.raises(RuntimeError, match="model 'failure', fold 0.*solver failed") as error:
            backtest(trend_panel, {"failure": Failure}, initial_window=3, horizon=2)
    assert isinstance(error.value.__cause__, ArithmeticError)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_window": 1, "horizon": 2},
        {"initial_window": True, "horizon": 2},
        {"initial_window": 2.0, "horizon": 2},
        {"initial_window": 2, "horizon": 0},
        {"initial_window": 2, "horizon": True},
        {"initial_window": 2, "horizon": 1.5},
        {"initial_window": 2, "horizon": 2, "step": 0},
        {"initial_window": 2, "horizon": 2, "step": True},
        {"initial_window": 2, "horizon": 2, "step": 1.5},
        {"initial_window": 7, "horizon": 2},
    ],
)
def test_invalid_fold_settings(trend_panel, kwargs):
    with pytest.raises(ValueError):
        backtest(trend_panel, {"fixed": FixedEstimator}, **kwargs)


@pytest.mark.parametrize(
    "estimators", [{}, [], {"": FixedEstimator}, {"  ": FixedEstimator}, {1: FixedEstimator}]
)
def test_invalid_candidate_mapping(trend_panel, estimators):
    with pytest.raises(ValueError):
        backtest(trend_panel, estimators, initial_window=3, horizon=2)


def test_requires_panel_and_factory_not_fitted_estimate(trend_panel):
    with pytest.raises(TypeError, match="panel must be"):
        backtest(trend_panel.y, {"fixed": FixedEstimator}, initial_window=3, horizon=2)
    for candidate in (FixedEstimator(), FixedEstimator().fit(trend_panel)):
        with pytest.raises(TypeError, match="zero-argument factory"):
            backtest(trend_panel, {"bad": candidate}, initial_window=3, horizon=2)


@pytest.mark.parametrize(
    "factory", [lambda: object(), lambda: None, lambda: {"fit": "not an estimator"}]
)
def test_factory_must_return_estimator(trend_panel, factory):
    with pytest.raises(RuntimeError, match="model 'bad', fold 0.*fresh estimator"):
        backtest(trend_panel, {"bad": factory}, initial_window=3, horizon=2)


@pytest.mark.parametrize(
    "malform",
    [
        lambda estimate: None,
        lambda estimate: replace(estimate, n_pre=estimate.n_pre + 1),
        lambda estimate: replace(estimate, n_pre=True),
        lambda estimate: replace(estimate, observed=estimate.observed + 1),
        lambda estimate: replace(estimate, observed=estimate.observed.iloc[::-1]),
        lambda estimate: replace(estimate, counterfactual=estimate.counterfactual.iloc[::-1]),
        lambda estimate: replace(estimate, counterfactual=estimate.counterfactual * np.nan),
        lambda estimate: replace(estimate, counterfactual=estimate.counterfactual.astype(str)),
        lambda estimate: replace(estimate, counterfactual=estimate.counterfactual + 1j),
        lambda estimate: replace(estimate, counterfactual=estimate.counterfactual.to_numpy()),
        lambda estimate: replace(estimate, weights=estimate.weights.iloc[::-1]),
        lambda estimate: replace(estimate, weights=estimate.weights * np.nan),
        lambda estimate: replace(estimate, weights=estimate.weights.astype(str)),
        lambda estimate: replace(estimate, weights=estimate.weights + 1j),
        lambda estimate: replace(estimate, weights=estimate.weights.to_numpy()),
        lambda estimate: replace(estimate, intercept=np.inf),
        lambda estimate: replace(estimate, intercept=1j),
        lambda estimate: replace(estimate, intercept="1.0"),
        lambda estimate: replace(estimate, time_weights=pd.Series([1, 0, 0])),
        lambda estimate: replace(
            estimate,
            time_weights=pd.Series(np.nan, index=estimate.observed.index[: estimate.n_pre]),
        ),
        lambda estimate: replace(estimate, metadata=None),
    ],
)
def test_malformed_estimates_fail_with_model_and_fold_context(trend_panel, malform):
    class Malformed:
        def fit(self, panel):
            return malform(FixedEstimator().fit(panel))

    with pytest.raises(RuntimeError, match="model 'malformed', fold 0"):
        backtest(trend_panel, {"malformed": Malformed}, initial_window=3, horizon=2)


def test_native_scm_can_be_selected_without_optional_backends(panel):
    result = select_estimator(
        panel,
        {"SCM": SyntheticControl, "ridge": lambda: SyntheticControl(ridge=10)},
        initial_window=10,
        horizon=5,
    )
    assert result.selected_name == "SCM"
    assert result.validation.summary.loc["SCM", "validation_rmse"] < 1e-5
    assert result.estimate.att == pytest.approx(3, abs=2e-5)


def test_scoring_is_invariant_to_large_outcome_units(trend_panel):
    original = backtest(
        trend_panel, {"bad": lambda: FixedEstimator(1)}, initial_window=3, horizon=2
    )
    scaled_panel = replace(trend_panel, y=trend_panel.y * 1e200)
    scaled = backtest(scaled_panel, {"bad": lambda: FixedEstimator(1)}, initial_window=3, horizon=2)
    for column in ("validation_rmse", "mae", "bias"):
        assert scaled.summary.loc["bad", column] / 1e200 == pytest.approx(
            original.summary.loc["bad", column]
        )
