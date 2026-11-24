"""Pre-treatment temporal validation and transparent estimator selection."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from .panel import Panel
from .results import Estimate, Estimator, _root_mean_square, _validate_estimate


def _mean(values: np.ndarray) -> float:
    """Mean scaled to avoid overflow in summing otherwise finite values."""
    scale = float(np.max(np.abs(values)))
    return 0.0 if scale == 0.0 else float(scale * np.mean(values / scale))


def _error_scores(errors: np.ndarray) -> dict[str, float]:
    if not np.isfinite(errors).all():
        raise ValueError("Prediction errors must be finite")
    return {
        "validation_rmse": _root_mean_square(errors),
        "mae": _mean(np.abs(errors)),
        "bias": _mean(errors),
    }


@dataclass
class BacktestResult:
    """Expanding-window prediction diagnostics, scored before treatment.

    ``scores`` contains one row per candidate and fold; ``predictions`` contains
    one row per candidate, fold and held-out time. Errors and bias are observed
    minus counterfactual, for the equally weighted average of treated units.
    ``metadata['fit_metadata']`` retains every fit's complete diagnostics.

    The summary pools prediction instances, including repeated times when
    folds overlap. It is a predictive diagnostic, not a causal-validity test
    or a confidence interval for the selected estimator's treatment effect.
    """

    scores: pd.DataFrame
    predictions: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> pd.DataFrame:
        """Rank candidates by pooled validation RMSE; preserve insertion-order ties."""
        rows = []
        for name, predictions in self.predictions.groupby("model", sort=False):
            rows.append(
                {
                    "model": name,
                    **_error_scores(predictions["error"].to_numpy(dtype=float)),
                    "n_predictions": len(predictions),
                    "n_folds": predictions["fold"].nunique(),
                }
            )
        summary = pd.DataFrame(rows).set_index("model")
        summary = summary.sort_values("validation_rmse", kind="stable")
        summary.insert(0, "rank", np.arange(1, len(summary) + 1))
        return summary

    @property
    def best_model(self) -> str:
        """Candidate with minimum pooled validation RMSE, with stable ties."""
        return str(self.summary.index[0])


@dataclass
class SelectionResult:
    """Pre-treatment selection diagnostics and a fresh full-panel winner fit.

    Selection changes the analysis procedure. Ordinary estimator-specific
    intervals do not automatically account for this selection step.
    """

    validation: BacktestResult
    selected_name: str
    estimate: Estimate


def _validate_inputs(panel, estimators, initial_window, horizon, step):
    if not isinstance(panel, Panel):
        raise TypeError("panel must be a Panel")
    if not isinstance(estimators, Mapping) or not estimators:
        raise ValueError(
            "estimators must be a nonempty mapping of names to zero-argument factories"
        )
    names = list(estimators)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Estimator names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("Estimator names must be unique")
    for name, factory in estimators.items():
        if isinstance(factory, Estimate) or not callable(factory):
            raise TypeError(f"Estimator {name!r} must be a zero-argument factory")
    step = horizon if step is None else step
    for name, value, minimum in (
        ("initial_window", initial_window, 2),
        ("horizon", horizon, 1),
        ("step", step, 1),
    ):
        if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if initial_window + horizon > panel.n_pre:
        raise ValueError("initial_window + horizon must fit entirely before actual treatment")
    return int(step)


def _fresh_fit(factory, panel, instances):
    estimator = factory()
    if isinstance(estimator, Estimate) or not callable(getattr(estimator, "fit", None)):
        raise TypeError("Each factory must return a fresh estimator with a fit method")
    if any(estimator is previous for previous in instances):
        raise ValueError("Each factory call must return a fresh estimator instance")
    instances.append(estimator)
    supplied = Panel(panel.y, panel.n_control, panel.n_pre, panel.times, panel.units)
    estimate = estimator.fit(supplied)
    _validate_estimate(panel, estimate)
    return estimate


def _run_backtest(panel, estimators, initial_window, horizon, step, instances):
    step = _validate_inputs(panel, estimators, initial_window, horizon, step)
    origins = tuple(range(initial_window, panel.n_pre - horizon + 1, step))
    scores, predictions, fit_metadata = [], [], []
    observed = panel.treated
    for name, factory in estimators.items():
        for fold, origin in enumerate(origins):
            stop = origin + horizon
            values = panel.y[:stop].copy()
            values[origin:, panel.n_control :] = values[origin - 1, panel.n_control :]
            masked = Panel(values, panel.n_control, origin, panel.times[:stop], panel.units)
            try:
                estimate = _fresh_fit(factory, masked, instances)
                counterfactual = estimate.counterfactual.iloc[origin:].to_numpy(dtype=float)
                with np.errstate(over="ignore", invalid="ignore"):
                    errors = observed[origin:stop] - counterfactual
                statistics = _error_scores(errors)
                train_rmse = estimate.pre_rmse
                diagnostics = deepcopy(dict(estimate.metadata))
            except Exception as exc:
                raise RuntimeError(
                    f"Backtest failed for model {name!r}, fold {fold} (origin={origin}): {exc}"
                ) from exc
            scores.append(
                {
                    "model": name,
                    "fold": fold,
                    "origin": origin,
                    "training_start": panel.times[0],
                    "training_end": panel.times[origin - 1],
                    "validation_start": panel.times[origin],
                    "validation_end": panel.times[stop - 1],
                    "train_rmse": train_rmse,
                    **statistics,
                    "n_validation": horizon,
                }
            )
            predictions.extend(
                {
                    "model": name,
                    "fold": fold,
                    "time": time,
                    "observed": float(actual),
                    "counterfactual": float(prediction),
                    "error": float(error),
                }
                for time, actual, prediction, error in zip(
                    panel.times[origin:stop], observed[origin:stop], counterfactual, errors
                )
            )
            fit_metadata.append(
                {"model": name, "fold": fold, "method": estimate.method, "metadata": diagnostics}
            )
    return BacktestResult(
        pd.DataFrame(scores),
        pd.DataFrame(predictions),
        {
            "protocol": "expanding_window_pre_treatment",
            "initial_window": int(initial_window),
            "horizon": int(horizon),
            "step": step,
            "origins": origins,
            "n_pre": panel.n_pre,
            "n_treated": panel.n_treated,
            "target_mask": "each treated unit's last training outcome",
            "validation_controls": "observed contemporaneous donor outcomes",
            "scoring": "pooled prediction instances; repeated times retained in overlapping folds",
            "actual_post_treatment_used": False,
            "fit_metadata": fit_metadata,
        },
    )


def backtest(
    panel: Panel,
    estimators: Mapping[str, Callable[[], Estimator]],
    *,
    initial_window: int,
    horizon: int,
    step: int | None = None,
) -> BacktestResult:
    """Compare candidates using only expanding windows before actual treatment.

    Each fold trains on rows ``[:origin]`` and validates on
    ``[origin:origin + horizon]``. Only complete folds wholly within
    ``panel.n_pre`` are used; ``step`` defaults to ``horizon``. Every fit gets
    a fresh estimator and a truncated panel with ``n_pre=origin``. Each
    treated unit's validation outcomes are replaced with that unit's last
    training outcome. Actual validation targets are kept outside the fit and
    used only for scoring. No actual post-treatment row is supplied to fits.

    Validation-period donor outcomes are retained. This evaluates conditional
    counterfactual prediction, not forecasting unavailable future donors.
    Factories must be zero-argument callables returning a new estimator, with
    any random seed and fitting settings set explicitly. Settings, solver
    warnings and fit metadata are preserved. Failures stop the comparison
    with candidate and fold context; no candidate is silently omitted.
    Factories should not capture full-data predictor objects or reuse fitted
    state: the panel masking cannot prevent access to external data.

    Selection by pre-treatment prediction does not establish identification,
    eliminate treatment-effect bias, or provide post-selection inference.
    """
    return _run_backtest(panel, estimators, initial_window, horizon, step, [])


def select_estimator(
    panel: Panel,
    estimators: Mapping[str, Callable[[], Estimator]],
    *,
    initial_window: int,
    horizon: int,
    step: int | None = None,
) -> SelectionResult:
    """Select the lowest pre-treatment validation RMSE and refit the winner.

    Ties follow the mapping's insertion order. After :func:`backtest`, a fresh
    instance of the selected candidate is fitted on the original full panel.
    Its normal result and metadata remain intact. The full-panel fit uses
    post-treatment observations as required by that estimator; those outcomes
    have no role in selecting the candidate. The selected treatment effect's
    usual uncertainty estimates do not incorporate model-selection uncertainty.
    """
    instances = []
    validation = _run_backtest(panel, estimators, initial_window, horizon, step, instances)
    selected_name = validation.best_model
    try:
        estimate = _fresh_fit(estimators[selected_name], panel, instances)
        if not np.isfinite(estimate.att) or not np.isfinite(estimate.pre_rmse):
            raise ValueError("Selected estimate ATT and pre-treatment RMSE must be finite")
    except Exception as exc:
        raise RuntimeError(f"Final fit failed for selected model {selected_name!r}: {exc}") from exc
    return SelectionResult(validation, selected_name, estimate)
