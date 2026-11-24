"""Common result objects and comparison helpers."""

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd

from .panel import Panel


def _root_mean_square(values) -> float:
    """Compute RMS without squaring residuals in their original units."""
    values = np.asarray(values, dtype=float)
    if not values.size:
        return float("nan")
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return 0.0
    return float(scale * np.sqrt(np.mean((values / scale) ** 2)))


def _mean_effect(values) -> float:
    """Average a finite, nonempty effect window, scaling only if its sum overflows."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("A mean effect requires a nonempty window of finite effects")
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(values.mean())
    if np.isfinite(mean):
        return mean
    scale = np.max(np.abs(values))
    return float((values / scale).mean() * scale)


def _real_series_values(series: pd.Series, name: str) -> np.ndarray:
    if not isinstance(series, pd.Series):
        raise TypeError(f"Estimate {name} must be a labelled Series")
    if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_complex_dtype(
        series.dtype
    ):
        raise TypeError(f"Estimate {name} must contain real numeric values")
    values = series.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise ValueError(f"Estimate {name} must be finite")
    return values


class Estimator(Protocol):
    """An estimator consumes a Panel and returns an Estimate."""

    def fit(self, panel: Panel) -> "Estimate": ...


@dataclass
class Estimate:
    """Observed and synthetic paths in the original outcome units.

    ``effects`` is observed minus counterfactual. For SDID the counterfactual
    includes its weighted pre-treatment level correction. ``att`` averages
    post-treatment effects; it is not an uncertainty estimate.
    """

    method: str
    observed: pd.Series
    counterfactual: pd.Series
    weights: pd.Series
    n_pre: int
    intercept: float = 0.0
    time_weights: pd.Series | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def effects(self) -> pd.Series:
        """Time-specific observed minus counterfactual outcomes."""
        observed = _real_series_values(self.observed, "observed")
        counterfactual = _real_series_values(self.counterfactual, "counterfactual")
        times = self.observed.index
        if not times.is_unique or times.hasnans or not times.is_monotonic_increasing:
            raise ValueError("Estimate time labels must be unique, nonmissing and increasing")
        if not times.equals(self.counterfactual.index):
            raise ValueError(
                "Estimate counterfactual labels and order must match observed outcomes"
            )
        if (
            isinstance(self.n_pre, bool)
            or not isinstance(self.n_pre, Integral)
            or not 1 <= self.n_pre < len(times)
        ):
            raise ValueError("Estimate timing must leave both pre- and post-treatment observations")
        # Integer series can encode adjacent values above 2**53, and their
        # difference can exceed int64. Subtract those integers before casting.
        if all(
            pd.api.types.is_integer_dtype(s.dtype) for s in (self.observed, self.counterfactual)
        ):
            values = (self.observed.astype(object) - self.counterfactual.astype(object)).to_numpy(
                dtype=float
            )
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                values = observed - counterfactual
        if not np.isfinite(values).all():
            raise ValueError("Estimate differences exceed finite floating-point range")
        return pd.Series(values, index=times, name="effect")

    @property
    def att(self) -> float:
        """Mean effect over all post-treatment periods."""
        return _mean_effect(self.effects.iloc[self.n_pre :])

    @property
    def pre_rmse(self) -> float:
        """Root mean squared pre-treatment discrepancy."""
        return _root_mean_square(self.effects.iloc[: self.n_pre])

    def to_frame(self) -> pd.DataFrame:
        """Export paths and the post-treatment indicator."""
        return pd.DataFrame(
            {
                "observed": self.observed,
                "counterfactual": self.counterfactual,
                "effect": self.effects,
                "post": np.arange(len(self.observed)) >= self.n_pre,
            }
        )

    def plot(self, *, ax=None, effects: bool = False, **kwargs):
        """Plot publication-style paths or effects and return a Matplotlib Axes.

        Additional keyword arguments go to :func:`pysynth.plot_paths`, including
        ``title``, ``ylabel``, ``xlabel``, and ``intervention_label``. Requires
        the optional ``plot`` extra. For estimator comparisons, use
        :func:`pysynth.plot_effects` or :func:`pysynth.plot_comparison`.
        """
        from .plotting import plot_paths

        return plot_paths(self, ax=ax, effects=effects, **kwargs)


def make_estimate(
    panel, method, weights, intercept=0.0, time_weights=None, metadata=None, counterfactual=None
):
    """Internal result constructor, retaining label alignment."""
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if len(weights) != panel.n_control or not np.isfinite(weights).all():
        raise ValueError("Invalid fitted donor weights")
    cf = (
        panel.control @ weights + intercept
        if counterfactual is None
        else np.asarray(counterfactual)
    )
    if cf.shape != panel.treated.shape or not np.isfinite(cf).all():
        raise ValueError("Invalid fitted counterfactual")
    return Estimate(
        method,
        pd.Series(panel.treated, index=panel.times, name="observed"),
        pd.Series(cf, index=panel.times, name="counterfactual"),
        pd.Series(weights, index=panel.units[: panel.n_control], name="weight"),
        panel.n_pre,
        float(intercept),
        None
        if time_weights is None
        else pd.Series(time_weights, index=panel.times[: panel.n_pre], name="time_weight"),
        metadata or {},
    )


def _validate_estimate(
    panel: Panel | None, estimate: Estimate, *, allow_subset_weights: bool = False
) -> None:
    """Check the labelled result and, when supplied, the data it purports to fit."""
    if not isinstance(estimate, Estimate):
        raise TypeError("fit must return an Estimate")
    # Calculating the path checks period alignment, timing and finite differences.
    _ = estimate.effects
    _real_series_values(estimate.weights, "weights")
    weights = estimate.weights.index
    if not len(weights) or not weights.is_unique or pd.Index(weights, tupleize_cols=False).hasnans:
        raise ValueError("Donor weights require unique, nonmissing unit labels")
    if panel is not None:
        if estimate.n_pre != panel.n_pre or not estimate.observed.index.equals(panel.times):
            raise ValueError(
                "Estimate observed time labels or treatment timing differ from the fitted panel"
            )
        if not np.array_equal(estimate.observed.to_numpy(), panel.treated):
            raise ValueError("Estimate observed outcomes do not match the fitted panel")
        donors = pd.Index(panel.units[: panel.n_control], tupleize_cols=False)
        if allow_subset_weights:
            if not weights.isin(donors).all():
                raise ValueError("Estimate weights contain donors outside the fitted panel")
        elif not weights.equals(donors):
            raise ValueError("Estimate donor weights and order must match the fitted panel")
    if not isinstance(estimate.intercept, Real) or not np.isfinite(float(estimate.intercept)):
        raise ValueError("Estimate intercept must be finite and real")
    if estimate.time_weights is not None:
        _real_series_values(estimate.time_weights, "time weights")
        if not estimate.time_weights.index.equals(estimate.observed.index[: estimate.n_pre]):
            raise ValueError("Estimate time weights must match pre-treatment labels and order")
    if not isinstance(estimate.metadata, Mapping):
        raise TypeError("Estimate metadata must be a mapping")


def compare(models: Mapping[str, Estimator | Estimate], panel: Panel | None = None) -> pd.DataFrame:
    """Compare fitted estimates or fit named estimators on a common panel.

    Failures propagate; methods are never silently omitted. Models may use
    different estimands or covariates. A supplied panel also validates prefitted
    results against its observed outcomes, timing and donor labels. The
    ``donors_used`` count uses absolute donor weight greater than 1e-6.
    """
    rows = []
    reference = None
    for name, model in models.items():
        if not isinstance(model, Estimate):
            if panel is None:
                raise ValueError("panel is required when comparing unfitted estimators")
            model = model.fit(panel)
        _validate_estimate(panel, model, allow_subset_weights=True)
        if reference is not None and (
            model.n_pre != reference.n_pre
            or not model.observed.index.equals(reference.observed.index)
            or not np.array_equal(model.observed.to_numpy(), reference.observed.to_numpy())
        ):
            raise ValueError("Compared estimates must share observed outcomes and treatment timing")
        reference = model
        rows.append(
            {
                "model": name,
                "att": model.att,
                "pre_rmse": model.pre_rmse,
                "donors_used": int((model.weights.abs() > 1e-6).sum()),
                "weight_sum": float(model.weights.sum()),
            }
        )
    if not rows:
        raise ValueError("models must not be empty")
    return pd.DataFrame(rows).set_index("model")
