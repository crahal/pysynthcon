"""Common result objects and comparison helpers."""

from dataclasses import dataclass, field
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
        return (self.observed - self.counterfactual).rename("effect")

    @property
    def att(self) -> float:
        """Mean effect over all post-treatment periods."""
        return float(self.effects.iloc[self.n_pre :].mean())

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


def compare(models: Mapping[str, Estimator | Estimate], panel: Panel | None = None) -> pd.DataFrame:
    """Compare fitted estimates or fit named estimators on a common panel.

    Failures propagate; methods are never silently omitted. Models may use
    different estimands or covariates: see the method documentation.
    """
    rows = []
    reference = None
    for name, model in models.items():
        if not isinstance(model, Estimate):
            if panel is None:
                raise ValueError("panel is required when comparing unfitted estimators")
            model = model.fit(panel)
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
