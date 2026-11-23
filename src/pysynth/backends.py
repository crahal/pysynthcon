"""Optional adapters to pysyncon and SyntheticControlMethods."""

import inspect
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any

import numpy as np
import pandas as pd

from .panel import Panel
from .results import Estimate, make_estimate


def _require(module):
    try:
        return __import__(module)
    except ImportError as exc:
        raise ImportError(
            "Install PySynth's optional backends: pip install -e '.[backends]'"
        ) from exc


def _canonical_frame(panel):
    return pd.DataFrame(
        {
            "unit": np.repeat(np.arange(panel.y.shape[1]), len(panel.y)),
            "time": np.tile(np.arange(len(panel.y)), panel.y.shape[1]),
            "outcome": panel.y.T.ravel(),
        }
    )


@dataclass
class PySyncon:
    """Adapt Synth, AugSynth, PenalizedSynth, or RobustSynth to a common result.

    ``method`` is ``synth``, ``augsynth``, ``pensynth``, or ``robust``.
    ``options`` are passed to the backend's ``fit``. ``dataprep`` optionally
    supplies a pysyncon.Dataprep, including published predictor specifications.
    It must use exactly the panel's labels, outcomes, and pre-periods. Without
    it, synth and pensynth match pre-outcomes as predictors; augsynth uses no
    extra covariates. Robust requires explicit ``lambda_`` and ``sv_count``
    or ``threshold`` in options. The fitted upstream object is ``backend_``.
    """

    method: str = "synth"
    dataprep: Any = None
    options: dict = field(default_factory=dict)

    def fit(self, panel: Panel) -> Estimate:
        """Fit the selected upstream implementation, preserving donor labels."""
        upstream = _require("pysyncon")
        classes = {
            "synth": upstream.Synth,
            "augsynth": upstream.AugSynth,
            "pensynth": upstream.PenalizedSynth,
            "robust": upstream.RobustSynth,
        }
        if self.method not in classes:
            raise ValueError(f"method must be one of {list(classes)}")
        if panel.n_treated != 1:
            raise ValueError("These upstream adapters require one treated unit")
        options = dict(self.options)
        if self.dataprep is None:
            frame = _canonical_frame(panel)
            dp = upstream.Dataprep(
                foo=frame,
                predictors=[],
                predictors_op="mean",
                dependent="outcome",
                unit_variable="unit",
                time_variable="time",
                treatment_identifier=panel.n_control,
                controls_identifier=list(range(panel.n_control)),
                time_predictors_prior=list(range(panel.n_pre)),
                time_optimize_ssr=list(range(panel.n_pre)),
                special_predictors=[("outcome", [t], "mean") for t in range(panel.n_pre)]
                if self.method in {"synth", "pensynth"}
                else None,
            )
            if self.method in {"synth", "pensynth"}:
                options.setdefault("custom_V", np.ones(panel.n_pre))
            order = np.arange(panel.n_control)
        else:
            dp = self.dataprep
            if dp.treatment_identifier != panel.units[-1] or set(dp.controls_identifier) != set(
                panel.units[:-1]
            ):
                raise ValueError("Dataprep units must match the Panel")
            if list(dp.time_optimize_ssr) != list(panel.times[: panel.n_pre]):
                raise ValueError("Dataprep fitting periods must equal Panel pre-periods")
            z0, z1 = dp.make_outcome_mats(time_period=list(panel.times))
            if not np.allclose(z0.loc[:, list(panel.units[:-1])], panel.control) or not np.allclose(
                z1, panel.treated
            ):
                raise ValueError("Dataprep outcomes must match the Panel")
            order = np.array([list(dp.controls_identifier).index(u) for u in panel.units[:-1]])
        model = classes[self.method]()
        if self.method == "augsynth" and self.dataprep is None:
            if "use_covariates" in inspect.signature(model.fit).parameters:
                options.setdefault("use_covariates", False)
        model.fit(dataprep=dp, **options)
        self.backend_ = model
        weights = np.asarray(model.W).reshape(-1)[order]
        return make_estimate(
            panel,
            f"pysyncon.{self.method}",
            weights,
            metadata={"backend_version": version("pysyncon"), "options": options},
        )


@dataclass
class SCM:
    """Adapter for SyntheticControlMethods 1.1.17's Synth and DiffSynth.

    ``method`` is ``synth`` or ``diffsynth``. Pre-period outcomes enter as
    repeated, precomputed lag predictors. ``n_lags=None`` uses every pre-period;
    an integer uses at most that many evenly spaced pre-periods. One optimisation start
    avoids the release's malformed random-start shape on modern SciPy. The
    DiffSynth subclass fixes pandas 2 groupby-index alignment and computes
    differences within units; no installed package or global state is changed.
    Both subclasses also correct predictor-axis scaling and target-array
    broadcasting in the released package's inner objective.
    These compatibility choices are recorded in result metadata.
    """

    method: str = "synth"
    pen: float = 0.0
    random_state: int = 0
    n_lags: int | None = None

    def fit(self, panel: Panel) -> Estimate:
        """Fit the upstream model on a canonical integer-indexed panel."""
        upstream = _require("SyntheticControlMethods")
        if self.method not in {"synth", "diffsynth"}:
            raise ValueError("method must be synth or diffsynth")
        if panel.n_treated != 1:
            raise ValueError("SCM requires exactly one treated unit")
        if not np.isfinite(self.pen) or self.pen < 0:
            raise ValueError("pen must be finite and nonnegative")
        if self.n_lags is not None and (
            not isinstance(self.n_lags, (int, np.integer)) or self.n_lags < 1
        ):
            raise ValueError("n_lags must be a positive integer or None")
        frame = _canonical_frame(panel)
        # Constant-by-unit lag columns are based solely on observed pre-outcomes.
        lag_times = (
            np.arange(panel.n_pre)
            if self.n_lags is None
            else np.unique(
                np.linspace(0, panel.n_pre - 1, min(self.n_lags, panel.n_pre), dtype=int)
            )
        )
        for t in lag_times:
            values = panel.y[t]
            if values.std() > 1e-12:
                frame[f"lag_{t}"] = frame.unit.map(dict(enumerate(values)))

        class CovariateShapeMixin:
            def _rescale_covariate_variance(
                self, treated_covariates, control_covariates, n_covariates
            ):
                combined = np.concatenate([treated_covariates, control_covariates], axis=1)
                # Rows are predictors: standardize across units, not across
                # differently measured predictors within each unit.
                scale = combined.std(axis=1, keepdims=True)
                scale[scale <= np.finfo(float).eps] = 1.0
                standardized = combined / scale
                return standardized[:, :1], standardized[:, 1:]

            def _process_input_data(self, *args, **kwargs):
                prepared = super()._process_input_data(*args, **kwargs)
                # 1.1.17 returns a column here but its loss transposes it.
                # Passing the intended row avoids a p-by-p broadcast residual.
                prepared["treated_covariates"] = prepared["treated_covariates"].T
                return prepared

        class CompatibleSynth(CovariateShapeMixin, upstream.Synth):
            pass

        class CompatibleDiffSynth(CovariateShapeMixin, upstream.DiffSynth):
            def difference_data(self, dataset, not_diff_cols):
                transformed = dataset.copy()
                transformed["outcome"] = transformed.groupby("unit", sort=False).outcome.diff()
                return transformed.loc[transformed.time > 0].reset_index(drop=True)

        cls = CompatibleSynth if self.method == "synth" else CompatibleDiffSynth
        frame["unit"] = frame.unit.map(lambda u: f"unit_{u:08d}")
        model = cls(
            frame,
            "outcome",
            "unit",
            "time",
            panel.n_pre,
            f"unit_{panel.n_control:08d}",
            n_optim=1,
            pen=self.pen,
            random_seed=self.random_state,
        )
        self.backend_ = model
        data = model.original_data
        weights = np.asarray(data.w).reshape(-1)
        offset = float(data.synth_constant) if self.method == "diffsynth" else 0.0
        return make_estimate(
            panel,
            f"SyntheticControlMethods.{self.method}",
            weights,
            offset,
            metadata={
                "backend_version": version("SyntheticControlMethods"),
                "n_optim": 1,
                "predictor_shape_fix": True,
                "predictor_scaling_fix": True,
                "lag_periods": lag_times.tolist(),
                "pandas_difference_fix": self.method == "diffsynth",
            },
        )
