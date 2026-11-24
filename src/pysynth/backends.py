"""Optional adapters to pysyncon and SyntheticControlMethods."""

import inspect
from dataclasses import dataclass, field
from importlib.metadata import version
from types import FunctionType, MethodType
from typing import Any

import numpy as np
import pandas as pd

from ._nested import StableNestedOptimizer
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


class _UpstreamOptimizationFailure(RuntimeError):
    """An upstream solver returned an unsuccessful or invalid result."""


def _recover_inner_qp(arguments):
    """Solve the identical simplex QP after upstream SLSQP reports failure."""
    import cvxpy as cp

    v = np.asarray(arguments["V_mat"])
    x, y = np.asarray(arguments["X0"]), np.asarray(arguments["X1"])
    penalty = arguments.get("lambda_", 0.0)
    # Every supported backend supplies a diagonal nonnegative predictor metric.
    if not np.allclose(v, np.diag(np.diag(v))) or np.min(np.diag(v)) < 0:
        raise ValueError("The predictor metric must be diagonal and nonnegative")
    a = np.sqrt(np.diag(v))[:, None] * (x - y[:, None])
    scale = np.max(np.abs(a))
    scale = scale if scale > 0 else 1.0
    a /= scale
    weights = cp.Variable(x.shape[1])
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(a @ weights) + penalty * np.sum(a**2, axis=0) @ weights),
        [weights >= 0, cp.sum(weights) == 1],
    )
    problem.solve(
        solver="CLARABEL",
        tol_gap_abs=1e-11,
        tol_gap_rel=1e-11,
        tol_feas=1e-11,
        max_iter=1000,
    )
    if problem.status != cp.OPTIMAL or weights.value is None:
        raise RuntimeError(f"pysyncon inner QP recovery failed: {problem.status}")
    w = np.asarray(weights.value)
    if not np.isfinite(w).all() or w.min() < -1e-8 or abs(w.sum() - 1) > 1e-8:
        raise RuntimeError("pysyncon inner QP recovery returned infeasible weights")
    w = np.maximum(w, 0)
    w /= w.sum()
    # Retain the upstream convention for its diagnostic (constant-free) loss.
    loss = 0.5 * (w @ (x.T @ v @ x) @ w) - y @ v @ x @ w
    loss += penalty / 2 * np.sum((x - y[:, None]) * (v @ (x - y[:, None])), axis=0) @ w
    return w, float(loss)


def _checked_upstream_optimizers(model, *, recover_outer=True):
    """Check upstream solver results without mutating its module globals.

    The pinned backend discards SciPy's success flag in both nested stages.
    Rebind only those functions' local global namespace to a checked minimize;
    their numerical code, starting values and stopping settings stay intact.
    """
    diagnostics = {"inner_calls": 0, "outer_calls": 0, "inner_recoveries": [], "outer_attempts": []}

    def checked(function, stage):
        original = getattr(function, "__func__", function)
        namespace = original.__globals__.copy()
        minimize = namespace["minimize"]

        def checked_minimize(*args, **kwargs):
            solved = minimize(*args, **kwargs)
            diagnostics[f"{stage}_calls"] += 1
            if stage == "outer":
                diagnostics["outer_attempts"].append(
                    {
                        "solver": kwargs.get("method"),
                        "success": bool(solved.success),
                        "message": str(solved.message),
                    }
                )
                if not solved.success and recover_outer and np.isfinite(solved.x).all():
                    # Preserve the full nested loss. User-specified iteration
                    # budgets are respected; defaults may recover with Powell.
                    solved = minimize(
                        fun=kwargs.get("fun", args[0] if args else None),
                        x0=solved.x,
                        method="Powell",
                        options={"maxiter": 10000, "maxfev": 1000000, "xtol": 1e-8, "ftol": 1e-10},
                    )
                    diagnostics["outer_attempts"].append(
                        {
                            "solver": "Powell",
                            "success": bool(solved.success),
                            "message": str(solved.message),
                        }
                    )
            if not solved.success or not np.isfinite(solved.fun) or not np.isfinite(solved.x).all():
                raise _UpstreamOptimizationFailure(
                    f"pysyncon {stage} optimization failed: {solved.message}"
                )
            if stage == "inner" and (np.min(solved.x) < -1e-8 or abs(np.sum(solved.x) - 1) > 1e-8):
                raise _UpstreamOptimizationFailure(
                    "pysyncon inner optimization returned infeasible weights"
                )
            return solved

        namespace["minimize"] = checked_minimize
        clone = FunctionType(
            original.__code__,
            namespace,
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        clone.__kwdefaults__ = original.__kwdefaults__
        return MethodType(clone, model) if inspect.ismethod(function) else clone

    if hasattr(model, "w_optimize"):
        original_inner = model.w_optimize
        verified_inner = checked(original_inner, "inner")
        signature = inspect.signature(original_inner)

        def inner(*args, **kwargs):
            try:
                return verified_inner(*args, **kwargs)
            except _UpstreamOptimizationFailure as exc:
                arguments = signature.bind(*args, **kwargs)
                arguments.apply_defaults()
                result = _recover_inner_qp(arguments.arguments)
                diagnostics["inner_recoveries"].append(
                    {
                        "solver": "CLARABEL",
                        "reason": str(exc),
                        "success": True,
                    }
                )
                return result

        model.w_optimize = inner
    if "minimize" in model.fit.__func__.__code__.co_names:
        model.fit = checked(model.fit, "outer")
    model.optimization_diagnostics_ = diagnostics


def _fit_robust(model, dataprep, panel, *, lambda_, threshold=None, sv_count=None):
    """Amjad--Shah--Shen Algorithm 1 on the complete, labelled Panel sample."""
    if not np.isfinite(lambda_) or lambda_ < 0:
        raise ValueError("Robust SCM lambda_ must be finite and nonnegative")
    if (threshold is None) == (sv_count is None):
        raise ValueError("Robust SCM requires exactly one of threshold and sv_count")
    if threshold is not None and (not np.isfinite(threshold) or threshold < 0):
        raise ValueError("Robust SCM threshold must be finite and nonnegative")
    if sv_count is not None and (
        isinstance(sv_count, bool)
        or not isinstance(sv_count, (int, np.integer))
        or not 1 <= sv_count <= min(panel.y.shape[0], panel.n_control)
    ):
        raise ValueError("Robust SCM sv_count must be a positive feasible rank")
    # Canonical integer dates are used only by the automatically built Dataprep.
    periods = list(dataprep.time_optimize_ssr)
    all_periods = sorted(dataprep.foo[dataprep.time_variable].unique())
    if len(all_periods) != len(panel.times):
        all_periods = list(panel.times)
    donors, treated = dataprep.make_outcome_mats(time_period=all_periods)
    y = donors.to_numpy()
    u, singular, vh = np.linalg.svd(y, full_matrices=False)
    keep = singular >= threshold if threshold is not None else np.arange(len(singular)) < sv_count
    # Panel rejects missing observations: an observed zero counts as observed.
    denoised = (u[:, keep] * singular[keep]) @ vh[keep]
    pre = donors.index.get_indexer(periods)
    if np.any(pre < 0):
        raise ValueError("Robust SCM fitting periods are absent from the outcome matrix")
    design = denoised[pre]
    target = treated.loc[periods].to_numpy()
    # An augmented least-squares solve is the ridge normal equation without
    # squaring its condition number; lambda_=0 permits rank-deficient designs.
    model.W = np.linalg.lstsq(
        np.vstack([design, np.sqrt(lambda_) * np.eye(panel.n_control)]),
        np.r_[target, np.zeros(panel.n_control)],
        rcond=None,
    )[0]
    model.dataprep, model.W_names = dataprep, donors.columns
    model.denoised_ = pd.DataFrame(denoised, index=donors.index, columns=donors.columns)
    model.singular_values_ = singular
    model.retained_rank_ = int(keep.sum())
    return denoised @ model.W


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
    Predictor windows must precede treatment unless an explicitly historical
    specification opts in with ``allow_post_predictors=True``. Failed inner
    simplex solves are recovered on the same quadratic objective. Failed
    default outer searches are recovered with Powell; explicit iteration
    budgets instead raise on failure. Recovery attempts are recorded in
    result metadata, without claiming global optimality of the nested search.
    """

    method: str = "synth"
    dataprep: Any = None
    options: dict = field(default_factory=dict)
    allow_post_predictors: bool = False

    def fit(self, panel: Panel) -> Estimate:
        """Fit the selected upstream implementation, preserving donor labels."""
        self.backend_ = None
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
        if self.method in {"augsynth", "pensynth"} and options.get("lambda_") is not None:
            if not np.isfinite(options["lambda_"]) or options["lambda_"] < 0:
                raise ValueError("lambda_ must be finite and nonnegative")
        if options.get("custom_V") is not None:
            metric = np.asarray(options["custom_V"], dtype=float)
            if (
                metric.ndim != 1
                or not np.isfinite(metric).all()
                or np.any(metric < 0)
                or not np.any(metric > 0)
            ):
                raise ValueError("custom_V must be a finite nonnegative vector with positive mass")
        post_predictor_periods = []
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
            order = np.arange(panel.n_control)
        else:
            dp = self.dataprep
            if (
                dp.treatment_identifier != panel.units[-1]
                or set(dp.controls_identifier) != set(panel.units[:-1])
                or len(dp.controls_identifier) != panel.n_control
            ):
                raise ValueError("Dataprep units must match the Panel")
            if list(dp.time_optimize_ssr) != list(panel.times[: panel.n_pre]):
                raise ValueError("Dataprep fitting periods must equal Panel pre-periods")
            predictor_periods = set(dp.time_predictors_prior) if dp.predictors else set()
            for _, periods, _ in dp.special_predictors or []:
                predictor_periods.update(periods)
            if not predictor_periods <= set(panel.times):
                raise ValueError("Dataprep predictor periods must be observed in the Panel")
            post_predictor_periods = [
                t for t in panel.times[panel.n_pre :] if t in predictor_periods
            ]
            if post_predictor_periods and not self.allow_post_predictors:
                raise ValueError(
                    "Dataprep predictors use post-treatment periods; historical replication "
                    "requires explicit allow_post_predictors=True"
                )
            z0, z1 = dp.make_outcome_mats(time_period=list(panel.times))
            if not np.array_equal(
                z0.loc[:, list(panel.units[:-1])], panel.control
            ) or not np.array_equal(z1, panel.treated):
                raise ValueError("Dataprep outcomes must match the Panel")
            order = np.array([list(dp.controls_identifier).index(u) for u in panel.units[:-1]])
        model = classes[self.method]()
        _checked_upstream_optimizers(model, recover_outer="optim_options" not in options)
        if self.method == "augsynth" and self.dataprep is None:
            if "use_covariates" in inspect.signature(model.fit).parameters:
                options.setdefault("use_covariates", False)
        counterfactual = None
        if self.method == "robust":
            counterfactual = _fit_robust(model, dp, panel, **options)
        else:
            model.fit(dataprep=dp, **options)
        self.backend_ = model
        weights = np.asarray(model.W).reshape(-1)[order]
        return make_estimate(
            panel,
            f"pysyncon.{self.method}",
            weights,
            counterfactual=counterfactual,
            metadata={
                "backend_version": version("pysyncon"),
                "options": options,
                "optimization": model.optimization_diagnostics_,
                "post_predictor_periods": post_predictor_periods,
                **(
                    {
                        "denoised_prediction": True,
                        "observed_fraction": 1.0,
                        "retained_rank": model.retained_rank_,
                    }
                    if self.method == "robust"
                    else {}
                ),
            },
        )


@dataclass
class SCM:
    """Adapter for SyntheticControlMethods 1.1.17's Synth and DiffSynth.

    ``method`` is ``synth`` or ``diffsynth``. Pre-period outcomes enter as
    repeated, precomputed lag predictors. ``n_lags=None`` uses every pre-period;
    an integer explicitly restricts the predictor specification. All ten upstream
    optimization starts are used by default. Failed gradient line searches are
    recovered with a bounded derivative-free solve of the same nested objective.
    The DiffSynth subclass fixes pandas 2 groupby-index alignment and computes
    differences within units; no installed package or global state is changed.
    Both subclasses also correct predictor-axis scaling and target-array
    broadcasting in the released package's inner objective.
    These compatibility choices are recorded in result metadata.
    """

    method: str = "synth"
    pen: float = 0.0
    random_state: int = 0
    n_lags: int | None = None
    n_optim: int = 10
    max_iter: int = 10000

    def fit(self, panel: Panel) -> Estimate:
        """Fit the upstream model on a canonical integer-indexed panel."""
        upstream = _require("SyntheticControlMethods")
        if self.method not in {"synth", "diffsynth"}:
            raise ValueError("method must be synth or diffsynth")
        if panel.n_treated != 1:
            raise ValueError("SCM requires exactly one treated unit")
        if not np.isfinite(self.pen) or self.pen < 0:
            raise ValueError("pen must be finite and nonnegative")
        for name in ("n_optim", "max_iter"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
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
            frame[f"lag_{t}"] = frame.unit.map(dict(enumerate(values)))

        class CovariateShapeMixin:
            def _rescale_covariate_variance(
                self, treated_covariates, control_covariates, n_covariates
            ):
                combined = np.concatenate([treated_covariates, control_covariates], axis=1)
                # Rows are predictors: standardize across units, not across
                # differently measured predictors within each unit.
                centered = combined - combined[:, :1]
                scale = centered.std(axis=1, keepdims=True)
                scale[scale == 0] = 1.0
                standardized = centered / scale
                return standardized[:, :1], standardized[:, 1:]

            def _process_input_data(self, *args, **kwargs):
                prepared = super()._process_input_data(*args, **kwargs)
                # 1.1.17 returns a column here but its loss transposes it.
                # Passing the intended row avoids a p-by-p broadcast residual.
                prepared["treated_covariates"] = prepared["treated_covariates"].T
                return prepared

        class CompatibleSynth(StableNestedOptimizer, CovariateShapeMixin, upstream.Synth):
            outer_max_iter = self.max_iter

        class CompatibleDiffSynth(StableNestedOptimizer, CovariateShapeMixin, upstream.DiffSynth):
            outer_max_iter = self.max_iter

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
            n_optim=self.n_optim,
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
                "n_optim": self.n_optim,
                "optimization": model.optimization_diagnostics_,
                "predictor_shape_fix": True,
                "predictor_scaling_fix": True,
                "lag_periods": lag_times.tolist(),
                "pandas_difference_fix": self.method == "diffsynth",
            },
        )
