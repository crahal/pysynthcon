"""Published synthetic-control estimators and compatibility adapters."""

import inspect
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import KW_ONLY, dataclass, field
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
            "Install PySynthCon's optional backends: pip install -e '.[backends]'"
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


def _feasible_improvement(weights, arguments):
    """Attainable decrease on the best-gradient vertex direction (half-loss units)."""
    # On the simplex we can center every donor on the treated predictor.
    # The upstream expanded loss can lose all finite-difference precision
    # for large common levels, even while reporting successful termination.
    delta = arguments["X0"] - arguments["X1"][:, None]
    v = arguments["V_mat"]
    residual = delta @ weights
    gradient = delta.T @ (v @ residual)
    penalty = arguments.get("lambda_", 0.0)
    if penalty:
        gradient += 0.5 * penalty * np.sum(delta * (v @ delta), axis=0)
    vertex = int(np.argmin(gradient))
    gap = max(0.0, float(weights @ gradient - gradient[vertex]))
    change = delta[:, vertex] - residual
    curvature = float(change @ (v @ change))
    step = min(1.0, gap / curvature) if curvature > 0 else 1.0
    return step * (gap - 0.5 * step * curvature)


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
            arguments = signature.bind(*args, **kwargs)
            arguments.apply_defaults()
            try:
                weights, loss = verified_inner(*args, **kwargs)
                improvement = _feasible_improvement(weights, arguments.arguments)
                ftol = arguments.arguments["qp_options"].get("ftol", 1e-6)
                if improvement > ftol:
                    raise _UpstreamOptimizationFailure(
                        f"feasible simplex step decreases inner loss by {improvement:g} "
                        f"(solver tolerance {ftol:g})"
                    )
                return weights, loss
            except _UpstreamOptimizationFailure as exc:
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


def _fit_augmented(model, dataprep, options):
    """Preserve completed ridge fits when a scaled imbalance has a zero denominator."""
    if options.get("residualize", False) and options.get("use_covariates", True):
        covariates, _ = dataprep.make_covariate_mats()
        centered = covariates.subtract(covariates.mean(axis=1), axis=0).to_numpy()
        if centered.size and np.linalg.matrix_rank(centered) < len(centered):
            raise ValueError(
                "Residualized AugSynth requires linearly independent centered covariates"
            )
    if options.get("lambda_") == 0:
        original_solve = model.solve_ridge

        def solve_ridge(A, B, W, lambda_):
            # The reference inverts B B' directly. A singular Gram matrix
            # can have a floating-point inverse with enormous spurious
            # entries, returning weights whose sum differs greatly from one.
            if lambda_ == 0 and np.linalg.matrix_rank(B) < B.shape[0]:
                raise ValueError(
                    "AugSynth with lambda_=0 requires a full-row-rank centered design; "
                    "choose a positive ridge penalty for this panel"
                )
            return original_solve(A, B, W, lambda_)

        model.solve_ridge = solve_ridge
    try:
        model.fit(dataprep=dataprep, **options)
    except ZeroDivisionError:
        # Pinned AugSynth computes these descriptive ratios after completing
        # all weight, outcome-model and bias calculations. Handle only a
        # demonstrably zero diagnostic denominator, leaving fit errors visible.
        if model.unif_l2_imbalance is None:
            raise
        if model.unif_l2_imbalance == 0:
            model.scaled_l2_imbalance = np.nan
        if not options.get("use_covariates", True):
            if model.unif_l2_imbalance != 0:
                raise
            return
        z0, z1 = dataprep.make_covariate_mats()
        if options.get("residualize", False):
            center = z0.mean(axis=1)
            z0, z1 = z0.subtract(center, axis=0), z1 - center
        else:
            x0, x1 = dataprep.make_outcome_mats()
            _, _, z0, z1 = model._normalize(x0, x1, z0, z1)
        uniform = np.full(len(model.W), 1 / len(model.W))
        denominator = np.linalg.norm(z0.to_numpy() @ uniform - z1.to_numpy())
        if model.unif_l2_imbalance != 0 and denominator != 0:
            raise
        imbalance = np.linalg.norm(z0.to_numpy() @ model.W - z1.to_numpy())
        model.covariate_l2_imbalance = float(imbalance)
        model.scaled_covariate_l2_imbalance = (
            float(imbalance / denominator) if denominator else np.nan
        )


def _robust_weights(donors, treated_pre, *, lambda_, threshold=None, sv_count=None):
    """Algorithm 1 of Amjad, Shah and Shen (2018) for a complete donor matrix."""
    if not np.isfinite(lambda_) or lambda_ < 0:
        raise ValueError("Robust SCM lambda_ must be finite and nonnegative")
    if (threshold is None) == (sv_count is None):
        raise ValueError("Robust SCM requires exactly one of threshold and sv_count")
    if threshold is not None and (not np.isfinite(threshold) or threshold < 0):
        raise ValueError("Robust SCM threshold must be finite and nonnegative")
    if sv_count is not None and (
        isinstance(sv_count, (bool, np.bool_))
        or not isinstance(sv_count, (int, np.integer))
        or not 1 <= sv_count <= min(donors.shape)
    ):
        raise ValueError("Robust SCM sv_count must be a positive feasible rank")
    u, singular, vh = np.linalg.svd(donors, full_matrices=False)
    keep = singular >= threshold if threshold is not None else np.arange(len(singular)) < sv_count
    # The observation fraction is one, including when a recorded outcome is zero.
    denoised = (u[:, keep] * singular[keep]) @ vh[keep]
    design = denoised[: len(treated_pre)]
    # This least-squares form also handles rank deficiency at zero penalty.
    weights = np.linalg.lstsq(
        np.vstack([design, np.sqrt(lambda_) * np.eye(donors.shape[1])]),
        np.r_[treated_pre, np.zeros(donors.shape[1])],
        rcond=None,
    )[0]
    return weights, denoised, singular, int(keep.sum())


@dataclass
class RobustSyntheticControl:
    """Amjad, Shah and Shen's (2018) robust synthetic control, Algorithm 1.

    Supply a nonnegative ridge penalty (the paper's eta) and either the
    singular-value threshold (mu) or a retained rank. All donor periods enter
    the SVD. Only pretreatment treated outcomes enter the ridge regression.
    The fitted path uses the denoised donor trajectories for every period.

    This implementation operates on complete panels. The observation fraction
    in the paper's equation (9) is one. See the method paper at
    https://jmlr.org/papers/v19/17-777.html.
    """

    lambda_: float
    _: KW_ONLY
    sv_count: int | None = None
    threshold: float | None = None

    def fit(self, panel: Panel) -> Estimate:
        """Fit unconstrained donor weights and return the denoised counterfactual."""
        if panel.n_treated != 1:
            raise ValueError("Robust synthetic control requires one treated unit")
        weights, denoised, singular, rank = _robust_weights(
            panel.control,
            panel.treated[: panel.n_pre],
            lambda_=self.lambda_,
            threshold=self.threshold,
            sv_count=self.sv_count,
        )
        return make_estimate(
            panel,
            "Robust SCM",
            weights,
            counterfactual=denoised @ weights,
            metadata={
                "method_reference": "https://jmlr.org/papers/v19/17-777.html",
                "denoised_prediction": True,
                "observed_fraction": 1.0,
                "retained_rank": rank,
                "singular_values": singular.tolist(),
                "options": {
                    "lambda_": self.lambda_,
                    "sv_count": self.sv_count,
                    "threshold": self.threshold,
                },
            },
        )


def _window_positions(panel, windows):
    """Map outcome-mean predictor windows from Panel time labels to internal rows."""
    if isinstance(windows, (str, bytes)) or not pd.api.types.is_list_like(windows):
        raise ValueError("predictor_windows must be a sequence of nonempty time-label sequences")
    pre = panel.times[: panel.n_pre]
    positions = []
    for window in windows:
        if isinstance(window, (str, bytes)) or not pd.api.types.is_list_like(window):
            raise ValueError("Each predictor window must be a sequence of time labels")
        labels = list(window)
        rows = pre.get_indexer(labels)
        if not labels or np.any(rows < 0) or len(set(rows)) != len(labels):
            raise ValueError(
                "Each predictor window must contain distinct observed pretreatment dates"
            )
        positions.append(rows.tolist())
    if not positions:
        raise ValueError("predictor_windows must contain at least one window")
    return positions


class _PredictorBackend:
    def _fit(
        self,
        panel,
        method,
        label,
        options,
        dataprep,
        allow_post_predictors=False,
        predictor_windows=None,
    ):
        """Fit a checked predictor specification with an optional external implementation."""
        self.backend_ = None
        if not isinstance(allow_post_predictors, (bool, np.bool_)):
            raise ValueError("allow_post_predictors must be a boolean")
        upstream = _require("pysyncon")
        classes = {
            "synth": upstream.Synth,
            "augsynth": upstream.AugSynth,
            "pensynth": upstream.PenalizedSynth,
            "robust": upstream.RobustSynth,
        }
        if method not in classes:
            raise ValueError(f"method must be one of {list(classes)}")
        if panel.n_treated != 1:
            raise ValueError("These upstream adapters require one treated unit")
        options = deepcopy(options)
        if method in {"augsynth", "pensynth"} and "lambda_" in options:
            penalty = options["lambda_"]
            if (method == "pensynth" and penalty is None) or (
                penalty is not None and (not np.isfinite(penalty) or penalty < 0)
            ):
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
        if predictor_windows is not None and dataprep is not None:
            raise ValueError("Choose either predictor_windows or dataprep")
        if dataprep is None:
            windows = (
                [[t] for t in range(panel.n_pre)]
                if predictor_windows is None
                else _window_positions(panel, predictor_windows)
            )
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
                special_predictors=[("outcome", window, "mean") for window in windows]
                if method in {"synth", "pensynth"}
                else None,
            )
            order = np.arange(panel.n_control)
        else:
            dp = dataprep
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
            if post_predictor_periods and not allow_post_predictors:
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
        if options.get("custom_V") is not None:
            predictor_count = len(dp.predictors) + len(dp.special_predictors or [])
            if len(metric) != predictor_count:
                raise ValueError("custom_V must have one diagonal weight per predictor")
        model = classes[method]()
        _checked_upstream_optimizers(model, recover_outer="optim_options" not in options)
        if method == "augsynth" and dataprep is None:
            if "use_covariates" in inspect.signature(model.fit).parameters:
                options.setdefault("use_covariates", False)
        counterfactual = None
        if method == "robust":
            periods = list(panel.times) if dataprep is not None else list(range(len(panel.times)))
            donors, treated = dp.make_outcome_mats(time_period=periods)
            model.W, denoised, model.singular_values_, model.retained_rank_ = _robust_weights(
                donors.to_numpy(), treated.to_numpy()[: panel.n_pre], **options
            )
            model.dataprep, model.W_names = dp, donors.columns
            model.denoised_ = pd.DataFrame(denoised, index=donors.index, columns=donors.columns)
            counterfactual = denoised @ model.W
        elif method == "augsynth":
            _fit_augmented(model, dp, options)
        else:
            model.fit(dataprep=dp, **options)
        weights = np.asarray(model.W).reshape(-1)[order]
        result = make_estimate(
            panel,
            label,
            weights,
            counterfactual=counterfactual,
            metadata={
                "backend_package": "pysyncon",
                "backend_method": method,
                "backend_version": version("pysyncon"),
                "options": options,
                "optimization": model.optimization_diagnostics_,
                "post_predictor_periods": post_predictor_periods,
                **(
                    {"predictor_windows": [panel.times.take(w).tolist() for w in windows]}
                    if predictor_windows is not None
                    else {}
                ),
                **(
                    {
                        "denoised_prediction": True,
                        "observed_fraction": 1.0,
                        "retained_rank": model.retained_rank_,
                    }
                    if method == "robust"
                    else {}
                ),
            },
        )
        self.backend_ = model
        return result


@dataclass
class PredictorSyntheticControl(_PredictorBackend):
    """Predictor-weighted synthetic control.

    Follows Abadie and Gardeazabal (2003) and Abadie, Diamond and Hainmueller (2010).

    Donor weights balance predictors on the simplex. Predictor importance is
    selected by pretreatment outcome fit unless a nonnegative diagonal metric
    is supplied in ``custom_V``. Use ``predictor_windows`` for one mean outcome
    per window of Panel time labels. By default each pretreatment outcome is
    its own predictor. For other covariates, supply a labelled Dataprep.
    """

    dataprep: Any = None
    _: KW_ONLY
    predictor_windows: Sequence[Sequence[Any]] | None = None
    custom_V: Sequence[float] | None = None
    optim_method: str = "Nelder-Mead"
    optim_initial: str = "equal"
    optim_options: dict | None = None
    allow_post_predictors: bool = False

    def fit(self, panel: Panel) -> Estimate:
        """Fit the specified predictor metric and simplex donor weights."""
        options = {"optim_method": self.optim_method, "optim_initial": self.optim_initial}
        if self.custom_V is not None:
            options["custom_V"] = self.custom_V
        if self.optim_options is not None:
            options["optim_options"] = self.optim_options
        result = self._fit(
            panel,
            "synth",
            "Predictor SCM",
            options,
            self.dataprep,
            self.allow_post_predictors,
            self.predictor_windows,
        )
        result.metadata["method_reference"] = "https://doi.org/10.1198/jasa.2009.ap08746"
        return result


@dataclass
class AugmentedSyntheticControl(_PredictorBackend):
    """Ridge-augmented synthetic control of Ben-Michael, Feller and Rothstein (2021).

    A ridge outcome model corrects imbalance remaining after the simplex fit.
    ``lambda_=None`` selects the ridge penalty by pretreatment cross-validation.
    The default fits outcome histories. A Dataprep can add covariates, either
    balanced jointly or residualized before the ridge fit.
    """

    lambda_: float | None = None
    _: KW_ONLY
    dataprep: Any = None
    use_covariates: bool | None = None
    residualize: bool = False
    allow_post_predictors: bool = False

    def fit(self, panel: Panel) -> Estimate:
        """Fit the ridge augmentation and return the bias-adjusted path."""
        available = self.dataprep is not None and bool(
            len(self.dataprep.predictors) or self.dataprep.special_predictors
        )
        covariates = available if self.use_covariates is None else self.use_covariates
        if covariates and not available:
            self.backend_ = None
            raise ValueError("use_covariates=True requires covariates in dataprep")
        options = {
            "lambda_": self.lambda_,
            "use_covariates": covariates,
            "residualize": self.residualize,
        }
        result = self._fit(
            panel,
            "augsynth",
            "Ridge-augmented SCM",
            options,
            self.dataprep,
            self.allow_post_predictors,
        )
        result.metadata["method_reference"] = "https://doi.org/10.1080/01621459.2021.1929245"
        result.metadata["ridge_penalty"] = float(self.backend_.lambda_)
        return result


@dataclass
class PenalizedSyntheticControl(_PredictorBackend):
    """Abadie and L'Hour's (2021) penalized SCM for one treated unit.

    The nonnegative penalty multiplies the weighted sum of the individual
    donor-to-treated squared predictor distances. The remaining loss is the
    squared discrepancy of the synthetic predictor profile. The default
    metric equally weights standardized predictors; ``custom_V`` provides
    diagonal metric entries. Set the penalty explicitly in the units of
    this objective.
    """

    lambda_: float
    _: KW_ONLY
    dataprep: Any = None
    custom_V: Sequence[float] | None = None
    allow_post_predictors: bool = False

    def fit(self, panel: Panel) -> Estimate:
        """Fit simplex donor weights with a pairwise discrepancy penalty."""
        options = {"lambda_": self.lambda_}
        if self.custom_V is not None:
            options["custom_V"] = self.custom_V
        result = self._fit(
            panel, "pensynth", "Penalized SCM", options, self.dataprep, self.allow_post_predictors
        )
        result.metadata["method_reference"] = "https://doi.org/10.1080/01621459.2021.1971535"
        return result


@dataclass
class PySyncon(_PredictorBackend):
    """Compatibility interface for selecting an estimator by its backend identifier.

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
    result metadata. A feasible step that improves the inner objective by
    more than its solver tolerance triggers the same recovery. AugSynth
    reports an undefined scaled imbalance as NaN when uniform donors
    already balance the corresponding outcomes or covariates exactly. Its
    unregularized fit requires a full-row-rank centered design, and its
    residualization requires linearly independent centered covariates.
    """

    method: str = "synth"
    dataprep: Any = None
    options: dict = field(default_factory=dict)
    allow_post_predictors: bool = False

    def fit(self, panel: Panel) -> Estimate:
        """Fit the selected backend and preserve its historical result identifier."""
        return self._fit(
            panel,
            self.method,
            f"pysyncon.{self.method}",
            self.options,
            self.dataprep,
            self.allow_post_predictors,
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
        self.backend_ = None
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


@dataclass
class DifferencedSyntheticControl:
    """First-differenced synthetic control from Engelbrektson (2021, sections 4–5).

    This fixed-penalty specification selects predictor weights on all available
    pretreatment first differences. The inner simplex fit balances the mean
    pretreatment change and the selected fixed outcome-level lags. The penalty
    weights pairwise discrepancies in the original predictor profiles.
    The intercept is the mean pretreatment gap in levels. Zero penalty is
    the relaxed, unpenalized comparison in Table 2. Specify a fixed positive
    penalty to fit the penalized variant. The paper also studies split-sample
    penalty tuning; the constructor takes the penalty directly.
    """

    lambda_: float = 0.0
    _: KW_ONLY
    n_lags: int | None = None
    n_optim: int = 10
    random_state: int = 0
    max_iter: int = 10000

    def fit(self, panel: Panel) -> Estimate:
        """Fit the first-difference objective and restore the pretreatment mean level."""
        self.backend_ = None
        adapter = SCM(
            "diffsynth",
            pen=self.lambda_,
            n_lags=self.n_lags,
            n_optim=self.n_optim,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        result = adapter.fit(panel)
        result.method = "Differenced SCM"
        result.metadata.update(
            method_reference="https://arxiv.org/abs/2111.10784",
            backend_package="SyntheticControlMethods",
            backend_method="diffsynth",
            penalty=self.lambda_,
            penalty_selection="fixed",
            outer_fit_periods=panel.times[1 : panel.n_pre].tolist(),
        )
        self.backend_ = adapter.backend_
        return result
