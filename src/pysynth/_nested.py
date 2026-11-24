"""Numerically checked nested optimization for the SCMethods adapter.

The statistical objective, bounds, and uniform/Dirichlet starts are those of
SyntheticControlMethods. Solver precision and failure recovery are explicit.
"""

import numpy as np
from scipy.optimize import OptimizeResult, minimize


class _DerivativeUnavailable(RuntimeError):
    """The inner optimum does not admit a numerically reliable derivative."""


def _polish_weights(a, pairwise, v, pen, weights):
    """Certify an active-set QP solution; return its KKT system and dual slack.

    An interior-point solver can leave tiny positive weights on inactive
    donors. Starting from its support, remove negative active weights and add
    donors with negative reduced gradients until the QP's KKT conditions hold.
    This solves the same constrained quadratic objective; it adds no penalty.
    """
    support = set(np.flatnonzero(weights > 1e-7).tolist()) or {int(np.argmax(weights))}
    visited = set()
    for _ in range(3 * len(weights)):
        key = tuple(sorted(support))
        if key in visited:
            raise _DerivativeUnavailable("active-set polishing cycled")
        visited.add(key)
        active = np.array(key, dtype=int)
        design = a[:, active]
        hessian = 2 * (design.T * v) @ design
        linear = pen * (pairwise[:, active].T @ v)
        kkt = np.block(
            [[hessian, np.ones((len(active), 1))], [np.ones((1, len(active))), np.zeros((1, 1))]]
        )
        if np.linalg.cond(kkt) > 1e12:
            raise _DerivativeUnavailable("singular or ill-conditioned active-set KKT system")
        try:
            solution = np.linalg.solve(kkt, np.r_[-linear, 1.0])
        except np.linalg.LinAlgError as exc:
            raise _DerivativeUnavailable("singular active-set KKT system") from exc
        if solution[:-1].min() < 1e-12:
            support.remove(int(active[np.argmin(solution[:-1])]))
            continue
        polished = np.zeros(len(weights))
        polished[active] = solution[:-1]
        dual = 2 * a.T @ (v * (a @ polished)) + pen * (pairwise.T @ v) + solution[-1]
        inactive = np.ones(len(weights), dtype=bool)
        inactive[active] = False
        if inactive.any() and dual[inactive].min() < -1e-9:
            support.add(int(np.flatnonzero(inactive)[np.argmin(dual[inactive])]))
            continue
        if (
            not np.isfinite(polished).all()
            or abs(polished.sum() - 1) > 1e-8
            or np.max(np.abs(dual[active])) > 1e-8
        ):
            raise _DerivativeUnavailable("active-set KKT residual exceeds tolerance")
        return polished, active, kkt, dual, inactive
    raise _DerivativeUnavailable("active-set polishing reached its iteration limit")


def _outer_derivative(a, pairwise, pen, z, residual, weights, active, kkt, dual, inactive):
    """Differentiate outer SSE through a regular inner QP using an adjoint solve."""
    if weights[active].min() < 1e-12 or (inactive.any() and dual[inactive].min() < 1e-12):
        raise _DerivativeUnavailable("inner QP lies on a nondifferentiable active-set boundary")
    try:
        adjoint = np.linalg.solve(kkt, np.r_[-2 * z[:, active].T @ residual, 0.0])[:-1]
    except np.linalg.LinAlgError as exc:
        raise _DerivativeUnavailable("singular adjoint KKT system") from exc
    derivative = -(
        2 * (a[:, active] @ adjoint) * (a @ weights) + pen * (pairwise[:, active] @ adjoint)
    )
    if not np.isfinite(derivative).all():
        raise _DerivativeUnavailable("nonfinite implicit derivative")
    return derivative


class StableNestedOptimizer:
    """Local mixin; never modifies the installed upstream package."""

    outer_max_iter = 10000

    def optimize(
        self,
        treated_outcome,
        treated_covariates,
        control_outcome,
        control_covariates,
        pairwise_difference,
        data,
        placebo,
        pen,
        steps=10,
        verbose=False,
    ):
        import cvxpy as cp

        if placebo:
            raise NotImplementedError("Use pysynth.placebo_test to refit the checked adapter")
        target = np.asarray(treated_covariates).reshape(-1)
        x = np.asarray(control_covariates)
        z = np.asarray(control_outcome)
        y = np.asarray(treated_outcome).reshape(-1)
        n_predictors, n_donors = x.shape
        v = cp.Parameter(n_predictors, nonneg=True)
        w = cp.Variable(n_donors)
        # On sum(w)=1 this is algebraically identical to target-Xw,
        # but it avoids cancellation between large outcome levels in the QP.
        a = x - target[:, None]
        discrepancy = cp.square(a @ w)
        pairwise = np.asarray(pairwise_difference) ** 2
        problem = cp.Problem(
            cp.Minimize(v @ discrepancy + pen * (v @ pairwise @ w)),
            [w >= 0, cp.sum(w) == 1],
        )
        # A scalar rescaling preserves the outer minimizer while avoiding
        # outcome-unit-dependent finite difference and stopping tolerances.
        loss_scale = max(float(np.var(y)), 1e-8) * len(y)
        best = {"loss": np.inf}
        cache = {}
        evaluations = 0

        def objective(candidate):
            nonlocal evaluations
            candidate = np.clip(np.asarray(candidate), 0, 1)
            if "candidate" in cache and np.array_equal(candidate, cache["candidate"]):
                return cache["loss"] / loss_scale
            evaluations += 1
            if candidate.sum() <= np.finfo(float).eps:
                cache.clear()
                cache.update(
                    candidate=candidate.copy(),
                    loss=1e30 * loss_scale,
                    unavailable="all predictor weights are zero",
                )
                return 1e30
            v.value = candidate
            problem.solve(
                solver="CLARABEL",
                warm_start=False,
                tol_gap_abs=1e-10,
                tol_gap_rel=1e-10,
                tol_feas=1e-10,
                max_iter=500,
            )
            if problem.status != cp.OPTIMAL or w.value is None:
                raise RuntimeError(f"SCM inner quadratic program failed: {problem.status}")
            weights = np.asarray(w.value).reshape(-1)
            if weights.min() < -1e-8 or abs(weights.sum() - 1) > 1e-8:
                raise RuntimeError("SCM inner quadratic program returned infeasible weights")
            weights = np.maximum(weights, 0)
            weights /= weights.sum()
            cache.clear()
            cache["candidate"] = candidate.copy()
            try:
                weights, active, kkt, dual, inactive = _polish_weights(
                    a, pairwise, candidate, pen, weights
                )
                cache.update(active=active, kkt=kkt, dual=dual, inactive=inactive)
            except _DerivativeUnavailable as exc:
                # The original convex QP remains solved and feasible. It can
                # still be evaluated by a derivative-free outer optimizer.
                cache["unavailable"] = str(exc)
            residual = y - z @ weights
            loss = float(residual @ residual)
            cache.update(loss=loss, weights=weights.copy(), residual=residual)
            if loss < best["loss"]:
                best.update(loss=loss, weights=weights.copy(), v=candidate.copy())
            return loss / loss_scale

        def gradient(candidate):
            objective(candidate)
            if "unavailable" in cache:
                raise _DerivativeUnavailable(cache["unavailable"])
            # Exact outer fit is its global lower bound, including a redundant
            # active constraint where strict complementarity need not hold.
            if cache["loss"] <= np.finfo(float).eps * loss_scale:
                return np.zeros(n_predictors)
            return (
                _outer_derivative(
                    a,
                    pairwise,
                    pen,
                    z,
                    cache["residual"],
                    cache["weights"],
                    cache["active"],
                    cache["kkt"],
                    cache["dual"],
                    cache["inactive"],
                )
                / loss_scale
            )

        diagnostics = []
        for start in range(steps):
            initial = (
                np.full(n_predictors, 1 / n_predictors)
                if start == 0
                else self.original_data.rng.dirichlet(np.ones(n_predictors)).reshape(-1)
            )
            bounds = [(0, 1)] * n_predictors
            initial_loss = objective(initial)
            before = evaluations
            try:
                solved = minimize(
                    objective,
                    initial,
                    jac=gradient,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={
                        "maxiter": self.outer_max_iter,
                        "maxls": 50,
                        "ftol": 1e-10,
                        "gtol": 1e-7,
                    },
                )
            except _DerivativeUnavailable as exc:
                solved = OptimizeResult(
                    x=cache["candidate"].copy(),
                    fun=cache["loss"] / loss_scale,
                    success=False,
                    nfev=evaluations - before,
                    message=str(exc),
                )
            attempts = [
                {
                    "solver": "L-BFGS-B",
                    "derivative": "active-set KKT adjoint",
                    "success": bool(solved.success),
                    "message": str(solved.message),
                    "evaluations": int(solved.nfev),
                }
            ]
            if not solved.success:
                # At a changing or singular active set an implicit derivative
                # need not exist. Powell solves the same full objective without
                # accepting an unreliable gradient or discarding this start.
                solved = minimize(
                    objective,
                    solved.x,
                    method="Powell",
                    bounds=bounds,
                    options={
                        "maxiter": self.outer_max_iter,
                        "maxfev": 1000000,
                        "xtol": 1e-6,
                        "ftol": 1e-10,
                    },
                )
                attempts.append(
                    {
                        "solver": "Powell",
                        "success": bool(solved.success),
                        "message": str(solved.message),
                        "evaluations": int(solved.nfev),
                    }
                )
            if not solved.success or not np.isfinite(solved.fun):
                raise RuntimeError(f"SCM optimization start {start + 1} failed: {solved.message}")
            diagnostics.append(
                {
                    "start": start + 1,
                    "initial_loss": initial_loss * loss_scale,
                    "final_loss": float(solved.fun * loss_scale),
                    "attempts": attempts,
                }
            )
        self.optimization_diagnostics_ = diagnostics
        data.min_loss = best["loss"]
        data.w = best["weights"][:, None]
        data.v = best["v"] / best["v"].sum()
        data.pen = pen
        data.synth_outcome = data.w.T @ data.control_outcome_all.T
        data.synth_covariates = data.control_covariates @ data.w
        if self.method == "DSC":
            self._update_original_data(placebo)
