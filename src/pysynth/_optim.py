"""Constrained least squares with analytic gradients and explicit failures."""

import numpy as np
from scipy.optimize import minimize


def simplex_fit(x, y, *, ridge=0.0, max_iter=10000, tol=1e-12):
    """Minimise mean squared error + ridge * squared norm on the simplex."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    # Global scaling improves SLSQP conditioning without changing the minimizer.
    scale = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), 1.0)
    a, b, penalty = x / scale, y / scale, ridge / scale**2
    n, p = a.shape

    def fun(w):
        e = a @ w - b
        return e @ e / n + penalty * (w @ w)

    def jac(w):
        return 2 * (a.T @ (a @ w - b) / n + penalty * w)

    sol = minimize(
        fun,
        np.full(p, 1 / p),
        jac=jac,
        method="SLSQP",
        bounds=[(0, 1)] * p,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(p)},
        options={"maxiter": max_iter, "ftol": tol},
    )
    if not sol.success or not np.isfinite(sol.x).all():
        raise RuntimeError(f"Synthetic control optimisation failed: {sol.message}")
    w = np.maximum(sol.x, 0)
    return w / w.sum(), {"iterations": sol.nit, "objective": fun(w) * scale**2}
