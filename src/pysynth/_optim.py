"""Constrained least squares with analytic gradients and explicit failures."""

import numpy as np
from scipy.optimize import minimize


def _improving_transfer(a, penalty, weights, tolerance):
    """Find a feasible donor-to-donor transfer that materially improves the same loss."""
    residual = a @ weights
    # The initial SLSQP solve is scaled to the most extreme eligible donor.
    # Check the achieved fit on the scale of donors actually used, otherwise
    # one distant, zero-weight donor can make a bad active-face fit look exact.
    local_scale = max(np.max(np.abs(a) @ weights) ** 2, penalty)
    best_gain = tolerance * local_scale
    gradient = a.T @ residual / len(a) + penalty * weights  # half gradient
    best = None
    # For a transfer of size s from i to j the exact decrease is
    # 2 s (g_i - g_j) - s² (mean((a_i-a_j)²) + 2*penalty).
    # The linear part alone bounds every possible gain out of donor i.
    for i in np.flatnonzero(2 * weights * (gradient - gradient.min()) > best_gain):
        slope = gradient[i] - gradient
        targets = np.flatnonzero(2 * weights[i] * slope > best_gain)
        if not len(targets):
            continue
        delta = a[:, targets] - a[:, i, None]
        curvature = np.einsum("ij,ij->j", delta, delta) / len(a) + 2 * penalty
        step = np.full(len(targets), weights[i])
        positive = curvature > 0
        step[positive] = np.minimum(step[positive], slope[targets][positive] / curvature[positive])
        gains = step * (2 * slope[targets] - curvature * step)
        k = int(np.argmax(gains))
        if gains[k] > best_gain:
            best_gain = gains[k]
            best = i, targets[k], step[k]
    return best


def simplex_fit(x, y, *, ridge=0.0, max_iter=10000, tol=1e-12):
    """Minimise mean squared error + ridge * squared norm on the simplex."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.shape != (x.shape[0],) or not all(x.shape):
        raise ValueError("x must be a nonempty matrix and y a matching outcome vector")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must be finite")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    # On the simplex, (X-y1')w = Xw-y. Scaling these differences instead
    # of outcome levels preserves the fit after any common time-varying shift
    # and avoids false convergence when the original units are very small.
    differences = x - y[:, None]
    scale = max(float(np.max(np.abs(differences))), float(np.sqrt(ridge)))
    if scale == 0:
        return np.full(x.shape[1], 1 / x.shape[1]), {
            "iterations": 0,
            "donor_transfers": 0,
            "objective": 0.0,
            "optimality_gap": 0.0,
            "scaled_optimality_gap": 0.0,
        }
    a, penalty = differences / scale, (np.sqrt(ridge) / scale) ** 2
    n, p = a.shape

    def fun(w):
        e = a @ w
        return e @ e / n + penalty * (w @ w)

    def jac(w):
        return 2 * (a.T @ (a @ w) / n + penalty * w)

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
    if sol.x.min() < -1e-8 or abs(sol.x.sum() - 1) > 1e-8:
        raise RuntimeError("Synthetic control optimisation returned infeasible weights")
    w = np.maximum(sol.x, 0)
    w /= w.sum()
    transfers = 0
    while (transfer := _improving_transfer(a, penalty, w, tol)) is not None:
        if sol.nit + transfers >= max_iter:
            raise RuntimeError(
                "Synthetic control optimisation exhausted max_iter with an improving donor transfer"
            )
        donor, recipient, step = transfer
        w[donor] -= step
        w[recipient] += step
        transfers += 1
    residual = differences @ w
    # For a convex differentiable objective over the simplex, this
    # Frank--Wolfe gap bounds f(w)-min(f), independently of solver status.
    gradient = jac(w)
    scaled_gap = max(0.0, float(gradient @ w - gradient.min()))
    return w, {
        "iterations": sol.nit + transfers,
        "donor_transfers": transfers,
        "objective": float(residual @ residual / n + ridge * (w @ w)),
        "optimality_gap": scaled_gap * scale * scale,
        "scaled_optimality_gap": scaled_gap,
    }
