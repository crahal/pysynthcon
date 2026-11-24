"""Checked numerical repair for a failed original MSCMT inner quadratic.

This process worker is invoked only after WNNLS reports failure. Its input
is exactly sqrt(V) times MSCMT's donor-minus-treated design, after MSCMT's
own scaling. Neither predictor weights nor the statistical objective change.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def solve(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not all(x.shape) or not np.isfinite(x).all():
        raise ValueError("MSCMT quadratic design must be a nonempty finite matrix")
    scale = float(np.max(np.abs(x)))
    p = x.shape[1]
    if scale == 0:
        return np.full(p, 1 / p), {"dual_gap": 0.0, "iterations": 0, "objective": 0.0}
    a = x / scale
    gram = a.T @ a / len(a)

    def fun(w):
        return float(w @ gram @ w)

    def jac(w):
        return 2 * gram @ w

    result = minimize(
        fun,
        np.full(p, 1 / p),
        jac=jac,
        method="SLSQP",
        bounds=[(0, 1)] * p,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(p)},
        options={"maxiter": 100000, "ftol": 1e-14},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"MSCMT inner quadratic repair failed: {result.message}")
    if result.x.min() < -1e-8 or abs(result.x.sum() - 1) > 1e-8:
        raise RuntimeError("MSCMT inner quadratic repair returned infeasible weights")
    w = np.maximum(result.x, 0)
    w /= w.sum()
    # On a simplex, this Frank-Wolfe gap bounds suboptimality of the convex
    # quadratic. Check it independently of the optimizer's stopping flag.
    gradient = jac(w)
    gap = max(0.0, float(w @ gradient - gradient.min()))
    if gap > 1e-8:
        raise RuntimeError(f"MSCMT inner quadratic repair has excessive KKT gap {gap:g}")
    # The Gram-matrix expression can be negative through cancellation at an
    # exact fit. Measure the residual directly in the original objective.
    residual = (a @ w) * scale
    rms = np.linalg.norm(residual / np.sqrt(len(residual)))
    objective = float(rms * rms)
    if not np.isfinite(objective):
        raise RuntimeError("MSCMT weighted residual objective exceeds floating-point range")
    return w, {"dual_gap": gap, "iterations": int(result.nit), "objective": objective}


if __name__ == "__main__":
    matrix = np.loadtxt(sys.argv[1], delimiter=",", ndmin=2)
    weights, diagnostics = solve(matrix)
    Path(sys.argv[2]).write_text(json.dumps({"weights": weights.tolist(), **diagnostics}))
