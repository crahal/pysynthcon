"""Synthetic Control Using Lasso with rolling-origin validation."""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import lasso_path

from .panel import Panel
from .results import Estimate, make_estimate


def _standardize(x, y):
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale[scale < np.finfo(float).eps] = 1.0
    return (x - mean) / scale, y - y.mean(), mean, scale


def _path(x, y, alphas, max_iter, tol):
    z, target, mean, scale = _standardize(x, y)
    _, coef, gaps = lasso_path(z, target, alphas=alphas, max_iter=max_iter, tol=tol)
    if np.any(gaps > tol * max(float(target @ target / len(y)), 1.0) * 10):
        raise RuntimeError("SCUL lasso path did not converge; increase max_iter or tolerance")
    weights = coef / scale[:, None]
    intercepts = y.mean() - mean @ weights
    return weights, intercepts


@dataclass
class SCUL:
    """Lasso synthetic control using only pre-treatment rolling validation.

    The lasso objective is ``||y-a-Xb||²/(2*n) + alpha*||b||₁`` with donor
    columns standardized on each training window. Weights may be negative
    and need not sum to one. ``selection='min'`` follows SCUL's common-grid
    lambda.min procedure. ``'1se'`` uses the upstream fold-SD rule (despite
    its name); ``'median'`` takes the median of fold-specific best penalties
    on the explicit grid. This last option differs from glmnet's automatically
    truncated, fold-specific default paths. An explicit ``alpha`` bypasses CV.

    ``initial_window`` defaults to half the pre-periods (at least three).
    ``horizon`` sets the contiguous validation window; origins advance by one.
    At least two validation origins are required for automatic selection.
    """

    alpha: float | None = None
    initial_window: int | None = None
    horizon: int = 1
    selection: str = "min"
    alphas: tuple | None = None
    n_alphas: int = 100
    max_iter: int = 100000
    tol: float = 1e-7

    def fit(self, panel: Panel) -> Estimate:
        """Select the penalty, refit all pre-periods, and predict all periods."""
        if panel.n_treated != 1:
            raise ValueError("SCUL requires exactly one treated series")
        if self.selection not in {"min", "1se", "median"}:
            raise ValueError("selection must be min, 1se, or median")
        if self.max_iter < 1 or not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("max_iter and tol must be positive")
        x, y = panel.control[: panel.n_pre], panel.treated[: panel.n_pre]
        metadata = {"selection": self.selection}
        if self.alpha is not None:
            if not np.isfinite(self.alpha) or self.alpha <= 0:
                raise ValueError("alpha must be finite and strictly positive")
            alpha = self.alpha
        else:
            initial = (
                max(3, panel.n_pre // 2) if self.initial_window is None else self.initial_window
            )
            if not isinstance(initial, (int, np.integer)) or initial < 2:
                raise ValueError("initial_window must be an integer >= 2")
            if not isinstance(self.horizon, (int, np.integer)) or self.horizon < 1:
                raise ValueError("horizon must be a positive integer")
            origins = list(range(initial, panel.n_pre - self.horizon + 1))
            if len(origins) < 2:
                raise ValueError("SCUL needs at least two rolling validation origins")
            if self.alphas is None:
                if not isinstance(self.n_alphas, (int, np.integer)) or self.n_alphas < 2:
                    raise ValueError("n_alphas must be an integer >= 2")
                maxima = []
                for end in origins:
                    z, target, _, _ = _standardize(x[:end], y[:end])
                    maxima.append(max(float(np.max(np.abs(z.T @ target)) / end), 1e-12))
                grid = np.geomspace(max(maxima), min(maxima) * 0.01, self.n_alphas)
            else:
                grid = np.asarray(self.alphas, dtype=float)
                if (
                    grid.ndim != 1
                    or len(grid) < 2
                    or not np.isfinite(grid).all()
                    or (grid <= 0).any()
                ):
                    raise ValueError("alphas must contain at least two finite positive values")
                grid = np.sort(np.unique(grid))[::-1]
            errors = []
            for end in origins:
                weights, offsets = _path(x[:end], y[:end], grid, self.max_iter, self.tol)
                prediction = x[end : end + self.horizon] @ weights + offsets
                errors.append(
                    np.mean((prediction - y[end : end + self.horizon, None]) ** 2, axis=0)
                )
            errors = np.asarray(errors)
            means = errors.mean(axis=0)
            best = int(np.argmin(means))
            if self.selection == "min":
                alpha = grid[best]
            elif self.selection == "1se":
                alpha = grid[np.flatnonzero(means <= means[best] + errors[:, best].std(ddof=1))[0]]
            else:
                alpha = np.median(grid[np.argmin(errors, axis=1)])
            metadata.update(
                {
                    "origins": origins,
                    "horizon": self.horizon,
                    "alphas": grid.tolist(),
                    "cv_mse": means.tolist(),
                    "fold_mse": errors.tolist(),
                }
            )
        weights, offsets = _path(x, y, np.array([alpha]), self.max_iter, self.tol)
        metadata["alpha"] = float(alpha)
        return make_estimate(panel, "SCUL", weights[:, 0], offsets[0], metadata=metadata)
