"""Synthetic Control Using Lasso with rolling-origin validation."""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import lasso_path

from ._r import run_reference
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
    """Original SCUL procedure using pre-treatment rolling validation.

    The default ``backend='r'`` runs the pinned original SCUL functions with
    glmnet 4.0-2. Its default median rule uses each fold's own automatic glmnet
    path. No approximate solver or shortened penalty path is substituted.
    ``initial_window`` and ``horizon`` default to the post-treatment length,
    matching the original setup function; specify them for short panels.

    ``backend='python'`` explicitly selects the earlier sklearn adaptation.
    It is useful for dependency-free experiments but is not numerically
    identical to the original procedure, particularly for median selection.

    The lasso objective is ``||y-a-Xb||²/(2*n) + alpha*||b||₁`` with donor
    columns standardized on each training window. Weights may be negative
    and need not sum to one. ``selection='min'`` follows SCUL's common-grid
    lambda.min procedure. ``'1se'`` uses the upstream fold-SD rule (despite
    its name). An explicit ``alpha`` bypasses CV.

    The Python adaptation uses a common grid for all three selection rules,
    defaults to half the pre-periods (at least three) for ``initial_window``,
    and defaults to a one-period ``horizon``. At least two validation origins
    are required for automatic selection with either backend.
    """

    alpha: float | None = None
    initial_window: int | None = None
    horizon: int | None = None
    selection: str = "median"
    alphas: tuple | None = None
    n_alphas: int = 100
    max_iter: int = 100000
    tol: float = 1e-7
    backend: str = "r"
    r_library: str | None = None

    def fit(self, panel: Panel) -> Estimate:
        """Select the penalty, refit all pre-periods, and predict all periods."""
        if panel.n_treated != 1:
            raise ValueError("SCUL requires exactly one treated series")
        if self.selection not in {"min", "1se", "median"}:
            raise ValueError("selection must be min, 1se, or median")
        if self.backend not in {"r", "python"}:
            raise ValueError("backend must be 'r' (original) or 'python' (adaptation)")
        if self.max_iter < 1 or not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("max_iter and tol must be positive")
        if self.backend == "r":
            return self._fit_reference(panel)
        x, y = panel.control[: panel.n_pre], panel.treated[: panel.n_pre]
        metadata = {"selection": self.selection, "backend": "python_adaptation"}
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
            horizon = 1 if self.horizon is None else self.horizon
            if not isinstance(horizon, (int, np.integer)) or horizon < 1:
                raise ValueError("horizon must be a positive integer")
            origins = list(range(initial, panel.n_pre - horizon + 1))
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
                prediction = x[end : end + horizon] @ weights + offsets
                errors.append(np.mean((prediction - y[end : end + horizon, None]) ** 2, axis=0))
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
                    "horizon": horizon,
                    "alphas": grid.tolist(),
                    "cv_mse": means.tolist(),
                    "fold_mse": errors.tolist(),
                }
            )
        weights, offsets = _path(x, y, np.array([alpha]), self.max_iter, self.tol)
        metadata["alpha"] = float(alpha)
        return make_estimate(panel, "SCUL", weights[:, 0], offsets[0], metadata=metadata)

    def _fit_reference(self, panel):
        if self.alphas is not None or self.n_alphas != 100:
            raise ValueError("The original SCUL procedure defines its own 100-lambda grid")
        if self.max_iter != 100000 or self.tol != 1e-7:
            raise ValueError("The original SCUL backend retains glmnet's default solver settings")
        if self.alpha is not None and (not np.isfinite(self.alpha) or self.alpha <= 0):
            raise ValueError("alpha must be finite and strictly positive")
        n_post = len(panel.y) - panel.n_pre
        initial = n_post if self.initial_window is None else self.initial_window
        horizon = n_post if self.horizon is None else self.horizon
        if not isinstance(initial, (int, np.integer)) or initial < 2:
            raise ValueError("initial_window must be an integer >= 2")
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        origins = list(range(initial, panel.n_pre - horizon + 1))
        if self.alpha is None and len(origins) < 2:
            raise ValueError(
                "SCUL needs at least two rolling validation origins. The original defaults "
                "use the post-period length for initial_window and horizon; specify an "
                "appropriate shorter design explicitly when the pretreatment period is too short."
            )
        # The original preprocessing removes donors constant in any training
        # window. Expanding windows make checking the first sufficient.
        train = panel.control[: initial if self.alpha is None else panel.n_pre]
        keep = np.std(train, axis=0) > 0
        if keep.sum() < 2:
            raise ValueError("Original glmnet SCUL needs at least two nonconstant donor series")
        if np.std(panel.treated[: initial if self.alpha is None else panel.n_pre]) == 0:
            raise ValueError("Original glmnet SCUL requires a nonconstant training outcome")
        result = run_reference(
            "scul",
            {"outcomes": np.column_stack([panel.control[:, keep], panel.treated])},
            {
                "n_pre": panel.n_pre,
                "initial_window": int(initial),
                "horizon": int(horizon),
                "selection": f"lambda.{self.selection}",
                "alpha": self.alpha,
            },
            r_library=self.r_library,
        )
        coef = np.asarray(result["coefficients"], dtype=float)
        weights = np.zeros(panel.n_control)
        weights[keep] = coef[1:]
        return make_estimate(
            panel,
            "SCUL",
            weights,
            coef[0],
            metadata={
                "backend": "original_r",
                "reference_commit": "121b588cf8e4eebcea68518b57a7c754040153bb",
                "glmnet_version": result["glmnet_version"],
                "r_version": result["r_version"],
                "selection": self.selection,
                "alpha": result["alpha"],
                "initial_window": initial,
                "horizon": horizon,
                "origins": origins if self.alpha is None else [],
                "removed_constant_donors": [panel.units[i] for i in np.flatnonzero(~keep)],
                "lambda_options": result.get("lambda"),
            },
        )
