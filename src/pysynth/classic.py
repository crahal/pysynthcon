"""Outcome-matching convex synthetic control."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ._optim import simplex_fit
from .panel import Panel
from .results import Estimate, make_estimate


@dataclass
class SyntheticControl:
    """Convex SCM matching every pre-treatment outcome with equal importance.

    This is the outcome-only SCM specification. Use ``PredictorSyntheticControl``
    for the nested predictor-weight optimisation in classic
    published applications. ``ridge`` is an optional squared-weight penalty.
    """

    ridge: float = 0.0
    max_iter: int = 10000
    tol: float = 1e-12

    def fit(self, panel: Panel) -> Estimate:
        """Fit donor weights using only pre-treatment observations."""
        if not np.isfinite(self.ridge) or self.ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        if (
            not isinstance(self.max_iter, Integral)
            or isinstance(self.max_iter, bool)
            or self.max_iter < 1
            or not np.isfinite(self.tol)
            or self.tol <= 0
        ):
            raise ValueError("max_iter and tol must be positive")
        w, meta = simplex_fit(
            panel.control[: panel.n_pre],
            panel.treated[: panel.n_pre],
            ridge=self.ridge,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        return make_estimate(panel, "SyntheticControl", w, metadata=meta)
