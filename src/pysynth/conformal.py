"""Time-permutation inference for outcome-matching synthetic control.

Chernozhukov, Wüthrich and Zhu (2021), JASA, Section 2 and Algorithm 1,
https://doi.org/10.1080/01621459.2021.1920957.
"""

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from ._optim import simplex_fit
from .panel import Panel


@dataclass(frozen=True)
class ConformalResult:
    """A sharp-path test, its null-imposed SCM fit, and all cyclic rotations.

    Row zero of `rotations` is the observed arrangement. Each other row
    shifts the first tested observation forward in panel order, wrapping at
    the end. `weights` and `residuals` belong to the fit under `null_effect`.
    """

    p_value: float
    statistic: float
    q: float
    null_effect: pd.Series
    weights: pd.Series
    residuals: pd.Series
    rotations: pd.DataFrame


def _real_values(values, name):
    raw = np.asanyarray(values)
    if raw.dtype.kind not in "fiu" or np.ma.is_masked(raw):
        raise ValueError(f"{name} must contain finite real numbers")
    array = np.asarray(raw, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array


def _interpolates(differences, weights, residuals):
    # Cancellation in the weighted dot product can leave a nonzero residual
    # for an exact mixture. Bound roundoff relative to each row's weighted
    # donor differences, so a common outcome shift or small measurement units
    # cannot on their own create a zero residual.
    eps = np.finfo(float).eps
    roundoff = 2 * (np.count_nonzero(weights) + 2) * eps * (np.abs(differences) @ weights)
    return np.all(np.abs(residuals) <= roundoff)


def _null_fit(donors, adjusted):
    # Time enters symmetrically in the fit, including every null-adjusted
    # post observation. A pre-period Estimate cannot supply these residuals.
    weights, _ = simplex_fit(donors, adjusted)
    differences = adjusted[:, None] - donors
    residuals = differences @ weights
    if _interpolates(differences, weights, residuals):
        return weights, np.zeros_like(residuals)
    # A least-squares stopping tolerance can also hide an exact convex fit.
    # When the solver is close, ask directly for feasible interpolating
    # weights. Verify every row at roundoff even if the LP reports success.
    used_scale = np.max(np.abs(differences) @ weights)
    if np.max(np.abs(residuals)) <= 1e-4 * used_scale:
        scale = np.max(np.abs(differences), axis=1, keepdims=True)
        scaled = differences / np.where(scale == 0, 1, scale)
        equality = np.vstack([np.ones(len(weights)), scaled])
        rhs = np.r_[1, np.zeros(len(adjusted))]
        feasible = linprog(np.zeros(len(weights)), A_eq=equality, b_eq=rhs, bounds=(0, None))
        if feasible.success:
            candidate = np.maximum(feasible.x, 0)
            candidate /= candidate.sum()
            if _interpolates(differences, candidate, differences @ candidate):
                return candidate, np.zeros_like(residuals)
    return weights, residuals


def _block_norm(block, q):
    # The replication code reports the unnormalised norm. A common positive
    # T_post factor in the paper's notation leaves every permutation rank intact.
    largest = float(block.max())
    if largest == 0 or q == np.inf:
        return largest
    if q == 1:
        # Scaling by a different maximum in each window can break genuine
        # ties, e.g. 0+4+11 versus 5+5+5, and thereby change the p-value.
        try:
            score = math.fsum(block)
        except OverflowError:
            score = math.inf
    elif q == 2:
        score = math.hypot(*block)
    else:
        scale_free_sum = math.fsum((block / largest) ** q)
        score = largest * scale_free_sum ** (1 / q)
    if not np.isfinite(score):
        raise ValueError("The block statistic exceeds float range; rescale the outcome and null")
    return score


def conformal_test(panel: Panel, null_effect=0.0, *, q: float = 1.0) -> ConformalResult:
    """Test a sharp effect path using every moving-block permutation.

    `null_effect` is a common effect at every post date, a vector in panel
    order, or a Series indexed by exactly the post-treatment dates. Subtract
    this path from the observed treated outcome, and fit nonnegative donor
    weights summing to one by least squares over all periods, without an
    intercept. The statistic is the L-q norm of the post-treatment residuals
    (q >= 1, including infinity). The p-value is the fraction of the T cyclic
    windows of this length with a statistic at least as large, including the
    observed window and all ties. The paper's application uses q=1.

    The sharp null specifies an effect at each post date. Finite-sample size
    control requires invariance under the selected group, as with exchangeable
    null residuals. Under serial dependence the paper gives approximate validity
    under additional conditions on the errors and counterfactual fit.
    """
    if panel.n_treated != 1:
        raise ValueError("Conformal inference requires exactly one treated unit")
    if isinstance(q, (bool, np.bool_)) or not isinstance(q, Real) or not q >= 1:
        raise ValueError("q must be a real number >= 1 or positive infinity")
    q = float(q)
    post_dates = panel.times[panel.n_pre :]
    if isinstance(null_effect, pd.Series) and not null_effect.index.equals(post_dates):
        raise ValueError("null_effect index must match post-treatment dates in order")
    path = _real_values(null_effect, "null_effect")
    if path.ndim == 0:
        path = np.full(panel.n_post, path.item())
    if path.shape != (panel.n_post,):
        raise ValueError("null_effect must specify one value per post-treatment date")
    adjusted = panel.treated.copy()
    adjusted[panel.n_pre :] -= path
    weights, residuals = _null_fit(panel.control, adjusted)
    n = len(adjusted)
    starts = (panel.n_pre + np.arange(n)) % n
    stops = starts + panel.n_post - 1
    magnitudes = np.tile(np.abs(residuals), 2)
    scores = np.array([_block_norm(magnitudes[i : i + panel.n_post], q) for i in starts])
    tail = scores >= scores[0]
    rotations = pd.DataFrame(
        {
            "window_start": panel.times.take(starts),
            "window_end": panel.times.take(stops % n),
            "wraps": stops >= n,
            "statistic": scores,
            "at_least_observed": tail,
        },
        index=pd.RangeIndex(n, name="shift"),
    )
    donors = pd.Index(panel.units[: panel.n_control], tupleize_cols=False, name="donor")
    return ConformalResult(
        p_value=float(tail.mean()),
        statistic=float(scores[0]),
        q=float(q),
        null_effect=pd.Series(path.copy(), index=post_dates, name="null_effect"),
        weights=pd.Series(weights, index=donors, name="null_weight"),
        residuals=pd.Series(residuals, index=panel.times, name="null_residual"),
        rotations=rotations,
    )


def conformal_confidence_set(panel: Panel, candidates, *, alpha: float = 0.1) -> pd.DataFrame:
    """Invert the single-date test over an explicit increasing grid of effects.

    For each post date, fit outcome SCM using the pre-period observations and
    that date alone, separately for each candidate. Return a long table with
    `time`, `candidate`, `p_value` and `accepted` (`p_value > alpha`). Selecting
    the accepted rows gives the finite-grid pointwise confidence set. The grid
    can be uneven, empty of accepted values or disconnected after inversion.
    Each date has T_pre + 1 possible cyclic positions. Joint coverage across
    dates requires a different construction. The returned table records the
    test level in `attrs["alpha"]`.
    """
    if panel.n_treated != 1:
        raise ValueError("Conformal inference requires exactly one treated unit")
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    grid = _real_values(candidates, "candidates")
    if grid.ndim != 1 or not len(grid) or not np.all(grid[1:] > grid[:-1]):
        raise ValueError("candidates must be a nonempty strictly increasing one-dimensional grid")
    p_values = []
    treated = panel.treated
    for t in range(panel.n_pre, len(panel.times)):
        rows = np.r_[0 : panel.n_pre, t]
        donors = panel.control[rows]
        adjusted = treated[rows].copy()
        for effect in grid:
            adjusted[-1] = treated[t] - effect
            _, residuals = _null_fit(donors, adjusted)
            # With one tested date each absolute residual is a rotation score.
            p_values.append(np.mean(np.abs(residuals) >= abs(residuals[-1])))
    p_values = np.asarray(p_values)
    result = pd.DataFrame(
        {
            "time": panel.times[panel.n_pre :].repeat(len(grid)),
            "candidate": np.tile(grid, panel.n_post),
            "p_value": p_values,
            "accepted": p_values > alpha,
        }
    )
    result.attrs["alpha"] = float(alpha)
    return result
