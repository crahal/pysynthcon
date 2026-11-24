"""Transparent in-space placebo diagnostics."""

import copy
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np
import pandas as pd

from .panel import Panel
from .results import Estimate, Estimator, _root_mean_square, _validate_estimate


@dataclass
class PlaceboResult:
    """Placebo fits, RMSPE ratios and a finite-sample rank p-value.

    A randomization interpretation requires exchangeable assignment and a
    donor pool unaffected by treatment; the diagnostic itself establishes
    neither assumption. The actual fit uses one more donor than the placebo
    fits, so this rank alone is not an exact randomization p-value. This is not
    synthdid's placebo variance estimator.
    """

    treated: Estimate
    placebos: dict
    table: pd.DataFrame
    p_value: float


def placebo_test(
    estimator: Estimator, panel: Panel, *, max_pre_rmse_ratio: float | None = None
) -> PlaceboResult:
    """Refit every donor as pseudo-treated, excluding the actually treated unit.

    Rank the treated post/pre RMSPE ratio among the eligible placebo ratios.
    ``p=(1 + # placebo ratios >= treated ratio)/(1 + # eligible placebos)``.
    An optional pre-fit cutoff is relative to the treated pre-RMSE; the treated
    fit is always included. No model failures are suppressed. Constant-zero
    pre and post gaps have ratio zero; zero pre with nonzero post has infinity.
    Only exact zero triggers these conventions: no outcome-unit-dependent
    cutoff is applied. RMSPEs are computed with scaling to avoid squaring
    overflow or underflow. The displayed ratio is a float; the
    ``at_least_treated_ratio`` column records the comparison of exact
    quotients of the computed RMSPEs, including ratios outside float range.
    The unequal actual/placebo donor-pool sizes make
    this a descriptive rank rather than an exact randomization test.
    """
    if panel.n_treated != 1 or panel.n_control < 2:
        raise ValueError("Placebo tests require one treated unit and at least two donors")
    if max_pre_rmse_ratio is not None and (
        not np.isfinite(max_pre_rmse_ratio) or max_pre_rmse_ratio <= 0
    ):
        raise ValueError("max_pre_rmse_ratio must be finite and positive")

    def fit_one(sample):
        fitted = copy.deepcopy(estimator).fit(sample)
        _validate_estimate(sample, fitted, allow_subset_weights=True)
        return copy.deepcopy(replace(fitted, metadata=dict(fitted.metadata)))

    treated = fit_one(panel)
    fits = {}
    for i, unit in enumerate(panel.units[: panel.n_control]):
        donors = [j for j in range(panel.n_control) if j != i]
        order = donors + [i]
        pseudo = Panel(
            panel.control[:, order],
            len(donors),
            panel.n_pre,
            panel.times,
            tuple(panel.units[j] for j in order),
        )
        fits[unit] = fit_one(pseudo)

    def row(fit, is_treated=False):
        pre = fit.pre_rmse
        post = _root_mean_square(fit.effects.iloc[fit.n_pre :])
        ratio = post / pre if pre > 0 else (np.inf if post > 0 else 0.0)
        eligible = (
            is_treated or max_pre_rmse_ratio is None or pre <= max_pre_rmse_ratio * treated.pre_rmse
        )
        return {
            "att": fit.att,
            "pre_rmse": pre,
            "post_rmse": post,
            "rmspe_ratio": ratio,
            "eligible": eligible,
            "treated": is_treated,
        }

    table = pd.DataFrame.from_dict(
        {panel.units[-1]: row(treated, True), **{key: row(fit) for key, fit in fits.items()}},
        orient="index",
    )

    # A finite post/pre quotient can exceed float64 or round to zero. Rank
    # exact quotients of the computed RMSPEs, preserving mathematical ties.
    def rank_ratio(post, pre):
        if pre == 0:
            return np.inf if post > 0 else 0
        return Fraction(post) / Fraction(pre)

    treated_ratio = rank_ratio(table.iloc[0].post_rmse, treated.pre_rmse)
    table["at_least_treated_ratio"] = [
        rank_ratio(post, pre) >= treated_ratio for post, pre in zip(table.post_rmse, table.pre_rmse)
    ]
    eligible = table.loc[(~table.treated) & table.eligible]
    rank = 1 + eligible.at_least_treated_ratio.sum()
    return PlaceboResult(treated, fits, table, float(rank / (1 + len(eligible))))
