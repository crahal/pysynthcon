"""Transparent in-space placebo diagnostics."""

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .panel import Panel
from .results import Estimate, Estimator, _root_mean_square


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
    overflow or underflow. The unequal actual/placebo donor-pool sizes make
    this a descriptive rank rather than an exact randomization test.
    """
    if panel.n_treated != 1 or panel.n_control < 2:
        raise ValueError("Placebo tests require one treated unit and at least two donors")
    if max_pre_rmse_ratio is not None and (
        not np.isfinite(max_pre_rmse_ratio) or max_pre_rmse_ratio <= 0
    ):
        raise ValueError("max_pre_rmse_ratio must be finite and positive")
    treated = copy.deepcopy(estimator).fit(panel)
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
        fits[unit] = copy.deepcopy(estimator).fit(pseudo)

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
    eligible = table.loc[(~table.treated) & table.eligible]
    rank = 1 + (eligible.rmspe_ratio >= table.iloc[0].rmspe_ratio).sum()
    return PlaceboResult(treated, fits, table, float(rank / (1 + len(eligible))))
