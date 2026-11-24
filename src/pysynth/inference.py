"""In-space synthetic control placebo paths, pre-fit screens and rank diagnostics."""

import warnings
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction
from numbers import Integral, Real
from typing import Literal

import numpy as np
import pandas as pd

from .panel import Panel
from .results import Estimate, Estimator, _root_mean_square, _validate_estimate


def _check_cutoff(value, name):
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")


def _exact_ratio(post, pre):
    # Compare the actual floating-point RMSPEs as exact quotients. Their
    # ratio (or square) can exceed float range or round distinct fits to ties.
    if pre == 0:
        return np.inf if post > 0 else 0
    return Fraction(post) / Fraction(pre)


@dataclass
class PlaceboResult:
    """All assignment fits, period-by-period gaps, and post/pre error ranks.

    `table` starts with the treated unit. `eligible` marks the paths retained
    by the selected pre-fit screen. `p_value` is the fraction of eligible
    assignments at least as extreme as the treated assignment, including ties.
    Interpreting an unfiltered reassignment rank as a randomization p-value
    requires the corresponding assignment mechanism under a sharp null.
    The default untreated-only design has unequal donor pools.
    """

    treated: Estimate
    placebos: dict
    table: pd.DataFrame
    p_value: float
    donor_pool: Literal["untreated", "reassign"] = "untreated"
    max_pre_mspe_ratio: float | None = None
    max_pre_rmse_ratio: float | None = None

    @property
    def gaps(self) -> pd.DataFrame:
        """All observed-minus-counterfactual paths by time and assigned unit."""
        gaps = pd.concat(
            [self.treated.effects, *(fit.effects for fit in self.placebos.values())], axis=1
        )
        actual = self.table.index[self.table.treated][0]
        gaps.columns = pd.Index([actual, *self.placebos], name="unit", tupleize_cols=False)
        return gaps

    @property
    def rank(self) -> int:
        """Upper-tail rank among eligible assignments, counting ties conservatively."""
        return int((self.table.eligible & self.table.at_least_treated_ratio).sum())

    def with_pre_mspe_cutoff(self, multiplier: float | None) -> "PlaceboResult":
        """Screen existing fits by pre-MSPE relative to treated pre-MSPE.

        The screen includes equality. Repeated calls start with every stored
        fit. Pass `None` to recover the unscreened paths and rank.
        """
        _check_cutoff(multiplier, "multiplier")
        return self._screened(multiplier, mspe=True)

    def _screened(self, multiplier, *, mspe):
        table = self.table.copy()
        if multiplier is None:
            table["eligible"] = True
        else:
            # Compare without squaring huge/tiny outcomes in float arithmetic.
            power = 2 if mspe else 1
            limit = Fraction(float(multiplier)) * Fraction(self.treated.pre_rmse) ** power
            table["eligible"] = [
                treated or Fraction(pre) ** power <= limit
                for treated, pre in zip(table.treated, table.pre_rmse)
            ]
        fraction = table.loc[table.eligible, "at_least_treated_ratio"].mean()
        return replace(
            self,
            table=table,
            p_value=float(fraction),
            max_pre_mspe_ratio=multiplier if mspe else None,
            max_pre_rmse_ratio=None if mspe else multiplier,
        )


def placebo_test(
    estimator: Estimator | Callable[[Panel], Estimator],
    panel: Panel,
    *,
    donor_pool: Literal["untreated", "reassign"] = "untreated",
    n_jobs: int = 1,
    max_pre_rmse_ratio: float | None = None,
    max_pre_mspe_ratio: float | None = None,
) -> PlaceboResult:
    """Refit each unit as treated and rank its post/pre prediction error ratio.

    `donor_pool="reassign"` implements the Abadie, Diamond and Hainmueller
    (2010, pp. 501–503) reassignment scheme. Each pseudo-treated unit uses
    all other units, including the originally treated unit, as candidate
    donors. `"untreated"` keeps that unit out of each placebo fit, preserving
    this library's original design. An estimator instance is copied for each
    fit. A callable taking the assignment Panel can construct a fresh
    estimator with the corresponding unit-specific predictor specification.
    `n_jobs` assigns independent, complete fits to worker processes. The
    default is serial. Parallel results retain the panel's assignment order;
    estimator factories and fitted results must be serializable by joblib.
    Fitting exceptions propagate. Worker warnings are replayed in the calling
    process, and each estimate retains the estimator's diagnostic metadata.

    An optional screen compares pre-RMSE or pre-MSPE with a multiple of
    the treated unit's corresponding error. The treated fit stays eligible.
    The two cutoff scales are mutually exclusive. With B eligible placebos,
    `p_value = (1 + # eligible placebo ratios >= treated ratio) / (1 + B)`.
    The unfiltered rank has a randomization interpretation under a uniform
    assignment mechanism, a sharp null and an assignment-symmetric fitting
    rule. A treated-relative screen yields a descriptive selected-sample rank.

    A zero pre-error gives an infinite ratio for a nonzero post-error and a
    zero ratio for a zero post-error. `table` gives both RMSPE and MSPE ratios.
    Exact comparisons of the computed RMSPEs preserve ranks at extreme scales.
    """
    if not isinstance(panel, Panel):
        raise TypeError("panel must be a Panel")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    if donor_pool not in ("untreated", "reassign"):
        raise ValueError("donor_pool must be 'untreated' or 'reassign'")
    if panel.n_treated != 1 or (panel.n_control < 2 and donor_pool == "untreated"):
        raise ValueError("Placebo tests require one treated unit and at least one donor per fit")
    _check_cutoff(max_pre_rmse_ratio, "max_pre_rmse_ratio")
    _check_cutoff(max_pre_mspe_ratio, "max_pre_mspe_ratio")
    if max_pre_rmse_ratio is not None and max_pre_mspe_ratio is not None:
        raise ValueError("Choose one pre-fit cutoff scale, RMSE or MSPE")
    fixed_model = callable(getattr(estimator, "fit", None))
    if not fixed_model and not callable(estimator):
        raise TypeError("estimator must implement fit or be a panel-to-estimator factory")

    def fit_one(sample):
        model = estimator if fixed_model else estimator(sample)
        if not callable(getattr(model, "fit", None)):
            raise TypeError("The estimator factory must return an object with fit(panel)")
        fitted = deepcopy(model).fit(sample)
        _validate_estimate(sample, fitted, allow_subset_weights=True)
        return deepcopy(replace(fitted, metadata=dict(fitted.metadata)))

    def assignments():
        yield panel
        pool_size = panel.n_control + (donor_pool == "reassign")
        for i in range(panel.n_control):
            donors = [j for j in range(pool_size) if j != i]
            order = [*donors, i]
            yield Panel(
                panel.y[:, order],
                len(donors),
                panel.n_pre,
                panel.times,
                tuple(panel.units[j] for j in order),
            )

    if n_jobs == 1:
        fitted = [fit_one(sample) for sample in assignments()]
    else:
        from joblib import Parallel, delayed

        caller_filters = warnings.filters.copy()

        def fit_in_worker(sample):
            # A warning on a worker's stderr is otherwise invisible to the
            # calling notebook's warning filters and capture machinery.
            with warnings.catch_warnings(record=True) as caught:
                warnings.filters[:] = caller_filters
                result = fit_one(sample)
            return result, [(w.message, w.category, w.filename, w.lineno) for w in caught]

        workers = min(n_jobs, len(panel.units))
        results = Parallel(n_jobs=workers, backend="loky", max_nbytes=None)(
            delayed(fit_in_worker)(sample) for sample in assignments()
        )
        fitted = []
        registries = {}
        for result, emitted in results:
            fitted.append(result)
            for message, category, filename, lineno in emitted:
                warnings.warn_explicit(
                    message,
                    category,
                    filename,
                    lineno,
                    registry=registries.setdefault(filename, {}),
                )
    treated = fitted[0]
    fits = dict(zip(panel.units[: panel.n_control], fitted[1:]))

    rows = []
    for fit in (treated, *fits.values()):
        pre = fit.pre_rmse
        post = _root_mean_square(fit.effects.iloc[fit.n_pre :])
        ratio = post / pre if pre > 0 else (np.inf if post > 0 else 0.0)
        rows.append(
            {
                "att": fit.att,
                "pre_rmse": pre,
                "post_rmse": post,
                "rmspe_ratio": ratio,
            }
        )
    units = pd.Index([panel.units[-1], *fits], name="unit", tupleize_cols=False)
    table = pd.DataFrame(rows, index=units)
    table["eligible"] = True
    table["treated"] = np.arange(len(table)) == 0
    ratios = [_exact_ratio(post, pre) for post, pre in zip(table.post_rmse, table.pre_rmse)]
    table["at_least_treated_ratio"] = [ratio >= ratios[0] for ratio in ratios]
    with np.errstate(over="ignore", under="ignore"):
        table["mspe_ratio"] = table.rmspe_ratio.to_numpy() ** 2
    result = PlaceboResult(
        treated, fits, table, float(table.at_least_treated_ratio.mean()), donor_pool
    )
    if max_pre_rmse_ratio is not None:
        return result._screened(max_pre_rmse_ratio, mspe=False)
    if max_pre_mspe_ratio is not None:
        return result.with_pre_mspe_cutoff(max_pre_mspe_ratio)
    return result
