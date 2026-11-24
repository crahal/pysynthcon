"""Full-refit donor exclusions for descriptive synthetic-control sensitivity."""

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from .panel import Panel
from .results import Estimate, Estimator
from .validation import _validate_estimate


@dataclass
class DonorSensitivityResult:
    """Baseline and refitted estimates after named donor exclusions.

    ``baseline`` maps model names to full-pool estimates. ``estimates`` maps
    scenario names to model-name/Estimate dictionaries; each Estimate retains
    the complete fitted paths, signed donor weights and optimizer metadata.
    Metadata is independently copied after each fit so later fits cannot
    overwrite an earlier scenario's recorded diagnostics.
    ``table`` has one row per model and exclusion scenario. ``summary`` is
    indexed by model, and its minima/maxima cover exclusion scenarios only.

    These are descriptive sensitivity results, not confidence intervals or
    tests. ``sign_changed`` means strictly positive to strictly negative, or
    the reverse: an exactly zero estimate is not counted as a sign reversal.
    ``worst_scenario`` is the first scenario with the largest absolute ATT
    change, following the supplied scenario order.
    """

    baseline: dict[str, Estimate]
    estimates: dict[str, dict[str, Estimate]]
    table: pd.DataFrame
    summary: pd.DataFrame


def _exclusion_scenarios(panel: Panel, exclusions) -> dict[str, tuple[Any, ...]]:
    donors = panel.units[: panel.n_control]
    if exclusions is None:
        if panel.n_control < 2:
            raise ValueError("Default donor exclusions require at least two control units")
        names = [f"omit:{unit}" for unit in donors]
        if len(set(names)) != len(names):
            # Distinct labels such as 1 and '1' may have identical string forms.
            names = [f"omit:{i}:{unit}" for i, unit in enumerate(donors)]
        return {name: (unit,) for name, unit in zip(names, donors)}
    if not isinstance(exclusions, Mapping) or not exclusions:
        raise ValueError("exclusions must be a nonempty mapping of names to donor sequences")
    scenarios = {}
    for name, members in exclusions.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Exclusion scenario names must be nonempty strings")
        if isinstance(members, (str, bytes)):
            raise ValueError(f"Exclusion {name!r} must contain a sequence of donor labels")
        try:
            members = tuple(members)
            excluded = set(members)
        except TypeError as exc:
            raise ValueError(f"Exclusion {name!r} must contain hashable donor labels") from exc
        if not members or len(excluded) != len(members):
            raise ValueError(f"Exclusion {name!r} must contain nonempty, unique donor labels")
        if not excluded <= set(donors):
            raise ValueError(f"Exclusion {name!r} contains unknown or treated units")
        if len(excluded) == panel.n_control:
            raise ValueError(f"Exclusion {name!r} must leave at least one control unit")
        # Canonical panel order makes equivalent exclusions directly comparable.
        scenarios[name] = tuple(unit for unit in donors if unit in excluded)
    return scenarios


def _without_donors(panel: Panel, excluded: tuple) -> Panel:
    remaining = [i for i in range(panel.n_control) if panel.units[i] not in excluded]
    columns = remaining + list(range(panel.n_control, panel.y.shape[1]))
    return Panel(
        panel.y[:, columns],
        len(remaining),
        panel.n_pre,
        panel.times,
        tuple(panel.units[i] for i in columns),
    )


def donor_sensitivity(
    panel: Panel,
    estimators: Mapping[str, Callable[[], Estimator]],
    *,
    exclusions: Mapping[str, Sequence[Any]] | None = None,
) -> DonorSensitivityResult:
    """Refit every estimator after each donor or named group is excluded.

    By default, omit each eligible donor in turn, including donors receiving
    zero or negative baseline weight. Alternatively supply named exclusions,
    such as ``{"bordering_regions": ["A", "B"]}``. Other donor order,
    treatment timing, observed outcomes and treated-unit membership are
    preserved. Every model is fully refitted, including its tuning and any
    nearest-neighbor selection: this is not weight deletion/renormalization.

    ``estimators`` supplies zero-argument factories, each returning a fresh
    object with a ``fit(panel)`` method. Factories must preserve the same
    fitting protocol across scenarios. Set random seeds in the factories to
    make runs reproducible; even then, changed donor pools change the random
    draws and this descriptive comparison cannot isolate sampling noise.
    Covariate-aware adapters must align their covariates to the received
    panel's remaining units inside ``fit``.

    No iteration, tuning or resampling budgets are reduced. The default costs
    ``n_models * (1 + n_control)`` fits. Any failed or malformed fit raises an
    error identifying the model and scenario; results are never silently
    omitted. This diagnostic does not establish absence of spillovers and
    does not provide a causal confidence interval or p-value.
    """
    if not isinstance(panel, Panel):
        raise TypeError("panel must be a Panel")
    if not isinstance(estimators, Mapping) or not estimators:
        raise ValueError("estimators must be a nonempty mapping of names to factories")
    for name, factory in estimators.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Estimator names must be nonempty strings")
        if not callable(factory):
            raise TypeError(f"Estimator {name!r} must be a zero-argument factory")
    scenarios = _exclusion_scenarios(panel, exclusions)

    # Retaining references prevents id reuse from falsely flagging new objects.
    instances = {}

    def fit_one(name, factory, fitting_panel, scenario):
        try:
            estimator = factory()
            if id(estimator) in instances:
                raise ValueError("Each factory call must return a fresh estimator object")
            instances[id(estimator)] = estimator
            if not callable(getattr(estimator, "fit", None)):
                raise TypeError("Estimator factories must return an object with fit(panel)")
            # Give each fit its own data copy, including the baseline fit.
            supplied = Panel(
                fitting_panel.y,
                fitting_panel.n_control,
                fitting_panel.n_pre,
                fitting_panel.times,
                fitting_panel.units,
            )
            estimate = estimator.fit(supplied)
            _validate_estimate(fitting_panel, estimate)
            if not np.isfinite(estimate.att) or not np.isfinite(estimate.pre_rmse):
                raise ValueError("Estimate ATT and pre-treatment RMSE must be finite")
            return replace(estimate, metadata=deepcopy(dict(estimate.metadata)))
        except Exception as exc:
            raise RuntimeError(
                f"Donor sensitivity failed for model {name!r} in scenario {scenario!r}: {exc}"
            ) from exc

    baseline = {
        name: fit_one(name, factory, panel, "baseline") for name, factory in estimators.items()
    }
    fits, rows = {}, []
    for scenario, excluded in scenarios.items():
        fitting_panel = _without_donors(panel, excluded)
        fits[scenario] = {}
        for name, factory in estimators.items():
            estimate = fit_one(name, factory, fitting_panel, scenario)
            fits[scenario][name] = estimate
            base = baseline[name]
            change = estimate.att - base.att
            if not np.isfinite(change):
                raise RuntimeError(
                    f"Donor sensitivity ATT change is nonfinite for model {name!r} "
                    f"in scenario {scenario!r}"
                )
            rows.append(
                {
                    "model": name,
                    "scenario": scenario,
                    "excluded_units": excluded,
                    "att": estimate.att,
                    "baseline_att": base.att,
                    "att_change": change,
                    "pre_rmse": estimate.pre_rmse,
                    "baseline_pre_rmse": base.pre_rmse,
                    "donor_count": fitting_panel.n_control,
                    "sign_changed": bool(np.sign(estimate.att) == -np.sign(base.att))
                    and estimate.att != 0
                    and base.att != 0,
                }
            )
    table = pd.DataFrame(rows)
    summary = []
    for name, base in baseline.items():
        model_rows = table.loc[table.model == name]
        worst = model_rows.loc[model_rows.att_change.abs().idxmax()]
        summary.append(
            {
                "model": name,
                "baseline_att": base.att,
                "min_att": float(model_rows.att.min()),
                "max_att": float(model_rows.att.max()),
                "max_abs_change": float(abs(worst.att_change)),
                "worst_scenario": worst.scenario,
                "n_scenarios": len(model_rows),
            }
        )
    return DonorSensitivityResult(baseline, fits, table, pd.DataFrame(summary).set_index("model"))
