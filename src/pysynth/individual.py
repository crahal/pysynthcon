"""Individual synthetic controls and event-time aggregation."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, Mapping

import numpy as np
import pandas as pd

from ._optim import simplex_fit
from ._r import run_reference
from .panel import Panel
from .results import Estimate, make_estimate


@dataclass
class IndividualEstimate:
    """Individual fits with equal-person or explicitly weighted aggregation."""

    estimates: dict
    summary: pd.DataFrame
    att: float


@dataclass(kw_only=True)
class IndividualSC:
    """Individual Synthetic Control using the original MSCMT optimizer.

    Fit nonnegative weights summing to one for each individual, using all
    donors in their panel. See :class:`TwoStageIndividualSC` for nearest-
    neighbor donor screening before weight fitting.
    Optional donor resampling averages ``n_bootstrap`` fits, following the
    individual-control resampling idea of Vagni and Breen (2021). These draws
    are sensitivity diagnostics, not calibrated confidence intervals.

    ``backend='mscmt'`` uses MSCMT 1.4.0's complete default optimization and
    averages 50 donor-resampled fits, as in the costofcare replication script.
    R's sampling RNG and optimizer seeds are explicit. ``predictors`` can
    supply the additional pretreatment covariate series required by a study.
    A reported WNNLS failure is repaired only by solving its unchanged
    weighted simplex quadratic with an independently checked optimality gap;
    every such call is recorded in ``inner_solver_repairs`` metadata.
    With no covariates the outcome is the sole predictor group; MSCMT then
    correctly has no nontrivial outer predictor-weight search to perform.

    ``backend='fixed_metric'`` explicitly requests the earlier simplified
    convex matcher. It is not an exact implementation of the published ISC
    workflow. Applying ISC to aggregate panels is a methodological comparison,
    not a replication of the original survey-data results.
    """

    n_bootstrap: int = 50
    random_state: int = 42
    optimizer_seed: int = 1
    backend: str = "mscmt"
    r_library: str | None = None
    _method: ClassVar[str] = "ISC"

    def _select_donors(self, x: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
        return np.arange(x.shape[1]), {"donor_selection": "all"}

    def fit(
        self,
        panel: Panel,
        *,
        features: np.ndarray | None = None,
        predictors: Mapping[str, np.ndarray] | None = None,
    ) -> Estimate:
        """Fit one case using all pretreatment observations.

        ``predictors`` maps names to n_pre-by-units arrays in Panel order.
        MSCMT uses each series as one predictor-weight group without temporal
        aggregation; 2SISC's pysyncon backend matches their temporal means.
        The outcome group is always included. For fixed_metric,
        ``features`` replaces the outcome design with a rows-by-units array.
        2SISC screens on the stacked, unscaled pretreatment design; scale
        features explicitly when units differ. Post outcomes never enter it.
        """
        return self._fit_case(panel, features=features, predictors=predictors)[0]

    def _fit_case(self, panel, *, features=None, predictors=None, sampling_state=None):
        """Return one estimate and the R sampling state for a subsequent case."""
        if panel.n_treated != 1:
            raise ValueError("Use fit_many for multiple individual treated cases")
        if self.backend not in {"mscmt", "fixed_metric"}:
            raise ValueError("backend must be 'mscmt' or 'fixed_metric'")
        if (
            isinstance(self.n_bootstrap, (bool, np.bool_))
            or not isinstance(self.n_bootstrap, (int, np.integer))
            or self.n_bootstrap < 0
        ):
            raise ValueError("n_bootstrap must be a nonnegative integer")
        for name, seed in [
            ("random_state", self.random_state),
            ("optimizer_seed", self.optimizer_seed),
        ]:
            if (
                isinstance(seed, (bool, np.bool_))
                or not isinstance(seed, (int, np.integer))
                or not 0 <= seed <= 2147483647
            ):
                raise ValueError(f"{name} must be an integer between 0 and 2147483647")
        if self.backend == "fixed_metric" and predictors:
            raise ValueError(
                "Use features for fixed_metric; predictors are MSCMT time-series groups"
            )
        if self.backend == "mscmt" and features is not None:
            raise ValueError(
                "MSCMT requires named time-series predictor groups; use predictors={name: matrix}. "
                "features is only available with backend='fixed_metric'."
            )
        groups = {}
        for name, value in (predictors or {}).items():
            if not isinstance(name, str) or not name or name == "outcome":
                raise ValueError("Predictor names must be nonempty strings other than 'outcome'")
            matrix = np.asarray(value, dtype=float)
            if matrix.shape != (panel.n_pre, panel.y.shape[1]) or not np.isfinite(matrix).all():
                raise ValueError(
                    "Each predictor must be a finite n_pre-by-units matrix in Panel order"
                )
            groups[name] = matrix
        design = panel.y[: panel.n_pre] if features is None else np.asarray(features, dtype=float)
        if groups:
            design = np.vstack([design, *groups.values()])
        if (
            design.ndim != 2
            or design.shape[1] != panel.y.shape[1]
            or not len(design)
            or not np.isfinite(design).all()
        ):
            raise ValueError("features must be a finite rows-by-units matrix in Panel unit order")
        x, target = design[:, : panel.n_control], design[:, -1]
        chosen, selection = self._select_donors(x, target)
        k = len(chosen)
        if self.backend == "mscmt":
            return self._fit_reference(panel, chosen, selection, groups, sampling_state)
        rng = np.random.default_rng(self.random_state)
        draws, atts = [], []
        for _ in range(max(1, self.n_bootstrap)):
            subset = rng.choice(chosen, size=k, replace=True) if self.n_bootstrap else chosen
            w, _ = simplex_fit(x[:, subset], target)
            full = np.zeros(panel.n_control)
            np.add.at(full, subset, w)
            draws.append(full)
            atts.append(
                float(np.mean(panel.treated[panel.n_pre :] - panel.control[panel.n_pre :] @ full))
            )
        estimate = make_estimate(
            panel,
            self._method,
            np.mean(draws, axis=0),
            metadata={
                **selection,
                "backend": "fixed_metric_adaptation",
                "selected_donors": [panel.units[i] for i in chosen],
                "n_eligible_donors": panel.n_control,
                "n_selected_donors": k,
                "matching_features": (
                    "pre_treatment_outcomes" if features is None else "supplied_features"
                ),
                "n_bootstrap": self.n_bootstrap,
                "bootstrap_atts": atts if self.n_bootstrap else [],
            },
        )
        return estimate, None

    def _fit_reference(self, panel, chosen, selection, groups, sampling_state):
        columns = np.r_[chosen, panel.n_control]
        matrices = {"outcomes": panel.y[:, columns]}
        for i, value in enumerate(groups.values(), 1):
            matrices[f"predictor_{i}"] = value[:, columns]
        result = run_reference(
            "isc",
            matrices,
            {
                "n_pre": panel.n_pre,
                "n_bootstrap": int(self.n_bootstrap),
                "random_state": int(self.random_state),
                "optimizer_seed": int(self.optimizer_seed),
                "predictor_names": list(groups),
                "sampling_state": sampling_state,
            },
            r_library=self.r_library,
        )
        draw_weights = np.atleast_2d(result["draw_weights"])
        full = np.zeros(panel.n_control)
        full[chosen] = np.atleast_1d(result["weights"])
        atts = np.mean(
            panel.treated[panel.n_pre :, None]
            - panel.control[panel.n_pre :, chosen] @ draw_weights.T,
            axis=0,
        )
        estimate = make_estimate(
            panel,
            self._method,
            full,
            metadata={
                **selection,
                "backend": "original_mscmt",
                "mscmt_version": result["mscmt_version"],
                "r_version": result["r_version"],
                "reference_commit": "6e7537d67b86fb08709d35b035c9ee9f45e83e93",
                "selected_donors": [panel.units[i] for i in chosen],
                "n_eligible_donors": panel.n_control,
                "n_selected_donors": len(chosen),
                "matching_features": "mscmt_time_series_predictors",
                "predictor_groups": ["outcome", *groups],
                "predictor_aggregation": "id",
                "outer_optimizer": "DEoptC (MSCMT defaults)",
                "inner_optimizer": "wnnlsOpt (MSCMT defaults)",
                "random_state": self.random_state,
                "optimizer_seed": self.optimizer_seed,
                "n_bootstrap": self.n_bootstrap,
                "bootstrap_atts": atts.tolist() if self.n_bootstrap else [],
                "bootstrap_donor_indices": [
                    chosen[np.atleast_1d(x)].tolist() for x in result["subsets"]
                ],
                "bootstrap_mspe": result["mspe"],
                "predictor_weights": result["predictor_weights"],
                "inner_solver_repairs": result["inner_solver_repairs"],
            },
        )
        return estimate, result["sampling_state"]

    def fit_many(
        self,
        panels: Mapping,
        *,
        person_weights: Mapping | None = None,
        predictors: Mapping | None = None,
    ) -> IndividualEstimate:
        """Fit case-specific panels, aligning periods by position relative to onset.

        All panels must use the same temporal frequency. At each event time,
        average only observed cases and report their count.
        ``predictors`` optionally maps each case key to its named pretreatment
        predictor matrices, as accepted by :meth:`fit`. Overall ATT is the
        weighted mean of each person's post-treatment mean, giving long
        observed trajectories no extra person weight. Explicit person weights
        must be finite and strictly positive and cover exactly the case keys.

        With MSCMT, donor sampling advances one R random stream through the
        supplied case order, matching the original ISC parallel workflow.
        Its state is saved before each case's optimizer resets its own seed.
        Reordering cases can therefore change resamples. A standalone fit
        always starts from ``random_state``.
        """
        if not panels:
            raise ValueError("panels must not be empty")
        if person_weights is not None and set(person_weights) != set(panels):
            raise ValueError("person_weights must cover exactly the panel keys")
        if predictors is not None and set(predictors) != set(panels):
            raise ValueError("predictors must cover exactly the panel keys")
        weights = {
            key: 1.0 if person_weights is None else float(person_weights[key]) for key in panels
        }
        if any(not np.isfinite(w) or w <= 0 for w in weights.values()):
            raise ValueError("person weights must be finite and strictly positive")
        fits, sampling_state = {}, None
        for key, panel in panels.items():
            fits[key], sampling_state = self._fit_case(
                panel,
                predictors=None if predictors is None else predictors[key],
                sampling_state=sampling_state,
            )
        rows = []
        for key, fit in fits.items():
            for event, effect in zip(np.arange(len(fit.observed)) - fit.n_pre, fit.effects):
                rows.append((event, effect, weights[key]))
        data = pd.DataFrame(rows, columns=["event_time", "effect", "weight"])
        summary = data.groupby("event_time").agg(n=("effect", "size"), weight_sum=("weight", "sum"))
        data["weighted_effect"] = data.effect * data.weight
        summary["effect"] = data.groupby("event_time").weighted_effect.sum() / summary.weight_sum
        att = sum(weights[key] * fit.att for key, fit in fits.items()) / sum(weights.values())
        return IndividualEstimate(fits, summary, float(att))


@dataclass(kw_only=True)
class TwoStageIndividualSC(IndividualSC):
    """Two-Stage Individual Synthetic Control (Petrillo et al., 2025).

    Stage one selects up to ``n_neighbors`` eligible donors by squared
    Euclidean distance on raw, stacked pretreatment outcomes and named
    covariate series. The default ``backend='pysyncon'`` follows the authors'
    Python fitting specification: nested SCM matches each series' temporal
    mean, selecting predictor importance against all pretreatment outcomes.
    It performs no within-donor resampling (``n_bootstrap=0``).

    ``n_neighbors`` counts donors, excluding the treated individual. Ties
    follow Panel unit order. When k exceeds the available donor count, all
    eligible donors are used. The published code's KDTree query includes the
    treated case in ``k_n``; absent ties, its query needs ``k_n=k+1`` for k
    donors. Caller-supplied panels determine eligibility and time windows.

    ``options`` passes optimizer settings to the checked PySyncon adapter.
    Explicit ``backend='mscmt', n_bootstrap=50`` retains PySynth's earlier
    variant: screen donors, then run ISC's time-series predictor groups and
    donor resampling. This variant is distinct from the authors' two-stage
    Python pipeline. ``backend='fixed_metric'`` remains a separate adaptation.
    """

    backend: str = "pysyncon"
    n_bootstrap: int = 0
    n_neighbors: int = 10
    options: dict = field(default_factory=dict)
    _method: ClassVar[str] = "2SISC"

    def _fit_case(self, panel, *, features=None, predictors=None, sampling_state=None):
        self.backend_ = None
        if self.backend not in {"pysyncon", "mscmt", "fixed_metric"}:
            raise ValueError("backend must be 'pysyncon', 'mscmt' or 'fixed_metric'")
        if self.backend != "pysyncon":
            if self.options:
                raise ValueError("options is only available with backend='pysyncon'")
            return super()._fit_case(
                panel, features=features, predictors=predictors, sampling_state=sampling_state
            )
        if panel.n_treated != 1:
            raise ValueError("Use fit_many for multiple individual treated cases")
        if (
            isinstance(self.n_bootstrap, (bool, np.bool_))
            or not isinstance(self.n_bootstrap, (int, np.integer))
            or self.n_bootstrap != 0
        ):
            raise ValueError(
                "backend='pysyncon' requires n_bootstrap=0; the authors' two-stage "
                "fit does not resample donors. Use backend='mscmt' explicitly for that variant."
            )
        if features is not None:
            raise ValueError(
                "The published two-stage fit requires named time-series predictors; "
                "features is only available with backend='fixed_metric'."
            )
        groups = {}
        for name, value in (predictors or {}).items():
            if not isinstance(name, str) or not name or name == "outcome":
                raise ValueError("Predictor names must be nonempty strings other than 'outcome'")
            matrix = np.asarray(value, dtype=float)
            if matrix.shape != (panel.n_pre, panel.y.shape[1]) or not np.isfinite(matrix).all():
                raise ValueError(
                    "Each predictor must be a finite n_pre-by-units matrix in Panel order"
                )
            groups[name] = matrix
        design = np.vstack([panel.y[: panel.n_pre], *groups.values()])
        chosen, selection = self._select_donors(design[:, : panel.n_control], design[:, -1])
        predictor_order = sorted(["outcome", *groups])
        full = np.zeros(panel.n_control)
        metadata = {
            **selection,
            "backend": "pysyncon_mean_predictors",
            "reference_commit": "6e7537d67b86fb08709d35b035c9ee9f45e83e93",
            "reference_path": "src/utils.py:sc_b",
            "method_reference": "Petrillo et al. (2025), SERPS 2025004, section 2.3",
            "selected_donors": [panel.units[i] for i in chosen],
            "n_eligible_donors": panel.n_control,
            "n_selected_donors": len(chosen),
            "neighbor_count_convention": "donors_excluding_treated",
            "matching_features": "raw_stacked_pre_treatment_series",
            "predictor_groups": predictor_order,
            "predictor_aggregation": "mean",
            "n_bootstrap": 0,
            "bootstrap_atts": [],
        }
        if len(chosen) == 1:
            # The only feasible simplex weight is one, independent of V.
            full[chosen[0]] = 1.0
            metadata["optimization"] = {"solution": "single_donor_simplex"}
        else:
            from .backends import PySyncon, _canonical_frame, _require

            upstream = _require("pysyncon")
            columns = np.r_[chosen, panel.n_control]
            # Internal integer labels avoid upstream sorting/type restrictions;
            # returned weights and paths retain the original Panel labels.
            reduced = Panel(panel.y[:, columns], len(chosen), panel.n_pre)
            frame = _canonical_frame(reduced)
            predictor_columns = {"outcome": "outcome"}
            for i, (name, matrix) in enumerate(groups.items()):
                column = f"covariate_{i}"
                values = np.full_like(reduced.y, np.nan)
                values[: panel.n_pre] = matrix[:, columns]
                frame[column] = values.T.ravel()
                predictor_columns[name] = column
            dataprep = upstream.Dataprep(
                foo=frame,
                # The authors' pivoted feature columns are sorted by name.
                predictors=[predictor_columns[name] for name in predictor_order],
                predictors_op="mean",
                dependent="outcome",
                unit_variable="unit",
                time_variable="time",
                treatment_identifier=reduced.n_control,
                controls_identifier=list(range(reduced.n_control)),
                time_predictors_prior=list(range(reduced.n_pre)),
                time_optimize_ssr=list(range(reduced.n_pre)),
            )
            means = np.vstack(
                [panel.y[: panel.n_pre, columns].mean(axis=0)]
                + [matrix[:, columns].mean(axis=0) for matrix in groups.values()]
            )
            constant = np.flatnonzero(np.ptp(means, axis=1) == 0)
            if len(constant):
                names = ["outcome", *groups]
                raise ValueError(
                    "The mean-predictor specification has constant predictors across the "
                    f"selected units: {[names[i] for i in constant]}. "
                    "Upstream variance scaling is undefined; revise the explicit specification."
                )
            # Keep recorded settings independent of later caller edits,
            # including nested optimizer dictionaries and custom V arrays.
            adapter = PySyncon("synth", dataprep=dataprep, options=deepcopy(self.options))
            fitted = adapter.fit(reduced)
            self.backend_ = adapter.backend_
            full[chosen] = fitted.weights.to_numpy()
            metadata.update(fitted.metadata)
            metadata["predictor_weights"] = np.asarray(self.backend_.V).tolist()
        return make_estimate(panel, self._method, full, metadata=metadata), None

    def _select_donors(self, x: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
        if (
            isinstance(self.n_neighbors, (bool, np.bool_))
            or not isinstance(self.n_neighbors, (int, np.integer))
            or self.n_neighbors < 1
        ):
            raise ValueError("n_neighbors must be a positive integer")
        with np.errstate(over="ignore", under="ignore"):
            difference = x - target[:, None]
            distances = np.sum(difference**2, axis=0)
            order_by = distances
            if not np.isfinite(distances).all() or np.any(
                (distances == 0) & np.any(difference != 0, axis=0)
            ):
                # Norms preserve the ordering when squared distances overflow
                # or underflow. Half-scale only when subtraction itself overflows.
                if not np.isfinite(difference).all():
                    difference = x * 0.5 - target[:, None] * 0.5
                magnitude = np.max(np.abs(difference), axis=0)
                nonzero = magnitude > 0
                order_by = np.full(x.shape[1], -np.inf)
                normalized = difference[:, nonzero] / magnitude[nonzero]
                order_by[nonzero] = np.log(magnitude[nonzero]) + 0.5 * np.log(
                    np.sum(normalized**2, axis=0)
                )
        k = min(self.n_neighbors, x.shape[1])
        chosen = np.sort(np.argsort(order_by, kind="stable")[:k])
        return chosen, {
            "donor_selection": "nearest_neighbors",
            "n_neighbors": self.n_neighbors,
            "squared_distances": [
                float(value) if np.isfinite(value) else None for value in distances
            ],
        }
