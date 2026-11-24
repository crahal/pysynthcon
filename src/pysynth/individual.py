"""Individual synthetic controls and event-time aggregation."""

from dataclasses import dataclass
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

        For MSCMT, ``predictors`` maps names to n_pre-by-units arrays in Panel
        order. Each series is one predictor-weight group, with no temporal
        aggregation. The outcome group is always included. For fixed_metric,
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
    """Two-Stage Individual Synthetic Control (2SISC).

    Stage one selects up to ``n_neighbors`` eligible donors by squared
    Euclidean distance to the treated individual's pretreatment outcomes,
    or to caller-supplied pretreatment features. Stage two fits ISC's convex
    weights within that selected group, using the same matching features.
    This is nearest-neighbor screening, not k-means clustering.

    ``n_neighbors`` counts donors, excluding the treated individual. Ties
    follow Panel unit order. When k exceeds the available donor count, all
    eligible donors are used. Selected donors retain their original order,
    making the full-pool fit identical to vanilla ISC.

    Donor resampling, when requested, is conditional on the stage-one pool;
    it does not rerun selection or estimate selection uncertainty.
    """

    n_neighbors: int = 10
    _method: ClassVar[str] = "2SISC"

    def _select_donors(self, x: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
        if (
            isinstance(self.n_neighbors, (bool, np.bool_))
            or not isinstance(self.n_neighbors, (int, np.integer))
            or self.n_neighbors < 1
        ):
            raise ValueError("n_neighbors must be a positive integer")
        distances = np.sum((x - target[:, None]) ** 2, axis=0)
        k = min(self.n_neighbors, x.shape[1])
        chosen = np.sort(np.argsort(distances, kind="stable")[:k])
        return chosen, {
            "donor_selection": "nearest_neighbors",
            "n_neighbors": self.n_neighbors,
            "squared_distances": distances.tolist(),
        }
