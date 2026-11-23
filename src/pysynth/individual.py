"""Individual synthetic controls and event-time aggregation."""

from dataclasses import dataclass
from typing import ClassVar, Mapping

import numpy as np
import pandas as pd

from ._optim import simplex_fit
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
    """Vanilla Individual Synthetic Control (ISC) over all eligible donors.

    Fit nonnegative weights summing to one for each individual, using all
    donors in their panel. See :class:`TwoStageIndividualSC` for nearest-
    neighbor donor screening before weight fitting.
    Optional donor resampling averages ``n_bootstrap`` fits, following the
    individual-control resampling idea of Vagni and Breen (2021). These draws
    are sensitivity diagnostics, not calibrated confidence intervals.

    The original paper uses MSCMT predictor weighting and survey microdata;
    this implementation does not claim numerical identity with that workflow.
    """

    n_bootstrap: int = 0
    random_state: int = 0
    _method: ClassVar[str] = "ISC"

    def _select_donors(self, x: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
        return np.arange(x.shape[1]), {"donor_selection": "all"}

    def fit(self, panel: Panel, *, features: np.ndarray | None = None) -> Estimate:
        """Fit one case; features, if given, are rows-by-units in Panel order.

        Caller-supplied features must be measured before treatment. They are
        used unscaled for fitting and, in 2SISC, screening. Scale them explicitly
        when units differ. Post-treatment outcomes never enter either stage.
        """
        if panel.n_treated != 1:
            raise ValueError("Use fit_many for multiple individual treated cases")
        if (
            isinstance(self.n_bootstrap, (bool, np.bool_))
            or not isinstance(self.n_bootstrap, (int, np.integer))
            or self.n_bootstrap < 0
        ):
            raise ValueError("n_bootstrap must be a nonnegative integer")
        design = panel.y[: panel.n_pre] if features is None else np.asarray(features, dtype=float)
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
        return make_estimate(
            panel,
            self._method,
            np.mean(draws, axis=0),
            metadata={
                **selection,
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

    def fit_many(
        self, panels: Mapping, *, person_weights: Mapping | None = None
    ) -> IndividualEstimate:
        """Fit case-specific panels, aligning periods by position relative to onset.

        All panels must use the same temporal frequency. At each event time,
        average only observed cases and report their count. Overall ATT is the
        weighted mean of each person's post-treatment mean, giving long
        observed trajectories no extra person weight. Explicit person weights
        must be finite and strictly positive and cover exactly the case keys.
        """
        if not panels:
            raise ValueError("panels must not be empty")
        if person_weights is not None and set(person_weights) != set(panels):
            raise ValueError("person_weights must cover exactly the panel keys")
        weights = {
            key: 1.0 if person_weights is None else float(person_weights[key]) for key in panels
        }
        if any(not np.isfinite(w) or w <= 0 for w in weights.values()):
            raise ValueError("person weights must be finite and strictly positive")
        fits = {key: self.fit(panel) for key, panel in panels.items()}
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
