"""Synthetic control methods with a shared, explicit panel interface."""

from .backends import SCM, PySyncon
from .classic import SyntheticControl
from .individual import IndividualEstimate, IndividualSC, TwoStageIndividualSC
from .inference import PlaceboResult, placebo_test
from .panel import Panel
from .plotting import (
    LiteratureBenchmark,
    ReferenceComparison,
    plot_comparison,
    plot_effects,
    plot_paths,
    publication_style,
    reference_comparison_table,
    save_figure,
)
from .results import Estimate, Estimator, compare
from .scul import SCUL
from .sdid import DifferenceInDifferences, SDIDInference, SyntheticDID

__version__ = "0.1.0"
__all__ = [
    "Panel",
    "Estimate",
    "Estimator",
    "SyntheticControl",
    "SCUL",
    "IndividualSC",
    "TwoStageIndividualSC",
    "IndividualEstimate",
    "SyntheticDID",
    "SDIDInference",
    "DifferenceInDifferences",
    "PySyncon",
    "SCM",
    "compare",
    "placebo_test",
    "PlaceboResult",
    "plot_paths",
    "plot_effects",
    "plot_comparison",
    "publication_style",
    "save_figure",
    "LiteratureBenchmark",
    "ReferenceComparison",
    "reference_comparison_table",
]
