"""Synthetic control methods with a shared, explicit panel interface."""

from .backends import SCM, PySyncon
from .classic import SyntheticControl
from .evaluation_plotting import plot_backtest, plot_donor_sensitivity
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
    publication_title_font,
    reference_comparison_table,
    save_figure,
)
from .results import Estimate, Estimator, compare
from .scul import SCUL
from .sdid import DifferenceInDifferences, SDIDInference, SyntheticDID
from .sensitivity import DonorSensitivityResult, donor_sensitivity
from .validation import BacktestResult, SelectionResult, backtest, select_estimator

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
    "publication_title_font",
    "save_figure",
    "LiteratureBenchmark",
    "ReferenceComparison",
    "reference_comparison_table",
    "backtest",
    "select_estimator",
    "BacktestResult",
    "SelectionResult",
    "donor_sensitivity",
    "DonorSensitivityResult",
    "plot_backtest",
    "plot_donor_sensitivity",
]
