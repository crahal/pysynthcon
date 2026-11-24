"""Synthetic control methods with a shared, explicit panel interface."""

from .backends import (
    SCM,
    AugmentedSyntheticControl,
    DifferencedSyntheticControl,
    PenalizedSyntheticControl,
    PredictorSyntheticControl,
    PySyncon,
    RobustSyntheticControl,
)
from .classic import SyntheticControl
from .conformal import ConformalResult, conformal_confidence_set, conformal_test
from .conformal_plotting import plot_conformal_set
from .evaluation_plotting import plot_backtest, plot_donor_sensitivity
from .individual import IndividualEstimate, IndividualSC, TwoStageIndividualSC
from .inference import PlaceboResult, placebo_test
from .panel import Panel
from .placebo_plotting import plot_placebo_gaps
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
    "PredictorSyntheticControl",
    "AugmentedSyntheticControl",
    "DifferencedSyntheticControl",
    "PenalizedSyntheticControl",
    "RobustSyntheticControl",
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
    "ConformalResult",
    "conformal_test",
    "conformal_confidence_set",
    "plot_placebo_gaps",
    "plot_conformal_set",
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
