"""Synthetic control methods with a shared, explicit panel interface."""

from .backends import SCM, PySyncon
from .classic import SyntheticControl
from .individual import IndividualEstimate, IndividualSC
from .inference import PlaceboResult, placebo_test
from .panel import Panel
from .results import Estimate, Estimator, compare
from .scul import SCUL
from .sdid import DifferenceInDifferences, SyntheticDID

__version__ = "0.1.0"
__all__ = [
    "Panel",
    "Estimate",
    "Estimator",
    "SyntheticControl",
    "SCUL",
    "IndividualSC",
    "IndividualEstimate",
    "SyntheticDID",
    "DifferenceInDifferences",
    "PySyncon",
    "SCM",
    "compare",
    "placebo_test",
    "PlaceboResult",
]
