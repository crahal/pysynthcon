"""Validated balanced panels with explicit treatment timing."""

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Panel:
    """A balanced panel, with time in rows and control units before treated units.

    Prefer :meth:`from_frame` for labelled data. Every treated unit begins
    treatment at ``times[n_pre]``. Missing observations and duplicate labels
    are rejected; imputation and donor eligibility are research decisions.
    """

    y: np.ndarray
    n_control: int
    n_pre: int
    times: pd.Index | None = None
    units: tuple | None = None

    def __post_init__(self):
        y = np.array(self.y, dtype=float, copy=True)
        if y.ndim != 2 or not np.isfinite(y).all():
            raise ValueError("y must be a finite two-dimensional time-by-unit matrix")
        if not isinstance(self.n_control, Integral) or not 1 <= self.n_control < y.shape[1]:
            raise ValueError("n_control must leave at least one control and one treated unit")
        if not isinstance(self.n_pre, Integral) or not 2 <= self.n_pre < y.shape[0]:
            raise ValueError("n_pre must leave at least two pre-periods and one post-period")
        times = pd.Index(range(y.shape[0])) if self.times is None else pd.Index(self.times)
        units = tuple(range(y.shape[1])) if self.units is None else tuple(self.units)
        if len(times) != y.shape[0] or not times.is_unique or times.hasnans:
            raise ValueError("times must be unique, nonmissing, and match the rows of y")
        if not times.is_monotonic_increasing:
            raise ValueError("times must be sorted in increasing order")
        if len(units) != y.shape[1] or not pd.Index(units).is_unique or pd.Index(units).hasnans:
            raise ValueError("units must be unique, nonmissing, and match the columns of y")
        y.flags.writeable = False
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "times", times.copy())
        object.__setattr__(self, "units", units)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        unit: str,
        time: str,
        outcome: str,
        treatment_time: Any,
        treated_unit: Any = None,
        treated_units: Sequence | None = None,
        controls: Sequence | None = None,
    ) -> "Panel":
        """Convert long data; ``treatment_time`` is the first affected time.

        Supply exactly one of ``treated_unit`` and ``treated_units``. Explicit
        ``controls`` sets both eligibility and order. Otherwise controls retain
        their first appearance order. Treatment must fall on an observed time.
        """
        if (treated_unit is None) == (treated_units is None):
            raise ValueError("Supply exactly one of treated_unit and treated_units")
        treated = [treated_unit] if treated_units is None else list(treated_units)
        if not treated or len(set(treated)) != len(treated):
            raise ValueError("treated units must be nonempty and unique")
        missing = {unit, time, outcome} - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if frame[[unit, time]].isna().any().any():
            raise ValueError("Unit and time labels may not be missing")
        donors = (
            [u for u in frame[unit].unique() if u not in treated]
            if controls is None
            else list(controls)
        )
        if not donors or len(set(donors)) != len(donors) or set(donors) & set(treated):
            raise ValueError("Controls must be nonempty, unique, and disjoint from treated units")
        order = donors + treated
        if not set(order) <= set(frame[unit]):
            raise ValueError("A requested unit is absent from the data")
        selected = frame.loc[frame[unit].isin(order), [unit, time, outcome]]
        if selected.duplicated([unit, time]).any():
            raise ValueError("Duplicate unit-time observations")
        wide = selected.pivot(index=time, columns=unit, values=outcome).sort_index()
        if treatment_time not in wide.index:
            raise ValueError("treatment_time must be an observed time (first affected period)")
        return cls(
            wide[order].to_numpy(),
            len(donors),
            wide.index.get_loc(treatment_time),
            wide.index,
            tuple(order),
        )

    @property
    def control(self) -> np.ndarray:
        """Time-by-donor outcome matrix."""
        return self.y[:, : self.n_control]

    @property
    def treated(self) -> np.ndarray:
        """Equal-weight average outcome of the treated units."""
        return self.y[:, self.n_control :].mean(axis=1)

    @property
    def n_treated(self) -> int:
        """Number of simultaneously treated units."""
        return self.y.shape[1] - self.n_control

    @property
    def n_post(self) -> int:
        """Number of post-treatment periods."""
        return self.y.shape[0] - self.n_pre

    def to_frame(self) -> pd.DataFrame:
        """Return long data using canonical names ``unit``, ``time``, ``outcome``."""
        return (
            pd.DataFrame(self.y, index=self.times, columns=self.units)
            .rename_axis(index="time", columns="unit")
            .reset_index()
            .melt(id_vars="time", value_name="outcome")
        )
