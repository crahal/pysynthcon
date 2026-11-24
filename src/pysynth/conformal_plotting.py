"""Publication figures for pointwise conformal inference."""

from numbers import Real

import numpy as np
import pandas as pd

from .plotting import (
    BLUE,
    INK,
    LITERATURE,
    MUTED,
    _axes_style,
    _figure_grid,
    _finish_figure,
    _time,
    _time_axis,
    _validate,
    publication_style,
)
from .results import Estimate


def plot_conformal_set(
    grid: pd.DataFrame,
    estimate: Estimate | None = None,
    *,
    alpha: float | None = None,
    reference_bounds: pd.DataFrame | None = None,
    reference_label: str = "Reference endpoints",
    title: str | None = None,
    subtitle: str | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
    note: str | None = None,
    figsize: tuple | None = None,
):
    """Plot tested and accepted effect candidates by date. Return (Figure, Axes).

    Pass the long table from :func:`pysynth.conformal_confidence_set`. Each accepted candidate
    has its own mark. There is no interpolation between grid values or dates.
    Supply the ordinary SCM fit on the same panel for observed-minus-synthetic
    point markers. The level comes from the table's `attrs["alpha"]`. Pass it
    explicitly for a table read from CSV to include the level in the legend. Optional
    reference endpoints must have columns `lower`, `upper` indexed by the
    same post-treatment dates. The sets provide pointwise inference.
    """
    from matplotlib.transforms import ScaledTranslation

    required = {"time", "candidate", "p_value", "accepted"}
    if not isinstance(grid, pd.DataFrame) or not required.issubset(grid):
        raise ValueError("grid must have time, candidate, p_value and accepted columns")
    if grid.empty or grid.time.isna().any():
        raise ValueError("grid must contain nonmissing time labels")
    stored_alpha = grid.attrs.get("alpha")
    if alpha is not None and stored_alpha is not None and alpha != stored_alpha:
        raise ValueError("alpha conflicts with the level stored in the grid")
    alpha = stored_alpha if alpha is None else alpha
    if alpha is not None:
        if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real) or not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
    effects = None
    if estimate is None:
        times = pd.Index(grid.time.unique(), name="Time").sort_values()
    else:
        _validate(estimate)
        effects = estimate.effects.iloc[estimate.n_pre :]
        times = effects.index
    positions = times.get_indexer(grid.time)
    if (positions < 0).any() or len(np.unique(positions)) != len(times):
        raise ValueError("grid times must match the estimate's post-treatment dates")
    if grid.duplicated(["time", "candidate"]).any():
        raise ValueError("Each grid time and candidate pair must be unique")
    for column in ("candidate", "p_value"):
        if (
            not pd.api.types.is_numeric_dtype(grid[column])
            or pd.api.types.is_bool_dtype(grid[column])
            or np.iscomplexobj(grid[column])
        ):
            raise ValueError(f"grid {column} must contain finite real numbers")
    numeric = grid[["candidate", "p_value"]].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(numeric).all() or not grid.p_value.between(0, 1).all():
        raise ValueError("Grid candidates must be finite and p-values must lie in [0, 1]")
    if not pd.api.types.is_bool_dtype(grid.accepted) or grid.accepted.isna().any():
        raise ValueError("grid accepted must contain booleans")
    accepted = grid.accepted.to_numpy(dtype=bool)
    if alpha is not None and not np.array_equal(accepted, numeric[:, 1] > alpha):
        raise ValueError("grid accepted must match p_value > alpha")
    bounds = None
    if reference_bounds is not None:
        if (
            not isinstance(reference_bounds, pd.DataFrame)
            or not {"lower", "upper"}.issubset(reference_bounds)
            or not reference_bounds.index.is_unique
            or len(reference_bounds) != len(times)
            or not times.isin(reference_bounds.index).all()
        ):
            raise ValueError("reference_bounds requires lower and upper for every post-period")
        columns = reference_bounds.loc[times, ["lower", "upper"]]
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in columns.dtypes) or (
            np.iscomplexobj(columns)
        ):
            raise ValueError("Reference endpoints must be finite real numbers")
        bounds = columns.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(bounds).all() or (bounds[:, 0] > bounds[:, 1]).any():
            raise ValueError("Reference endpoints must be finite with lower <= upper")

    with publication_style():
        label = "Accepted grid" if alpha is None else f"{100 * (1 - alpha):g}% accepted grid"
        fig, body = _figure_grid(
            figsize or (7.2, 4.25), title or "Pointwise conformal effect sets", subtitle, note
        )
        ax = fig.add_subplot(body)
        _axes_style(ax)
        x, kind = _time(times)
        if len(times) > 1:
            _time_axis(ax, times, x, kind)
        if len(times) <= 8 and kind != "date":
            ax.set_xticks(x, [str(t) for t in times])
        elif len(times) == 1:
            ax.set_xticks(x, [str(times[0])])
        margin = (x[-1] - x[0]) * 0.07 if len(times) > 1 else 0.5
        ax.set_xlim(x[0] - margin, x[-1] + margin)
        ax.axhline(0, color=MUTED, linewidth=0.9, zorder=2)
        # Use discrete markers for the entire accepted grid, including singletons.
        ax.scatter(
            x[positions[accepted]],
            numeric[accepted, 0],
            marker="_",
            s=65,
            linewidths=1.3,
            color=BLUE,
            zorder=3,
            label=label,
        )
        for empty in np.setdiff1d(np.arange(len(times)), positions[accepted]):
            ax.text(
                x[empty],
                0.025,
                "Empty",
                ha="center",
                va="bottom",
                transform=ax.get_xaxis_transform(),
                fontsize=7,
                color=MUTED,
            )
        if effects is not None:
            ax.scatter(
                x,
                effects,
                s=30,
                marker="D",
                color=INK,
                edgecolors="white",
                linewidths=0.7,
                zorder=5,
                label="SCM point estimate",
            )
        if bounds is not None:
            # A small physical offset leaves coincident grid and reference marks legible.
            offset = ax.transData + ScaledTranslation(6 / 72, 0, fig.dpi_scale_trans)
            ax.scatter(
                np.repeat(x, 2),
                bounds.ravel(),
                marker="<",
                s=20,
                color=LITERATURE,
                linewidths=0,
                transform=offset,
                zorder=4,
                label=reference_label,
            )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(0, 1.005),
            ncols=len(handles),
            borderaxespad=0,
            handlelength=1.1,
            columnspacing=1.4,
            handletextpad=0.35,
            fontsize=8,
        )
        ax.set_xlabel(xlabel or times.name or "Time")
        ax.set_ylabel(ylabel or "Treatment effect")
        if not accepted.any() and effects is None and bounds is None:
            ax.update_datalim(np.column_stack([x[positions], numeric[:, 0]]))
            ax.autoscale_view(scalex=False)
        ax.margins(y=0.11)
        _finish_figure(fig)
        return fig, ax
