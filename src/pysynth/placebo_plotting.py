"""Publication figures for in-space placebo gap paths."""

from .inference import PlaceboResult
from .plotting import (
    BLUE,
    MUTED,
    _axes_style,
    _figure_grid,
    _finish_figure,
    _intervention,
    _time,
    _time_axis,
    publication_style,
)


def plot_placebo_gaps(
    result: PlaceboResult,
    *,
    title: str = "In-space placebo gaps",
    ylabel: str = "Observed − counterfactual",
    xlabel: str | None = None,
    treated_label: str | None = None,
    note: str | None = None,
    figsize: tuple[float, float] = (7.2, 4.4),
):
    """Plot the treated gap and eligible placebo paths. Return a figure and one axis.

    The rank and denominator refer to the displayed units. Select another
    pre-fit screen with `result.with_pre_mspe_cutoff(multiplier)` before
    plotting. Shading starts at the first treated observation. The calendar,
    gaps and assignment fits come directly from the supplied result.
    """
    from matplotlib.lines import Line2D

    if not isinstance(result, PlaceboResult):
        raise TypeError("result must be a PlaceboResult")
    selected = result.table.loc[result.table.eligible]
    actual = result.table.index[result.table.treated][0]
    treated_name = str(actual) if treated_label is None else treated_label
    count = len(selected) - 1
    count_label = f"{count} of {len(result.placebos)} placebo paths"
    if result.max_pre_mspe_ratio is not None:
        count_label += f" · pre-MSPE ≤ {result.max_pre_mspe_ratio:g}× treated"
    if result.max_pre_rmse_ratio is not None:
        count_label += f" · pre-RMSE ≤ {result.max_pre_rmse_ratio:g}× treated"
    rank_label = f"{treated_name} · post/pre MSPE rank {result.rank}/{len(selected)}"
    times = result.treated.effects.index
    x, kind = _time(times)
    with publication_style():
        fig, body = _figure_grid(figsize, title, None, note)
        ax = fig.add_subplot(body)
        _axes_style(ax)
        _time_axis(ax, times, x, kind)
        _intervention(ax, x, result.treated.n_pre)
        ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
        for unit in selected.index[~selected.treated]:
            ax.plot(
                x,
                result.placebos[unit].effects,
                color=MUTED,
                linewidth=0.7,
                alpha=0.48,
                zorder=2,
                label=str(unit),
            )
        (line,) = ax.plot(x, result.treated.effects, color=BLUE, linewidth=2, zorder=3)
        ax.legend(
            [line, Line2D([], [], color=MUTED, linewidth=1)],
            [rank_label, count_label],
            loc="best",
            fontsize=8,
            frameon=True,
            framealpha=0.94,
            facecolor="white",
            edgecolor="none",
        )
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel or times.name or ("Time (days)" if kind == "duration" else "Time"))
        _finish_figure(fig)
        return fig, ax
