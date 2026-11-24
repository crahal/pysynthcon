"""Publication figures for pretreatment validation and donor sensitivity."""

from textwrap import fill

import numpy as np

from .plotting import (
    BLUE,
    GRID,
    INK,
    LITERATURE,
    MUTED,
    _figure_grid,
    _finish_figure,
    _number,
    _tick_number,
    publication_style,
)


def _rows(ax, names, labels, *, show_labels=True):
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_yticks(
        np.arange(len(names)),
        [fill(str((labels or {}).get(name, name)), 25) for name in names]
        if show_labels
        else [""] * len(names),
    )
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0, pad=9, labelsize=8)
    ax.tick_params(axis="x", length=3, width=0.6, labelcolor=MUTED)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.xaxis.set_major_formatter(FuncFormatter(_tick_number))


def plot_backtest(
    result,
    *,
    title="Pretreatment validation",
    units="outcome units",
    labels=None,
    figsize=None,
    note=None,
):
    """Plot pooled and fold-specific held-out RMSE; return Figure, two Axes.

    Rows follow increasing pooled validation RMSE. Every heatmap cell uses
    the same color scale. No confidence intervals or causal ranking are
    implied. The lowest pooled RMSE is labelled beside its point, including
    all exact ties. Validation donor outcomes are available; held-out treated
    outcomes are masked during fitting. Requires a :class:`BacktestResult`.
    Use :func:`pysynth.save_figure` for vector and high-resolution export.
    """
    from .validation import BacktestResult

    if not isinstance(result, BacktestResult):
        raise TypeError("result must be a BacktestResult")
    table = result.summary.sort_values("rank", kind="stable")
    if table.empty:
        raise ValueError("backtest results must not be empty")
    names = table.index.tolist()
    matrix = result.scores.pivot(index="model", columns="fold", values="validation_rmse")
    matrix = matrix.reindex(names)
    values = matrix.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("fold RMSE must be finite, nonnegative and present for every model")
    caption = "Pretreatment predictions only; lower RMSE does not establish causal validity."
    if result.predictions.duplicated(["model", "time"]).any():
        caption += " Overlapping folds count repeated prediction instances."
    counts = table.n_predictions.unique()
    sample_size = (
        f"{counts[0]} held-out predictions per candidate"
        if len(counts) == 1
        else f"{len(result.predictions)} held-out predictions in total"
    )
    with publication_style():
        fig, body = _figure_grid(
            figsize or (7.2, 1.9 + 0.48 * len(names)),
            title,
            None,
            caption if note is None else note,
        )
        grid = body.subgridspec(1, 2, width_ratios=[1.1, 1], wspace=0.14)
        axes = np.array([fig.add_subplot(grid[0]), fig.add_subplot(grid[1])])
        left, right = axes
        _rows(left, names, labels)
        pooled = table.validation_rmse.to_numpy()
        left.scatter(pooled, np.arange(len(names)), s=29, color=BLUE, zorder=3)
        minima = pooled == pooled.min()
        for row, value in enumerate(pooled):
            left.annotate(
                _number(value),
                (value, row),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
            if minima[row]:
                # Label all exact ties: choosing the first row would imply a
                # distinction the validation results do not support.
                right_aligned = value > 0.6 * (pooled.max() or 1)
                left.annotate(
                    "Lowest validation RMSE" + (" (tie)" if minima.sum() > 1 else ""),
                    (value, row),
                    xytext=(-7 if right_aligned else 7, -10),
                    textcoords="offset points",
                    va="top",
                    ha="right" if right_aligned else "left",
                    fontsize=7,
                    color=BLUE,
                )
        left.set_xlim(0, (pooled.max() or 1) * 1.45)
        left.set_xlabel(f"Pooled held-out RMSE\n({units})\n{sample_size}", fontsize=7.5)
        left.set_title("A  Overall validation", loc="left", fontsize=9, fontweight="semibold")

        maximum = float(values.max()) or 1.0
        right.imshow(values, cmap="Blues", vmin=0, vmax=maximum, aspect="auto")
        right.set_yticks([])
        right.set_xticks(np.arange(len(matrix.columns)), [str(x) for x in matrix.columns])
        right.set_xlabel("Validation fold (earlier → later)", fontsize=8)
        right.tick_params(length=0, labelcolor=MUTED)
        right.spines[:].set_visible(False)
        right.set_title("B  Held-out RMSE by fold", loc="left", fontsize=9, fontweight="semibold")
        for row, column in np.ndindex(values.shape):
            value = values[row, column]
            right.text(
                column,
                row,
                _number(value),
                va="center",
                ha="center",
                fontsize=max(5.5, min(8, 34 / len(matrix.columns))),
                color="white" if value / maximum > 0.6 else INK,
            )
        _finish_figure(fig)
        return fig, axes


def plot_donor_sensitivity(
    result,
    *,
    title="Sensitivity to donor exclusions",
    units="outcome units",
    labels=None,
    unit_labels=None,
    figsize=None,
    note=None,
):
    """Show every exclusion ATT, its baseline and maximum absolute change.

    Horizontal spans describe the fitted scenario range, not uncertainty.
    Each scenario is a complete refit, including repeated donor selection.
    Baseline values and the exclusion responsible for each largest change
    are annotated beside their points. Optional ``unit_labels`` maps original
    donor IDs to display names for single-donor omissions. Rows preserve
    estimator order.
    Requires a :class:`DonorSensitivityResult`.
    Returns Figure and two editable Matplotlib Axes; no models are refitted.
    """
    from .sensitivity import DonorSensitivityResult

    if not isinstance(result, DonorSensitivityResult):
        raise TypeError("result must be a DonorSensitivityResult")
    summary = result.summary
    if summary.empty:
        raise ValueError("sensitivity results must not be empty")
    names = summary.index.tolist()
    caption = "Ochre dots: complete exclusion refits. Spans show scenario ranges, not confidence intervals."
    with publication_style():
        fig, body = _figure_grid(
            figsize or (7.2, 1.9 + 0.62 * len(names)),
            title,
            None,
            caption if note is None else note,
        )
        grid = body.subgridspec(1, 2, width_ratios=[1.3, 1], wspace=0.15)
        axes = np.array([fig.add_subplot(grid[0]), fig.add_subplot(grid[1])])
        left, right = axes
        _rows(left, names, labels)
        _rows(right, names, labels, show_labels=False)
        left.axvline(0, color=MUTED, linewidth=0.8)
        all_values = [0.0]
        for row, name in enumerate(names):
            baseline = float(summary.loc[name, "baseline_att"])
            model_rows = result.table.loc[result.table.model == name]
            values = model_rows["att"].to_numpy()
            if not len(values) or not np.isfinite(np.r_[baseline, values]).all():
                raise ValueError("scenario and baseline estimates must be finite")
            all_values.extend([baseline, *values])
            left.hlines(row, values.min(), values.max(), color=LITERATURE, linewidth=1, alpha=0.7)
            # A deterministic vertical spread keeps coincident scenarios visible.
            offset = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else np.zeros(1)
            left.scatter(values, row + offset, s=14, color=LITERATURE, alpha=0.55, zorder=3)
            left.scatter([baseline], [row], s=34, marker="D", color=BLUE, zorder=4)
            left.annotate(
                f"Baseline {_number(baseline)}",
                (baseline, row),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color=BLUE,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2},
            )
            change = float(summary.loc[name, "max_abs_change"])
            right.scatter([change], [row], color=BLUE, s=28, zorder=3)
            right.annotate(
                _number(change),
                (change, row),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
            scenario = str(summary.loc[name, "worst_scenario"])
            scenario_label = f"Omit {scenario[5:]}" if scenario.startswith("omit:") else scenario
            if unit_labels is not None and scenario.startswith("omit:"):
                excluded = model_rows.loc[model_rows.scenario == scenario, "excluded_units"].iloc[0]
                if len(excluded) == 1 and excluded[0] in unit_labels:
                    scenario_label = f"Omit {unit_labels[excluded[0]]}"
            right.annotate(
                fill(scenario_label, 23),
                (change, row),
                xytext=(0, -11),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=7,
                color=MUTED,
            )
        low, high = min(all_values), max(all_values)
        span = high - low or 1.0
        left.set_xlim(low - 0.1 * span, high + 0.1 * span)
        right.set_xlim(0, (summary.max_abs_change.max() or 1) * 1.4)
        right.set_xlabel(f"Maximum |ATT change|\n({units})", fontsize=8)
        left.set_title(
            "A  Baseline and exclusion fits", loc="left", fontsize=9, fontweight="semibold"
        )
        right.set_title(
            "B  Largest change · exclusion", loc="left", fontsize=8.5, fontweight="semibold"
        )
        counts = summary.n_scenarios.unique()
        count_note = (
            f"{counts[0]} exclusion refits per estimator"
            if len(counts) == 1
            else f"{len(result.table)} exclusion refits in total"
        )
        left.set_xlabel(f"ATT ({units})\n{count_note}", fontsize=7.5)
        _finish_figure(fig)
        return fig, axes
