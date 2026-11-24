"""Publication figures for fitted estimates, with explicit units and common samples.

Matplotlib is imported only when a plotting function is called. All functions
return editable Matplotlib objects, leave figures open, and preserve global
plot settings. No function estimates uncertainty or refits an estimator.
"""

from contextlib import contextmanager
from pathlib import Path
from textwrap import fill
from typing import Mapping

import numpy as np
import pandas as pd

from .results import Estimate, compare

INK = "#243442"
BLUE = "#286A8C"
MUTED = "#667580"
GRID = "#E4E8EB"
POST = "#F1F4F6"


def _matplotlib():
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib: pip install 'pysynth-control[plot]'"
        ) from exc
    return mpl, plt


@contextmanager
def publication_style():
    """Temporarily use legible typography, restrained axes and vector fonts.

    DejaVu Sans ships with Matplotlib. PDF text uses embedded TrueType fonts;
    SVG text remains editable. Use :func:`save_figure` to apply the export
    settings at save time as well. No external fonts or TeX are required.
    """
    mpl, _ = _matplotlib()
    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "font.weight": "normal",
            "text.usetex": False,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#A8B2BA",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "savefig.bbox": None,
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "pysynth-control",
        }
    ):
        yield


def _validate(estimate):
    if not isinstance(estimate, Estimate):
        raise TypeError("Plotting expects fitted Estimate objects")
    index = estimate.observed.index
    if (
        len(index) < 2
        or not index.is_unique
        or index.hasnans
        or not index.is_monotonic_increasing
        or not index.equals(estimate.counterfactual.index)
        or not isinstance(estimate.n_pre, (int, np.integer))
        or not 1 <= estimate.n_pre < len(index)
    ):
        raise ValueError("Estimate must have aligned, ordered times and a valid treatment boundary")
    values = np.column_stack([estimate.observed, estimate.counterfactual, estimate.effects])
    if not np.isfinite(values).all():
        raise ValueError(
            "Estimate paths and effects must be finite; missing periods are not imputed"
        )


def _estimates(estimates):
    if not isinstance(estimates, Mapping) or not estimates:
        raise ValueError("estimates must be a nonempty mapping of labels to fitted estimates")
    for estimate in estimates.values():
        _validate(estimate)
    table = compare(estimates)
    if not np.isfinite(table[["att", "pre_rmse"]]).all().all():
        raise ValueError("ATT and pretreatment RMSE must be finite")
    return table


def _time(index):
    """Return numeric coordinates while retaining true numeric/calendar spacing."""
    if isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)):
        from matplotlib.dates import date2num

        dates = index.to_timestamp() if isinstance(index, pd.PeriodIndex) else index
        return date2num(dates.to_pydatetime()), "period" if isinstance(
            index, pd.PeriodIndex
        ) else "date"
    if isinstance(index, pd.TimedeltaIndex):
        return index.total_seconds().to_numpy() / 86400, "duration"
    if pd.api.types.is_numeric_dtype(index.dtype):
        return index.to_numpy(dtype=float), "numeric"
    return np.arange(len(index), dtype=float), "category"


def _time_axis(ax, index, x, kind):
    from matplotlib import dates, ticker

    if kind == "date":
        locator = dates.AutoDateLocator(minticks=3, maxticks=5, tz=index.tz)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(dates.ConciseDateFormatter(locator, tz=index.tz))
    elif kind in {"category", "period"}:
        selected = np.unique(np.linspace(0, len(index) - 1, min(5, len(index)), dtype=int))
        ax.xaxis.set_major_locator(ticker.FixedLocator(x[selected]))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter([str(index[i]) for i in selected]))
    else:
        ax.xaxis.set_major_locator(
            ticker.MaxNLocator(
                nbins=4,
                steps=[1, 2, 2.5, 5, 10],
                integer=np.allclose(x, np.round(x)),
                min_n_ticks=3,
            )
        )
        formatter = ticker.ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
    margin = (x[-1] - x[0]) * 0.018
    ax.set_xlim(x[0] - margin, x[-1] + margin)


def _number(value):
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude < 1e-5 or magnitude >= 1e7:
        return f"{value:.2g}".replace("-", "−")
    decimals = min(6, max(0, 2 - int(np.floor(np.log10(magnitude)))))
    return f"{value:,.{decimals}f}".replace("-", "−")


def _tick_number(value, _=None):
    text = _number(value)
    return text.rstrip("0").rstrip(".") if "." in text and "e" not in text else text


def _axes_style(ax, *, grid="y"):
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#A8B2BA")
    ax.spines[["left", "bottom"]].set_linewidth(0.6)
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color=GRID, linewidth=0.6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_formatter(FuncFormatter(_tick_number))
    ax.tick_params(length=3, width=0.6, color="#A8B2BA", labelcolor=MUTED)


def _intervention(ax, x, n_pre):
    ax.axvspan(x[n_pre], ax.get_xlim()[1], color=POST, linewidth=0, zorder=0)
    ax.axvline(
        x[n_pre], color=MUTED, linewidth=0.85, linestyle=(0, (4, 3)), label="Intervention", zorder=2
    )


def _limits(values):
    low = min(float(np.min(values)), 0.0)
    high = max(float(np.max(values)), 0.0)
    span = high - low or 1.0
    return low - span * 0.08, high + span * 0.12


def _figure_grid(figsize, title, subtitle, note):
    """Reserve real layout rows for headings and notes, avoiding clipped text."""
    _, plt = _matplotlib()
    width, height = figsize
    if width <= 0 or height <= 0:
        raise ValueError("figsize must contain positive dimensions in inches")
    wrapped_title = fill(str(title), width=max(25, int(width * 11))) if title else ""
    wrapped_subtitle = fill(str(subtitle), width=max(25, int(width * 16))) if subtitle else ""
    wrapped_note = fill(str(note), width=max(25, int(width * 19))) if note else ""
    head_height = (
        0.23 * len(wrapped_title.splitlines()) + 0.15 * len(wrapped_subtitle.splitlines()) + 0.12
    )
    foot_height = 0.14 * len(wrapped_note.splitlines()) + 0.06 if note else 0.02
    fig = plt.figure(figsize=figsize, layout="constrained")
    fig.get_layout_engine().set(h_pad=0.06, w_pad=0.06, hspace=0.045, wspace=0.05)
    outer = fig.add_gridspec(
        3, 1, height_ratios=[head_height, max(1, height - head_height - foot_height), foot_height]
    )
    heading = fig.add_subplot(outer[0])
    heading.set_axis_off()
    title_artist = subtitle_artist = note_artist = None
    if wrapped_title:
        title_artist = heading.text(
            0,
            1,
            wrapped_title,
            transform=heading.transAxes,
            va="top",
            fontsize=13,
            fontweight="semibold",
            linespacing=1.25,
        )
    if wrapped_subtitle:
        subtitle_artist = heading.text(
            0,
            0,
            wrapped_subtitle,
            transform=heading.transAxes,
            va="bottom",
            fontsize=8.5,
            color=MUTED,
            linespacing=1.3,
        )
    footer = fig.add_subplot(outer[2])
    footer.set_axis_off()
    if wrapped_note:
        note_artist = footer.text(
            0,
            1,
            wrapped_note,
            transform=footer.transAxes,
            va="top",
            fontsize=7.5,
            color=MUTED,
            linespacing=1.35,
        )
    # Caption widths must not force constrained_layout to squeeze the plot.
    # Wrap against measured axes widths after the plotting axes are present.
    for artist in (title_artist, subtitle_artist, note_artist):
        if artist is not None:
            artist.set_in_layout(False)
    fig._pysynth_caption_layout = (
        outer,
        title_artist,
        subtitle_artist,
        note_artist,
        title,
        subtitle,
        note,
    )
    return fig, outer[1]


def _wrap_measured(text, artist, renderer, width):
    lines = []
    for paragraph in str(text).splitlines():
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            measured = renderer.get_text_width_height_descent(
                candidate, artist.get_fontproperties(), ismath=False
            )[0]
            if line and measured > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return "\n".join(lines)


def _finish_figure(fig):
    """Fit captions using actual font metrics, including after a figure resize."""
    if getattr(fig, "_pysynth_caption_layout", None) is None:
        return
    original_canvas = fig.canvas
    try:
        if not hasattr(original_canvas, "get_renderer"):
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            FigureCanvasAgg(fig)
        _layout_captions(fig)
    finally:
        fig.set_canvas(original_canvas)


def _layout_captions(fig):
    layout = getattr(fig, "_pysynth_caption_layout", None)
    if layout is None:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outer, title, subtitle, note, title_text, subtitle_text, note_text = layout
    heights = []
    for artist, text in zip((title, subtitle, note), (title_text, subtitle_text, note_text)):
        if artist is None:
            heights.append(0)
            continue
        width = artist.axes.get_window_extent(renderer).width * 0.96
        wrapped = _wrap_measured(text, artist, renderer, width)
        artist.set_text(wrapped)
        heights.append(len(wrapped.splitlines()) * artist.get_fontsize() / 72 * 1.4)
    head = heights[0] + heights[1] + 0.12
    foot = heights[2] + 0.07 if note is not None else 0.02
    outer.set_height_ratios([head, max(1, fig.get_figheight() - head - foot), foot])
    for artist, original in getattr(fig, "_pysynth_panel_titles", []):
        width = artist.axes.get_window_extent(renderer).width * 0.98
        artist.set_text(_wrap_measured(original, artist, renderer, width))
    fig.canvas.draw()


def plot_paths(
    estimate: Estimate,
    *,
    ax=None,
    effects: bool = False,
    title: str | None = None,
    subtitle: str | None = None,
    note: str | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
    observed_label: str = "Observed",
    counterfactual_label: str = "Synthetic control",
    intervention_label: str | None = None,
    color: str = BLUE,
    figsize: tuple = (7.2, 3.8),
):
    """Plot an observed/counterfactual pair or a single effect path; return Axes.

    ``effects=True`` plots observed minus counterfactual and a zero reference.
    The dashed vertical line marks the first affected observation; the light
    background marks the post-period. It is not an uncertainty band. Numeric
    and calendar time retain their actual spacing. Supply ``ylabel`` for
    substantive units and ``ax`` to compose a larger figure. ``subtitle`` and
    ``note`` reserve figure-level caption space and require ``ax=None``.
    """
    _validate(estimate)
    if ax is not None and (subtitle is not None or note is not None):
        raise ValueError("subtitle and note require ax=None; add captions to your parent figure")
    with publication_style():
        own_figure = ax is None
        if ax is None:
            fig, body = _figure_grid(figsize, title or estimate.method, subtitle, note)
            ax = fig.add_subplot(body)
        x, kind = _time(estimate.observed.index)
        _axes_style(ax)
        _time_axis(ax, estimate.observed.index, x, kind)
        _intervention(ax, x, estimate.n_pre)
        if effects:
            ax.axhline(0, color=MUTED, linewidth=0.8, zorder=2)
            ax.plot(
                x, estimate.effects, color=color, linewidth=1.65, label=estimate.method, zorder=3
            )
            ax.set_ylim(*_limits(estimate.effects))
        else:
            ax.plot(x, estimate.observed, color=INK, linewidth=1.8, label=observed_label, zorder=3)
            ax.plot(
                x,
                estimate.counterfactual,
                color=color,
                linewidth=1.8,
                linestyle=(0, (5, 2.5)),
                label=counterfactual_label,
                zorder=3,
            )
            handles, labels = ax.get_legend_handles_labels()
            selected = [
                (handle, label) for handle, label in zip(handles, labels) if label != "Intervention"
            ]
            ax.legend(
                *zip(*selected),
                loc="lower left",
                bbox_to_anchor=(0, 1.01),
                ncols=2,
                borderaxespad=0,
                handlelength=2.8,
                columnspacing=2,
            )
        if intervention_label is not None:
            ax.text(
                x[estimate.n_pre],
                0.98,
                f" {intervention_label}",
                transform=ax.get_xaxis_transform(),
                va="top",
                fontsize=8,
                color=MUTED,
                bbox={"facecolor": POST, "edgecolor": "none", "pad": 2},
            )
        if not own_figure:
            ax.set_title(
                title or estimate.method,
                loc="left",
                fontweight="semibold",
                pad=31 if not effects else 12,
            )
        ax.set_ylabel(ylabel or ("Observed − counterfactual" if effects else "Outcome"))
        ax.set_xlabel(
            xlabel
            or estimate.observed.index.name
            or ("Time (days)" if kind == "duration" else "Time")
        )
        if own_figure:
            _finish_figure(ax.figure)
        return ax


def plot_effects(
    estimates: Mapping[str, Estimate],
    *,
    ncols: int = 3,
    labels: Mapping[str, str] | None = None,
    colors: Mapping[str, str] | None = None,
    title: str = "Treatment effects across estimators",
    subtitle: str | None = None,
    ylabel: str = "Observed − counterfactual",
    xlabel: str | None = None,
    sharey: bool = True,
    ylim: tuple | None = None,
    figsize: tuple | None = None,
    note: str
    | None = "Common outcome sample and treatment date. Shading marks the post-period; no uncertainty intervals are shown.",
):
    """Compare effect paths in labelled small multiples; return Figure, Axes array.

    Mapping insertion order is preserved. Both axes are shared by default;
    common y limits include every effect and zero, without clipping outliers.
    ``sharey=False`` explicitly allows different scales. ``labels`` and
    ``colors`` customize display without renaming estimates. Unused panels
    are hidden. The returned array always has shape (nrows, ncols).
    """
    _estimates(estimates)
    if isinstance(ncols, bool) or not isinstance(ncols, (int, np.integer)) or ncols < 1:
        raise ValueError("ncols must be a positive integer")
    if ylim is not None and (len(ylim) != 2 or not np.isfinite(ylim).all() or ylim[0] >= ylim[1]):
        raise ValueError("ylim must be two finite, increasing values")
    ncols = min(ncols, len(estimates))
    nrows = (len(estimates) + ncols - 1) // ncols
    reference = next(iter(estimates.values()))
    with publication_style():
        fig, body = _figure_grid(figsize or (7.2, 1.1 + 1.85 * nrows), title, subtitle, note)
        grid = body.subgridspec(nrows, ncols, hspace=0.22, wspace=0.12)
        axes = np.empty((nrows, ncols), dtype=object)
        fig._pysynth_panel_titles = []
        x, kind = _time(reference.observed.index)
        common = ylim or _limits(np.concatenate([r.effects.to_numpy() for r in estimates.values()]))
        for position, (name, result) in enumerate(estimates.items()):
            row, col = divmod(position, ncols)
            first = axes[0, 0] if position else None
            ax = fig.add_subplot(grid[row, col], sharex=first, sharey=first if sharey else None)
            axes[row, col] = ax
            _axes_style(ax)
            _time_axis(ax, reference.observed.index, x, kind)
            _intervention(ax, x, reference.n_pre)
            ax.axhline(0, color=MUTED, linewidth=0.75, zorder=2)
            ax.plot(
                x,
                result.effects,
                color=(colors or {}).get(name, BLUE),
                linewidth=1.4,
                label=name,
                zorder=3,
            )
            ax.set_ylim(*(common if sharey or ylim is not None else _limits(result.effects)))
            display_name = (labels or {}).get(name, name)
            letter = chr(65 + position) if position < 26 else str(position + 1)
            title_artist = ax.set_title(
                f"{letter}  {fill(str(display_name), width=max(16, int(fig.get_figwidth() / ncols * 11)))}",
                loc="left",
                fontsize=9,
                fontweight="semibold",
                pad=9,
            )
            fig._pysynth_panel_titles.append((title_artist, f"{letter}  {display_name}"))
            bottom = position + ncols >= len(estimates)
            ax.tick_params(labelbottom=bottom, labelleft=col == 0 or not sharey)
            if bottom:
                ax.set_xlabel(
                    xlabel
                    or reference.observed.index.name
                    or ("Time (days)" if kind == "duration" else "Time"),
                    fontsize=8,
                )
        for position in range(len(estimates), nrows * ncols):
            row, col = divmod(position, ncols)
            axes[row, col] = fig.add_subplot(grid[row, col])
            axes[row, col].set_visible(False)
        fig.supylabel(ylabel, fontsize=9)
        _finish_figure(fig)
        return fig, axes


def plot_comparison(
    estimates: Mapping[str, Estimate],
    *,
    labels: Mapping[str, str] | None = None,
    colors: Mapping[str, str] | None = None,
    title: str = "Average effects and pretreatment fit",
    subtitle: str | None = None,
    units: str = "outcome units",
    figsize: tuple | None = None,
    note: str
    | None = "Dots are point estimates, without confidence intervals. Pretreatment RMSE describes fit, not causal credibility.",
):
    """Plot aligned ATT and pre-RMSE dot plots; return Figure, two Axes.

    Methods retain mapping order, with a shared label column and separate
    x scales. ATT includes zero; RMSE starts at zero. Every method must share
    the same observed path, ordered time index and treatment boundary. No
    error bars, significance tests, rankings or uncertainty are inferred.
    """
    table = _estimates(estimates)
    with publication_style():
        from matplotlib.ticker import FuncFormatter, MaxNLocator

        fig, body = _figure_grid(figsize or (7.2, 1.55 + 0.31 * len(table)), title, subtitle, note)
        grid = body.subgridspec(1, 2, width_ratios=[1.2, 1], wspace=0.15)
        axes = np.array([fig.add_subplot(grid[0]), fig.add_subplot(grid[1])])
        y = np.arange(len(table))
        display = [fill(str((labels or {}).get(name, name)), width=26) for name in table.index]
        palette = [(colors or {}).get(name, BLUE) for name in table.index]
        for ax, metric, heading in zip(
            axes, ["att", "pre_rmse"], ["A  Average post-treatment effect", "B  Pretreatment fit"]
        ):
            values = table[metric].to_numpy()
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.set_yticks(y, display if metric == "att" else [""] * len(y))
            ax.tick_params(axis="y", length=0, pad=10, labelsize=8.5, labelcolor=INK)
            ax.tick_params(axis="x", length=3, width=0.6)
            ax.set_ylim(len(table) - 0.5, -0.5)
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            ax.axvline(0, color=MUTED, linewidth=0.8, zorder=2)
            ax.scatter(
                values,
                y,
                s=27,
                c=palette,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
                label=metric,
                clip_on=False,
            )
            for value, row in zip(values, y):
                ax.annotate(
                    _number(value),
                    (value, row),
                    xytext=(7, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                    color=INK,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
                )
            low, high = min(0, values.min()), max(0, values.max())
            span = high - low or 1.0
            ax.set_xlim(0 if metric == "pre_rmse" else low - span * 0.12, high + span * 0.37)
            ax.xaxis.set_major_locator(
                MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3)
            )
            ax.xaxis.set_major_formatter(FuncFormatter(_tick_number))
            ax.set_title(heading, loc="left", fontsize=9, fontweight="semibold", pad=12)
            ax.set_xlabel(f"{'ATT' if metric == 'att' else 'RMSE'} ({units})", fontsize=8.5)
        _finish_figure(fig)
        return fig, axes


def save_figure(
    figure,
    path: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf", "svg", "png"),
    dpi: int = 300,
) -> tuple[Path, ...]:
    """Save an explicit Figure as vector PDF/SVG and/or high-resolution PNG.

    ``path`` is a filename stem. A .pdf/.svg/.png suffix instead selects that
    single format. Parent directories are created. Export preserves the
    figure's physical dimensions, embeds PDF fonts and retains SVG text.
    PNG DPI affects raster resolution only. Return the written paths; the
    Figure stays open. Existing destination files are replaced.
    """
    path = Path(path)
    if path.suffix:
        formats = (path.suffix.lstrip(".").lower(),)
        path = path.with_suffix("")
    if not formats or any(fmt not in {"pdf", "svg", "png"} for fmt in formats):
        raise ValueError("formats must contain pdf, svg or png")
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    destinations = []
    with publication_style():
        _finish_figure(figure)
        for fmt in dict.fromkeys(formats):
            destination = path.parent / f"{path.name}.{fmt}"
            # Keep author-selected inch dimensions rather than crop to a
            # content-dependent tight bounding box. The figures use layout.
            if fmt == "pdf":
                metadata = {"Creator": "PySynth", "CreationDate": None, "ModDate": None}
            elif fmt == "svg":
                metadata = {"Creator": "PySynth", "Date": None}
            else:
                metadata = {"Software": "PySynth"}
            figure.savefig(
                destination,
                format=fmt,
                dpi=dpi,
                facecolor="white",
                bbox_inches=None,
                metadata=metadata,
            )
            destinations.append(destination)
    return tuple(destinations)
