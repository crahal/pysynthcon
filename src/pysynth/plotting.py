"""Publication figures for fitted estimates, with explicit units and common samples.

Matplotlib is imported only when a plotting function is called. All functions
return editable Matplotlib objects, leave figures open, and preserve global
plot settings. No function estimates uncertainty or refits an estimator.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from textwrap import fill
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .results import Estimate, _mean_effect, compare

INK = "#243442"
BLUE = "#286A8C"
MUTED = "#667580"
GRID = "#E4E8EB"
POST = "#F1F4F6"
LITERATURE = "#87582E"

_FONT_DIR = Path(__file__).with_name("fonts")
_FONT_FACES = (
    ("Source Sans 3", "SourceSans3-Regular.ttf", 400),
    ("Source Sans 3", "SourceSans3-Semibold.ttf", 600),
    ("STIX Two Text", "STIXTwoText-SemiBold.ttf", 600),
)


@dataclass(frozen=True)
class LiteratureBenchmark:
    """A sourced literature result, separate from the estimates being plotted.

    ``headline`` states the result and units; ``citation`` includes its source
    locator. ``context`` records the method, window and specification caveats.
    ``source_url`` links the citation in vector exports. Set ``value`` only for
    a mean effect in the plot's outcome units, with the *same* averaging window.
    In that case ``window`` must give the first and last post-treatment index
    labels. Omit both for a percentage headline on a level-scale figure.
    Numeric guides are never drawn on outcome-level or RMSE axes.
    """

    headline: str
    citation: str
    source_url: str
    context: str
    value: float | None = None
    window: tuple | None = None
    label: str = "Published result"

    def __post_init__(self):
        for name in ("headline", "citation", "source_url", "context", "label"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not self.source_url.startswith(("https://", "http://")):
            raise ValueError("source_url must be an HTTP(S) source link")
        if self.value is not None:
            if not np.isscalar(self.value) or not np.isfinite(self.value):
                raise ValueError("benchmark value must be finite")
            if self.window is None or len(self.window) != 2:
                raise ValueError("a numeric benchmark requires a two-endpoint window")
        elif self.window is not None:
            raise ValueError("window is only used with a numeric benchmark")


@dataclass(frozen=True)
class ReferenceComparison:
    """Compare one fitted specification with a sourced result over a named window.

    The reproduced number is calculated from ``estimate``, never supplied by
    hand. ``mean_percent_effect`` averages 100 * (observed - counterfactual) /
    observed; its difference is in percentage points. A rounded or approximate
    source number is descriptive, not an optimization target or a tolerance.
    Set ``comparable=False`` for a qualitative headline that accompanies a
    related calculation. Its difference is undefined. Only the identical
    Estimate object receives a numeric guide for a comparable statistic.
    """

    label: str
    estimate: Estimate = field(repr=False, compare=False)
    reference: float
    window: tuple
    units: str
    citation: str
    source_url: str
    context: str
    statistic: str = "mean_effect"
    precision: int = 3
    reference_precision: int = 1
    approximate: bool = False
    source_label: str = "Published"
    window_label: str | None = None
    comparable: bool = True

    def __post_init__(self):
        for name in ("label", "units", "citation", "source_url", "context", "source_label"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not self.source_url.startswith(("https://", "http://")):
            raise ValueError("source_url must be an HTTP(S) source link")
        if not np.isscalar(self.reference) or not np.isfinite(self.reference):
            raise ValueError("reference must be finite")
        if self.statistic not in {"mean_effect", "mean_percent_effect"}:
            raise ValueError("statistic must be mean_effect or mean_percent_effect")
        if self.window_label is not None and (
            not isinstance(self.window_label, str) or not self.window_label.strip()
        ):
            raise ValueError("window_label must be a nonempty string")
        for value in (self.precision, self.reference_precision):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 8:
                raise ValueError("precision must be an integer between 0 and 8")
        if not isinstance(self.comparable, bool):
            raise ValueError("comparable must be a boolean")
        self._positions()

    def _positions(self):
        _validate(self.estimate)
        if not isinstance(self.window, tuple) or len(self.window) != 2:
            raise ValueError("reference window must contain two index labels")
        positions = self.estimate.observed.index.get_indexer(self.window)
        if positions[0] < self.estimate.n_pre or positions[1] < positions[0]:
            raise ValueError("reference window must be an ordered, observed post-treatment window")
        return slice(positions[0], positions[1] + 1)

    @property
    def reproduced(self) -> float:
        """Mean effect on the source window and, if requested, percentage scale."""
        positions = self._positions()
        values = self.estimate.effects.iloc[positions]
        if self.statistic == "mean_percent_effect":
            observed = self.estimate.observed.iloc[positions]
            if (observed == 0).any():
                raise ValueError("percentage comparison requires nonzero observed outcomes")
            # Preserve the calculation for ordinary data. At very large
            # scales the numerator can overflow before division cancels it.
            with np.errstate(over="ignore", invalid="ignore"):
                percentage = 100 * values / observed
                overflow = ~np.isfinite(percentage)
                if overflow.any():
                    percentage.loc[overflow] = (values[overflow] / observed[overflow]) * 100
            if not np.isfinite(percentage).all():
                raise ValueError("percentage comparison exceeds finite floating-point range")
            values = percentage
        return _mean_effect(values)

    @property
    def difference(self) -> float | None:
        """Return reproduced minus source when both values represent the same statistic."""
        return self.reproduced - self.reference if self.comparable else None


def reference_comparison_table(references: Sequence[ReferenceComparison]) -> pd.DataFrame:
    """Return full numeric values and provenance for figure annotations."""
    references = _references(references)
    return pd.DataFrame(
        [
            {
                "label": ref.label,
                "source_label": ref.source_label,
                "reference": ref.reference,
                "reproduced": ref.reproduced,
                "difference": ref.difference,
                "difference_label": _reference_difference(ref) if ref.comparable else None,
                "units": ref.units,
                "difference_units": (
                    ("percentage points" if ref.statistic == "mean_percent_effect" else ref.units)
                    if ref.comparable
                    else None
                ),
                "window_start": ref.window[0],
                "window_end": ref.window[1],
                "statistic": ref.statistic,
                "approximate": ref.approximate,
                "citation": ref.citation,
                "source_url": ref.source_url,
                "context": ref.context,
            }
            for ref in references
        ]
    )


def _references(references, estimate=None):
    if references is None:
        return ()
    references = tuple(references)
    for ref in references:
        if not isinstance(ref, ReferenceComparison):
            raise TypeError("references must contain ReferenceComparison objects")
        if not np.isfinite(ref.reproduced):
            raise ValueError("reproduced reference must be finite")
        if estimate is not None:
            # Sources may use a different subset of this common post-period,
            # but a reference fit must describe the same observed case.
            _estimates({"plotted": estimate, "reference": ref.estimate})
    return references


def _plot_references(references, estimate, benchmark):
    references = _references(references, estimate)
    if references and benchmark is not None:
        raise ValueError("use references or benchmark, not both in the same figure")
    return references


def _reference_number(value, precision, *, signed=False):
    if abs(value) < 0.5 * 10 ** (-precision):
        value = 0.0
    return (
        f"{value:+,.{precision}f}".replace("-", "−")
        if signed
        else f"{value:,.{precision}f}".replace("-", "−")
    )


def _reference_difference(ref, *, symbol=True):
    """Format a comparable discrepancy without rounding a nonzero value to zero."""
    value = ref.difference
    unit = " pp" if ref.statistic == "mean_percent_effect" else ""
    resolution = 10.0 ** (-ref.precision)
    if 0 < abs(value) < resolution:
        return f"|Δ| < {_reference_number(resolution, ref.precision)}{unit}"
    label = "Δ " if symbol else ""
    if ref.approximate:
        label += "≈ "
    return label + _reference_number(value, ref.precision, signed=value != 0) + unit


def _matched_references(references, estimate, *, full_window=False):
    return [
        ref
        for ref in references
        if ref.estimate is estimate
        and ref.comparable
        and ref.statistic == "mean_effect"
        and (
            not full_window
            or ref.window == (estimate.observed.index[estimate.n_pre], estimate.observed.index[-1])
        )
    ]


def _reference_effect(ax, x, references, estimate):
    for ref in _matched_references(references, estimate):
        positions = ref._positions()
        ax.plot(
            [x[positions.start], x[positions.stop - 1]],
            [ref.reference, ref.reference],
            color=LITERATURE,
            linewidth=1,
            linestyle=(0, (2, 2)),
            label=f"{ref.source_label}: {ref.label}",
            zorder=2.5,
        )


def _annotation_style(style):
    if style not in {"direct", "strip"}:
        raise ValueError("reference_style must be 'direct' or 'strip'")


def _reference_fits(estimates, references):
    """Give an unplotted historical fit its own explicitly named row/panel."""
    result = dict(estimates)
    for ref in references:
        if any(ref.estimate is fit for fit in result.values()):
            continue
        name = f"Reference · {ref.label}"
        while name in result:
            name += " · historical"
        result[name] = ref.estimate
    return result


def _references_by_fit(references):
    groups = {}
    for ref in references:
        groups.setdefault(id(ref.estimate), []).append(ref)
    return groups


def _source_numbers(references):
    sources = {}
    for ref in references:
        key = (ref.citation, ref.source_url)
        if key not in sources:
            sources[key] = len(sources) + 1
    return sources


def _direct_note(note, references, benchmark, *, include_citations=True):
    """Short citations; links live on callouts and full details stay in the table."""
    sources = _source_numbers(references)
    lines = [note] if note else []
    if any(ref.comparable for ref in references):
        lines.append("Δ = fit − source; source rounding retained.")
    if include_citations:
        lines.extend(f"[{number}] {citation}" for (citation, _), number in sources.items())
    if benchmark is not None:
        lines.append(f"[1] {benchmark.citation} · {benchmark.context}")
    return "\n".join(lines) or None


def _reference_text(ref, number, *, compact=False, facet=False):
    source = {
        "Published": "Paper",
        "Original R software": "R Synth",
        "Upstream software": "Upstream",
        "Authors’ vignette": "Vignette",
    }.get(ref.source_label, ref.source_label)
    prefix = "≈ " if ref.approximate else ""
    unit = "%" if ref.statistic == "mean_percent_effect" else ""
    reported = prefix + _reference_number(ref.reference, ref.reference_precision) + unit
    reproduced = _reference_number(ref.reproduced, ref.precision) + unit
    if facet:
        source = "Paper" if source == "Fig. 1 headline" else source
        return f"[{number}] {source} {reported} · fit {reproduced}"
    window = ref.window_label or f"{ref.window[0]}–{ref.window[1]}"
    key = f" [{number}]" if number is not None else ""
    if not ref.comparable:
        return f"{source} {reported}{key}\n{window} mean {reproduced}\n{ref.units}"
    difference = _reference_difference(ref)
    if compact:
        return f"{source} {reported} · {difference}{key}"
    scale = f"\n{ref.units}" if unit else ""
    return f"{source} {reported} · fit {reproduced}\n{window} · {difference}{key}{scale}"


def _comparison_source_notes(references, grouped, *, in_panel=()):
    """Key each source's checked window and scale to its fitted specification."""
    notes = []
    for (citation, url), number in _source_numbers(references).items():
        matches = [r for r in references if (r.citation, r.source_url) == (citation, url)]
        shared_source = len({id(r.estimate) for r in matches}) > 1
        parts = []
        for ref in matches:
            if id(ref.estimate) in grouped:
                if id(ref.estimate) in in_panel:
                    window = ref.window_label or f"{ref.window[0]}–{ref.window[1]}"
                    detail = f"{window} · {ref.units}"
                    if ref.comparable:
                        detail += f" · {_reference_difference(ref)}"
                    else:
                        detail = f"Fit window {detail}"
                else:
                    detail = _reference_text(ref, None).replace("%\n% ", "% ").replace("\n", " · ")
                parts.append(f"{ref.label} · {detail}" if shared_source else detail)
        notes.append((f"[{number}] " + "  ·  ".join([*parts, citation]), url))
    return notes


def _reference_lines(ref):
    compact = bool(_matched_references([ref], ref.estimate, full_window=True))
    return len(_reference_text(ref, 1, compact=compact).splitlines())


def _reference_spacing(ref):
    return 0.6 + 0.4 * _reference_lines(ref)


def _callout(ax, text, xy, *, width=36, fontsize=7.8, color=LITERATURE, url=None):
    """Place a callout in open space. Only point to data for a numeric target."""
    original = text
    text = "\n".join(fill(line, width=width) for line in text.splitlines())
    artist = ax.annotate(
        text,
        xy=(0.5, 0.5) if xy is None else xy,
        xycoords="axes fraction" if xy is None else ax.transData,
        xytext=(0.03, 0.96),
        textcoords="axes fraction",
        ha="left",
        va="top",
        fontsize=fontsize,
        color=color,
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.94,
        },
        arrowprops={
            "arrowstyle": "-",
            "color": color,
            "linewidth": 0.6,
            "relpos": (0.5, 0.5),
        }
        if xy is not None
        else None,
        annotation_clip=False,
        zorder=6,
    )
    artist.set_in_layout(False)
    artist._pysynth_callout_text = original
    if url:
        artist.set_url(url)
    ax.figure._pysynth_direct_annotations = getattr(
        ax.figure, "_pysynth_direct_annotations", []
    ) + [artist]
    return artist


def _annotate_references(ax, x, references, estimate, *, effects=False, small=False, color=BLUE):
    matched = [ref for ref in references if ref.estimate is estimate]
    numbers = _source_numbers(references)
    if not matched:
        if effects:
            ax.plot(
                [x[estimate.n_pre], x[-1]],
                [estimate.att, estimate.att],
                color=color,
                linewidth=0.6,
                linestyle=(0, (2, 3)),
                alpha=0.65,
                label="Fitted mean ATT",
                zorder=2.5,
            )
            label = ax.annotate(
                f"Mean ATT {_number(estimate.att)}",
                xy=(0, 0),
                xycoords="axes fraction",
                xytext=(4, 3),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=7.5,
                color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.8},
                zorder=5,
            )
            label.set_in_layout(False)
        return
    if small and len(matched) > 1:
        labels = []
        keys = []
        for ref in matched:
            number = numbers[(ref.citation, ref.source_url)]
            keys.append(str(number))
            label = ax.text(
                0,
                0,
                _reference_text(ref, number, facet=True),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.4,
                color=LITERATURE,
                url=ref.source_url,
                zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.8},
            )
            label.set_in_layout(False)
            labels.append(label)
        ax.figure._pysynth_facet_labels[ax] = (
            id(estimate),
            labels,
            ", ".join(dict.fromkeys(keys)),
        )
        return
    # Each callout uses one actual source window; unlike a full-period ATT,
    # percentage and subset-window results never acquire a level-scale guide.
    for ref in matched:
        pos = ref._positions()
        middle = (pos.start + pos.stop - 1) // 2
        if effects:
            # A percentage or contextual headline has no ordinate on a level-effect axis.
            xy = None
            if ref.comparable and ref.statistic == "mean_effect":
                xy = (x[middle], ref.reference)
        else:
            xy = (
                x[middle],
                (estimate.observed.iloc[middle] + estimate.counterfactual.iloc[middle]) / 2,
            )
        _callout(
            ax,
            _reference_text(ref, numbers[(ref.citation, ref.source_url)]),
            xy,
            width=29 if small else 47,
            fontsize=7.4 if small else 8,
            url=ref.source_url,
        )


def _place_facet_sources(ax, labels, renderer):
    """Place the linked labels as a group, preferring clear space at lower left."""
    from matplotlib.transforms import Bbox

    bounds = ax.get_window_extent(renderer)
    pad = renderer.points_to_pixels(4)
    gap = renderer.points_to_pixels(2)
    line_boxes = [label.get_window_extent(renderer) for label in labels]
    width = max(box.width for box in line_boxes)
    height = sum(box.height for box in line_boxes) + (len(labels) - 1) * gap
    if width + 2 * pad > bounds.width or height + 2 * pad > bounds.height:
        return False
    paths = [
        line.get_transform().transform_path(line.get_path())
        for line in ax.lines
        if line.get_visible() and line.get_label() != "Intervention"
    ]
    left, right = bounds.x0 + pad, bounds.x1 - pad - width
    bottom, top = bounds.y0 + pad, bounds.y1 - pad - height
    for y in (bottom, top, (bottom + top) / 2):
        for x in (left, right):
            block = Bbox.from_bounds(x, y, width, height).padded(gap)
            if any(path.intersects_bbox(block, filled=False) for path in paths):
                continue
            cursor = y + height
            for label, box in zip(labels, line_boxes):
                cursor -= box.height
                label.set_position(ax.transAxes.inverted().transform((x, cursor)))
                cursor -= gap
            return True
    return False


def _layout_effect_sources(fig):
    """Keep the grid fixed; carry extra detail into its citation only if needed."""
    references, grouped = fig._pysynth_effect_sources
    panels = fig._pysynth_facet_labels
    in_panel = set(grouped)
    note_artists = [artist for artist, _ in fig._pysynth_comparison_notes]
    for _, labels, _ in panels.values():
        for label in labels:
            label.set_visible(True)
    while True:
        notes = _comparison_source_notes(references, grouped, in_panel=in_panel)
        fig._pysynth_comparison_notes = [
            (artist, text) for artist, (text, _) in zip(note_artists, notes)
        ]
        titles = {ax: f" [{keys}]" for ax, (fit, _, keys) in panels.items() if fit not in in_panel}
        _layout_captions(fig, title_suffixes=titles)
        renderer = fig.canvas.get_renderer()
        crowded = {
            fit
            for ax, (fit, labels, _) in panels.items()
            if fit in in_panel and not _place_facet_sources(ax, labels, renderer)
        }
        if not crowded:
            return
        # Each pass moves at least one fit to the caption. A longer caption
        # may reduce the room available for other panels on a fixed-size export.
        in_panel -= crowded
        for fit, labels, _ in panels.values():
            if fit in crowded:
                for label in labels:
                    label.set_visible(False)


def _layout_annotations(fig):
    """Choose empty corners using rendered path geometry and prior callouts."""
    from matplotlib.transforms import Bbox

    renderer = fig.canvas.get_renderer()
    occupied = {}
    for artist in getattr(fig, "_pysynth_direct_annotations", []):
        ax = artist.axes
        bounds = ax.get_window_extent(renderer)
        artist.set_text(
            _wrap_measured(artist._pysynth_callout_text, artist, renderer, bounds.width * 0.87)
        )
        points = []
        for line in ax.lines:
            if line.get_label() == "Intervention":
                continue
            vertices = line.get_transform().transform(line.get_path().vertices)
            if len(vertices) > 1:
                # Sample segments too: sparse annual data must not hide collisions.
                vertices = np.concatenate(
                    [
                        a + np.linspace(0, 1, 12)[:, None] * (b - a)
                        for a, b in zip(vertices[:-1], vertices[1:])
                    ]
                )
            points.extend(vertices)
        points = np.asarray(points).reshape(-1, 2)
        previous = occupied.setdefault(
            ax,
            [
                t.get_window_extent(renderer)
                for t in ax.texts
                if t not in getattr(fig, "_pysynth_direct_annotations", [])
            ],
        )
        best = None
        for horizontal, vertical in [
            (0.03, 0.96),
            (0.03, 0.04),
            (0.97, 0.96),
            (0.97, 0.04),
            (0.03, 0.50),
            (0.97, 0.50),
        ]:
            artist.set_position((horizontal, vertical))
            artist.set_ha("left" if horizontal < 0.5 else "right")
            artist.set_va("top" if vertical > 0.5 else "bottom")
            artist.update_positions(renderer)
            artist.update_bbox_position_size(renderer)
            box = artist.get_bbox_patch().get_window_extent(renderer).expanded(1.03, 1.08)
            # Font descent and box padding are asymmetric: equal fractional
            # margins can put only the bottom candidate beyond the axes. Move
            # each candidate inside first, so a tiny overhang cannot make us
            # prefer a position covering a curve over an otherwise empty corner.
            horizontal += (max(0, bounds.x0 - box.x0) - max(0, box.x1 - bounds.x1)) / bounds.width
            vertical += (max(0, bounds.y0 - box.y0) - max(0, box.y1 - bounds.y1)) / bounds.height
            artist.set_position((horizontal, vertical))
            artist.update_positions(renderer)
            artist.update_bbox_position_size(renderer)
            box = artist.get_bbox_patch().get_window_extent(renderer).expanded(1.03, 1.08)
            outside = (
                max(0, bounds.x0 - box.x0)
                + max(0, box.x1 - bounds.x1)
                + max(0, bounds.y0 - box.y0)
                + max(0, box.y1 - bounds.y1)
            )
            overlap = sum(
                (Bbox.intersection(box, old).width * Bbox.intersection(box, old).height)
                for old in previous
                if Bbox.intersection(box, old) is not None
            )
            covered = np.sum(
                (points[:, 0] >= box.x0)
                & (points[:, 0] <= box.x1)
                & (points[:, 1] >= box.y0)
                & (points[:, 1] <= box.y1)
            )
            target = (ax.transData if artist.arrow_patch is not None else ax.transAxes).transform(
                artist.xy
            )
            distance = np.linalg.norm(
                target - np.array([(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2])
            )
            score = outside * 1000 + overlap * 10 + covered * 8 + distance * 0.01
            if best is None or score < best[0]:
                best = (score, horizontal, vertical, box)
        _, horizontal, vertical, box = best
        artist.set_position((horizontal, vertical))
        artist.set_ha("left" if horizontal < 0.5 else "right")
        artist.set_va("top" if vertical > 0.5 else "bottom")
        previous.append(box)
    fig.canvas.draw()


def _layout_row_annotations(fig):
    renderer = fig.canvas.get_renderer()
    for artist, original in getattr(fig, "_pysynth_row_annotations", []):
        ax = artist.axes
        bounds = ax.get_window_extent(renderer)
        text = _wrap_measured(original, artist, renderer, bounds.width * 0.94)
        artist.set_text(text)
        width = max(
            renderer.get_text_width_height_descent(line, artist.get_fontproperties(), ismath=False)[
                0
            ]
            for line in text.splitlines()
        )
        target = ax.transData.transform(artist.xy)[0]
        left = min(target + 7 * fig.dpi / 72, bounds.x1 - width - bounds.width * 0.02)
        fraction = max(0.02, (left - bounds.x0) / bounds.width)
        artist.set_anncoords(("axes fraction", "offset points"))
        artist.set_position((fraction, -12))


def _direct_path_labels(ax, x, estimate, observed_label, counterfactual_label, color=BLUE):
    end = x[-1]
    span = x[-1] - x[0]
    ticks = ax.get_xticks()
    ax.set_xlim(ax.get_xlim()[0], end + 0.32 * span)
    ax.set_xticks([tick for tick in ticks if x[0] <= tick <= end])
    low, high = ax.get_ylim()
    values = [estimate.observed.iloc[-1], estimate.counterfactual.iloc[-1]]
    separation = (high - low) * 0.10
    positions = list(values)
    if abs(values[0] - values[1]) < separation:
        order = np.argsort(values)
        center = sum(values) / 2
        positions[order[0]], positions[order[1]] = center - separation / 2, center + separation / 2
    for value, position, label, color in zip(
        values, positions, (observed_label, counterfactual_label), (INK, color)
    ):
        ax.annotate(
            fill(label, width=20),
            xy=(end, value),
            xytext=(end + 0.035 * span, position),
            ha="left",
            va="center",
            fontsize=8,
            fontweight="semibold",
            color=color,
            linespacing=1.25,
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.7},
            annotation_clip=False,
            zorder=5,
        )


def _validate_benchmark(benchmark, estimate):
    if benchmark is None:
        return
    if not isinstance(benchmark, LiteratureBenchmark):
        raise TypeError("benchmark must be a LiteratureBenchmark")
    if benchmark.value is not None:
        index = estimate.observed.index
        positions = index.get_indexer(benchmark.window)
        if not np.array_equal(positions, [estimate.n_pre, len(index) - 1]):
            raise ValueError(
                "benchmark window must match the full post-treatment sample; "
                "use a text-only benchmark for a different window"
            )


def _benchmark_effect(ax, x, n_pre, benchmark):
    if benchmark is not None and benchmark.value is not None:
        ax.plot(
            [x[n_pre], x[-1]],
            [benchmark.value, benchmark.value],
            color=LITERATURE,
            linewidth=1,
            linestyle=(0, (2, 2)),
            label="Literature mean ATT",
            zorder=2.5,
        )


def _effect_limits(values, benchmark):
    if benchmark is not None and benchmark.value is not None:
        values = np.append(values, benchmark.value)
    return _limits(values)


def _matplotlib():
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib: pip install 'pysynth-control[plot]'"
        ) from exc
    return mpl, plt


@cache
def _register_fonts():
    from matplotlib import font_manager

    manager = font_manager.fontManager
    original_count = len(manager.ttflist)
    for _, filename, _ in _FONT_FACES:
        manager.addfont(_FONT_DIR / filename)
    # addfont may append multiple names per face. findfont breaks ties on
    # the first match, so prefer all bundled entries over system duplicates.
    manager.ttflist[:] = manager.ttflist[original_count:] + manager.ttflist[:original_count]


def publication_title_font():
    """Return fresh Matplotlib font properties for a STIX Two Text semibold heading.

    Pass the result as ``fontproperties`` to a custom title. The explicit
    TrueType file keeps the same face in PDFs even on hosts with a system
    OpenType version of STIX Two Text.
    """
    _matplotlib()
    from matplotlib.font_manager import FontProperties

    return FontProperties(
        family=["STIX Two Text", "serif"],
        fname=_FONT_DIR / "STIXTwoText-SemiBold.ttf",
        weight="semibold",
    )


@contextmanager
def publication_style():
    """Temporarily use legible typography, restrained axes and vector fonts.

    The packaged Source Sans 3 faces have tabular lining numerals, and
    figure headings use STIX Two Text. Use :func:`save_figure` to embed
    fonts in vector exports. No external fonts or TeX are required.
    """
    mpl, _ = _matplotlib()
    _register_fonts()
    with mpl.rc_context(
        {
            "font.family": ["Source Sans 3", "sans-serif"],
            "font.sans-serif": ["DejaVu Sans"],
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "font.size": 9.5,
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
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
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


def _figure_grid(figsize, title, subtitle, note, benchmark=None, references=(), *, source_notes=()):
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
    for text, _ in source_notes:
        foot_height += 0.14 * len(fill(text, width=max(25, int(width * 19))).splitlines())
    fig = plt.figure(figsize=figsize, layout="constrained")
    fig.get_layout_engine().set(h_pad=0.06, w_pad=0.06, hspace=0.045, wspace=0.05)
    ratios = [head_height, max(1, height - head_height - foot_height), foot_height]
    if benchmark is not None or references:
        ratios.insert(1, 0.65)
    outer = fig.add_gridspec(len(ratios), 1, height_ratios=ratios)
    if benchmark is not None:
        from matplotlib.patches import Rectangle

        strip = fig.add_subplot(outer[1])
        strip.set_axis_off()
        strip.add_patch(
            Rectangle(
                (0, 0),
                1,
                1,
                transform=strip.transAxes,
                facecolor="#F7F3ED",
                edgecolor="none",
                zorder=0,
            )
        )
        strip.plot(
            [0, 0], [0, 1], transform=strip.transAxes, color=LITERATURE, linewidth=2, clip_on=False
        )
        originals = [
            f"{benchmark.label} · {benchmark.headline}",
            benchmark.citation,
            benchmark.context,
        ]
        artists = []
        for i, text in enumerate(originals):
            artist = strip.text(
                0.018,
                1,
                text,
                transform=strip.transAxes,
                va="top",
                fontsize=8.5 if i == 0 else 7.3,
                fontweight="semibold" if i == 0 else "normal",
                color=LITERATURE if i == 0 else INK,
                linespacing=1.3,
            )
            artist.set_in_layout(False)
            if i == 1:
                artist.set_url(benchmark.source_url)
            artists.append(artist)
        fig._pysynth_benchmark_layout = (strip, artists, originals)
    elif references:
        _reference_strip(fig, outer[1], references)
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
            fontsize=15,
            fontproperties=publication_title_font(),
            linespacing=1.18,
        )
    if wrapped_subtitle:
        subtitle_artist = heading.text(
            0,
            0,
            wrapped_subtitle,
            transform=heading.transAxes,
            va="bottom",
            fontsize=9.5,
            color=MUTED,
            linespacing=1.3,
        )
    footer = fig.add_subplot(outer[-1])
    footer.set_axis_off()
    if source_notes:
        fig._pysynth_comparison_notes = []
        for text, url in source_notes:
            artist = footer.text(
                0,
                1,
                text,
                transform=footer.transAxes,
                va="top",
                fontsize=7.8,
                color=LITERATURE,
                linespacing=1.2,
                url=url,
            )
            artist.set_in_layout(False)
            fig._pysynth_comparison_notes.append((artist, text))
    if wrapped_note:
        note_artist = footer.text(
            0,
            1,
            wrapped_note,
            transform=footer.transAxes,
            va="top",
            fontsize=8,
            color=MUTED,
            linespacing=1.3,
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
        heading,
    )
    return fig, outer[-2]


def _reference_strip(fig, cell, references):
    """A compact, explicitly paired source/result table above the plotting area."""
    from matplotlib.patches import Rectangle

    strip = fig.add_subplot(cell)
    strip.set_axis_off()
    strip.add_patch(
        Rectangle((0, 0), 1, 1, transform=strip.transAxes, facecolor="#F7F3ED", edgecolor="none")
    )
    strip.plot(
        [0, 0], [0, 1], transform=strip.transAxes, color=LITERATURE, linewidth=2, clip_on=False
    )

    def text(x, content, *, size=8, color=INK, align="left", weight="normal"):
        artist = strip.text(
            x,
            1,
            content,
            transform=strip.transAxes,
            va="top",
            ha=align,
            fontsize=size,
            color=color,
            fontweight=weight,
            linespacing=1.35,
        )
        artist.set_in_layout(False)
        return artist

    heading = text(0.018, "REFERENCE CHECK · Δ = reproduced − source", size=7.5, color=MUTED)
    headings = [
        text(x, label, size=7.3, color=MUTED, align="right")
        for x, label in [(0.62, "Source"), (0.80, "Reproduced"), (0.982, "Δ")]
    ]
    rows = []
    for ref in references:
        window = ref.window_label or f"{ref.window[0]}–{ref.window[1]}"
        label = f"{ref.label} · {window}\n{ref.units}"
        label_artist = text(0.018, label, size=8, weight="semibold")
        prefix = "≈ " if ref.approximate else ""
        values = [
            prefix + _reference_number(ref.reference, ref.reference_precision),
            _reference_number(ref.reproduced, ref.precision),
            _reference_difference(ref, symbol=False) if ref.comparable else "—",
        ]
        value_artists = [
            text(x, value, size=9, align="right", color=color, weight="semibold")
            for x, value, color in zip((0.62, 0.80, 0.982), values, (LITERATURE, BLUE, INK))
        ]
        provenance = f"{ref.source_label}: {ref.citation}. {ref.context}"
        source_artist = text(0.018, provenance, size=7, color=MUTED)
        source_artist.set_url(ref.source_url)
        rows.append((label_artist, label, value_artists, source_artist, provenance))
    fig._pysynth_reference_layout = (strip, heading, headings, rows)


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
        if hasattr(fig, "_pysynth_effect_sources"):
            _layout_effect_sources(fig)
        else:
            _layout_captions(fig)
        _layout_row_annotations(fig)
        _layout_annotations(fig)
        if getattr(fig, "_pysynth_body_y_label", None) is not None:
            label, axes = fig._pysynth_body_y_label
            positions = [ax.get_position() for ax in axes if ax.get_visible()]
            label.set_y((min(b.y0 for b in positions) + max(b.y1 for b in positions)) / 2)
    finally:
        fig.set_canvas(original_canvas)


def _layout_captions(fig, *, title_suffixes=None):
    layout = getattr(fig, "_pysynth_caption_layout", None)
    if layout is None:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outer, title, subtitle, note, title_text, subtitle_text, note_text, heading_axes = layout
    # GridSpec distributes the space left after axis labels, ticks and margins.
    # A ratio unit consequently occupies fewer pixels than a whole figure inch.
    ratios = outer.get_height_ratios()
    grid_height = heading_axes.get_window_extent(renderer).height * sum(ratios) / ratios[0]
    pixels_per_unit = grid_height / fig.get_figheight()

    def text_height(artist):
        nominal = len(artist.get_text().splitlines()) * artist.get_fontsize() / 72 * 1.4
        return max(nominal, artist.get_window_extent(renderer).height / pixels_per_unit)

    heights = []
    for artist, text in zip((title, subtitle, note), (title_text, subtitle_text, note_text)):
        if artist is None:
            heights.append(0)
            continue
        width = artist.axes.get_window_extent(renderer).width * 0.96
        wrapped = _wrap_measured(text, artist, renderer, width)
        artist.set_text(wrapped)
        heights.append(text_height(artist))
    head = heights[0] + heights[1] + 0.12
    foot = heights[2] + 0.07 if note is not None else 0.02
    source_notes = getattr(fig, "_pysynth_comparison_notes", [])
    if source_notes:
        offset = heights[2] + 0.04 if note is not None else 0
        positions = []
        for artist, original in source_notes:
            width = artist.axes.get_window_extent(renderer).width * 0.98
            artist.set_text(_wrap_measured(original, artist, renderer, width))
            positions.append(offset)
            offset += text_height(artist) + 0.025
        foot = offset + 0.04
        for (artist, _), position in zip(source_notes, positions):
            artist.set_y(1 - position / foot)
    benchmark_layout = getattr(fig, "_pysynth_benchmark_layout", None)
    reference_layout = getattr(fig, "_pysynth_reference_layout", None)
    benchmark_height = 0
    if benchmark_layout is not None:
        strip, artists, originals = benchmark_layout
        text_heights = []
        for artist, original in zip(artists, originals):
            width = strip.get_window_extent(renderer).width * 0.95
            wrapped = _wrap_measured(original, artist, renderer, width)
            artist.set_text(wrapped)
            text_heights.append(text_height(artist))
        benchmark_height = sum(text_heights) + 0.18
        offset = 0.055
        for artist, text_height in zip(artists, text_heights):
            artist.set_y(1 - offset / benchmark_height)
            offset += text_height + 0.015
    elif reference_layout is not None:
        strip, heading, headings, rows = reference_layout
        width = strip.get_window_extent(renderer).width
        row_heights = []
        for label, original, values, source, provenance in rows:
            label.set_text(_wrap_measured(original, label, renderer, width * 0.45))
            source.set_text(_wrap_measured(provenance, source, renderer, width * 0.96))
            row_height = max(0.22, text_height(label), *(text_height(value) for value in values))
            source_height = text_height(source)
            row_heights.append((row_height, source_height))
        benchmark_height = 0.38 + sum(sum(heights) + 0.13 for heights in row_heights)
        heading.set_y(1 - 0.055 / benchmark_height)
        for artist in headings:
            artist.set_y(1 - 0.23 / benchmark_height)
        offset = 0.40
        for (label, _, values, source, _), (row_height, source_height) in zip(rows, row_heights):
            for artist in [label, *values]:
                artist.set_y(1 - offset / benchmark_height)
            offset += row_height + 0.025
            source.set_y(1 - offset / benchmark_height)
            offset += source_height + 0.105
    for ax, original in getattr(fig, "_pysynth_axis_y_labels", []):
        ax.set_ylabel(
            _wrap_measured(
                original, ax.yaxis.label, renderer, ax.get_window_extent(renderer).height * 0.95
            )
        )
    body_height = max(1, fig.get_figheight() - head - foot - benchmark_height)
    ratios = [head, body_height, foot]
    if benchmark_layout is not None or reference_layout is not None:
        ratios.insert(1, benchmark_height)
    outer.set_height_ratios(ratios)
    for artist, original in getattr(fig, "_pysynth_panel_titles", []):
        width = artist.axes.get_window_extent(renderer).width * 0.98
        if title_suffixes:
            original += title_suffixes.get(artist.axes, "")
        artist.set_text(_wrap_measured(original, artist, renderer, width))
    fig.canvas.draw()
    # Shared labels belong beside the data panels, not halfway down a figure
    # whose upper half may contain several source/reference rows.
    body_label = getattr(fig, "_pysynth_body_y_label", None)
    if body_label is not None:
        artist, axes = body_label
        positions = [ax.get_position() for ax in axes if ax.get_visible()]
        bottom, top = min(box.y0 for box in positions), max(box.y1 for box in positions)
        original = getattr(fig, "_pysynth_body_y_label_text", artist.get_text())
        artist.set_text(
            _wrap_measured(original, artist, renderer, (top - bottom) * fig.bbox.height * 0.95)
        )
        artist.set_y((bottom + top) / 2)


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
    benchmark: LiteratureBenchmark | None = None,
    references: Sequence[ReferenceComparison] | None = None,
    reference_style: str = "direct",
    color: str = BLUE,
    figsize: tuple | None = None,
):
    """Plot an observed/counterfactual pair or a single effect path; return Axes.

    ``effects=True`` plots observed minus counterfactual and a zero reference.
    The dashed vertical line marks the first affected observation; the light
    background marks the post-period. It is not an uncertainty band. Numeric
    and calendar time retain their actual spacing. Supply ``ylabel`` for
    substantive units and ``ax`` to compose a larger figure. ``subtitle`` and
    ``note`` reserve figure-level caption space and require ``ax=None``.
    ``benchmark`` adds a sourced annotation; its numeric guide is drawn only
    with ``effects=True``, never on an outcome-level axis.
    ``references`` annotates source, reproduction and signed difference beside
    the matched path. Curves are labelled directly. Use it instead of ``benchmark``.
    ``reference_style="strip"`` restores the earlier table above the axes.
    """
    _annotation_style(reference_style)
    _validate(estimate)
    _validate_benchmark(benchmark, estimate)
    references = _plot_references(references, estimate, benchmark)
    if ax is not None and (
        subtitle is not None or note is not None or benchmark is not None or references
    ):
        raise ValueError(
            "subtitle, note and references/benchmark require ax=None; add captions to your parent figure"
        )
    if reference_style == "direct":
        references = tuple(ref for ref in references if ref.estimate is estimate)
    with publication_style():
        own_figure = ax is None
        if ax is None:
            fig, body = _figure_grid(
                figsize
                or (
                    7.2,
                    3.8
                    + (0.75 if benchmark and reference_style == "strip" else 0)
                    + (
                        0.4 + 0.85 * len(references)
                        if references and reference_style == "strip"
                        else 0
                    ),
                ),
                title or estimate.method,
                subtitle,
                _direct_note(note, references, benchmark) if reference_style == "direct" else note,
                benchmark if reference_style == "strip" else None,
                references if reference_style == "strip" else (),
            )
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
            _benchmark_effect(ax, x, estimate.n_pre, benchmark)
            _reference_effect(ax, x, references, estimate)
            values = np.append(
                estimate.effects,
                [ref.reference for ref in _matched_references(references, estimate)],
            )
            ax.set_ylim(*_effect_limits(values, benchmark))
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
            if reference_style == "direct":
                _direct_path_labels(ax, x, estimate, observed_label, counterfactual_label, color)
            else:
                ax.legend(
                    *zip(*selected),
                    loc="lower left",
                    bbox_to_anchor=(0, 1.01),
                    ncols=2,
                    borderaxespad=0,
                    handlelength=2.8,
                    columnspacing=2,
                )
        if reference_style == "direct":
            _annotate_references(ax, x, references, estimate, effects=effects, color=color)
            if benchmark is not None:
                _callout(
                    ax,
                    f"{benchmark.label}: {benchmark.headline} [1]",
                    ((x[estimate.n_pre] + x[-1]) / 2, benchmark.value)
                    if effects and benchmark.value is not None
                    else None,
                    url=benchmark.source_url,
                )
        if intervention_label is not None:
            ax.text(
                x[estimate.n_pre],
                1.02 if reference_style == "direct" else 0.98,
                f" {intervention_label}",
                transform=ax.get_xaxis_transform(),
                va="bottom" if reference_style == "direct" else "top",
                fontsize=8,
                color=MUTED,
                bbox={"facecolor": POST, "edgecolor": "none", "pad": 2},
            )
        if not own_figure:
            ax.set_title(
                title or estimate.method,
                loc="left",
                fontproperties=publication_title_font(),
                pad=31 if not effects else 12,
            )
        label = ylabel or ("Observed − counterfactual" if effects else "Outcome")
        ax.set_ylabel(label)
        if own_figure:
            ax.figure._pysynth_axis_y_labels = [(ax, label)]
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
    benchmark: LiteratureBenchmark | None = None,
    references: Sequence[ReferenceComparison] | None = None,
    reference_style: str = "direct",
    note: str
    | None = "Common outcome sample and treatment date. Shading marks the post-period; no uncertainty intervals are shown.",
):
    """Compare effect paths in labelled small multiples; return Figure, Axes array.

    Mapping insertion order is preserved. Both axes are shared by default;
    common y limits include every effect and zero, without clipping outliers.
    ``sharey=False`` explicitly allows different scales. ``labels`` and
    ``colors`` customize display without renaming estimates. Unused panels
    are hidden. The returned array always has shape (nrows, ncols).
    A numeric ``benchmark`` adds a dotted source mean over the matching
    post-period. Panels without a paired source show the mean ATT in the
    lower left, with a dotted line across the post-treatment period.
    ``references`` pairs each source with its exact fitted specification. Any
    reference fit absent from ``estimates`` gets a separate, labelled panel.
    Every fit occupies one equal-sized panel. Multiple sources are labelled
    together inside the matching panel. A keyed citation carries the details
    when the panel has insufficient room. Use
    ``reference_style="strip"`` for the earlier table layout without added panels.
    """
    _annotation_style(reference_style)
    _estimates(estimates)
    if isinstance(ncols, bool) or not isinstance(ncols, (int, np.integer)) or ncols < 1:
        raise ValueError("ncols must be a positive integer")
    if ylim is not None and (len(ylim) != 2 or not np.isfinite(ylim).all() or ylim[0] >= ylim[1]):
        raise ValueError("ylim must be two finite, increasing values")
    reference = next(iter(estimates.values()))
    _validate_benchmark(benchmark, reference)
    references = _plot_references(references, reference, benchmark)
    if reference_style == "direct":
        estimates = _reference_fits(estimates, references)
    ncols = min(ncols, len(estimates))
    nrows = (len(estimates) + ncols - 1) // ncols
    by_fit = _references_by_fit(references) if reference_style == "direct" else {}
    grouped = {key: refs for key, refs in by_fit.items() if len(refs) > 1}
    source_notes = (
        _comparison_source_notes(references, grouped, in_panel=grouped) if grouped else []
    )
    with publication_style():
        fig, body = _figure_grid(
            figsize
            or (
                7.2,
                1.1
                + 1.85 * nrows
                + sum(0.15 * len(fill(text, width=110).splitlines()) for text, _ in source_notes)
                + (0.75 if benchmark and reference_style == "strip" else 0)
                + (
                    0.4 + 0.85 * len(references) if references and reference_style == "strip" else 0
                ),
            ),
            title,
            subtitle,
            _direct_note(note, references, benchmark, include_citations=not grouped)
            if reference_style == "direct"
            else note,
            benchmark if reference_style == "strip" else None,
            references if reference_style == "strip" else (),
            source_notes=source_notes,
        )
        grid = body.subgridspec(nrows, ncols, hspace=0.22, wspace=0.12)
        axes = np.empty((nrows, ncols), dtype=object)
        fig._pysynth_panel_titles = []
        if grouped:
            fig._pysynth_effect_sources = (references, grouped)
            fig._pysynth_facet_labels = {}
        x, kind = _time(reference.observed.index)
        common = ylim or _effect_limits(
            np.concatenate(
                [
                    *[r.effects.to_numpy() for r in estimates.values()],
                    [
                        ref.reference
                        for result in estimates.values()
                        for ref in _matched_references(references, result)
                    ],
                ]
            ),
            benchmark,
        )
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
            _benchmark_effect(ax, x, reference.n_pre, benchmark)
            _reference_effect(ax, x, references, result)
            ax.set_ylim(
                *(
                    common
                    if sharey or ylim is not None
                    else _effect_limits(
                        np.append(
                            result.effects,
                            [ref.reference for ref in _matched_references(references, result)],
                        ),
                        benchmark,
                    )
                )
            )
            if reference_style == "direct":
                _annotate_references(
                    ax,
                    x,
                    references,
                    result,
                    effects=True,
                    small=True,
                    color=(colors or {}).get(name, BLUE),
                )
                if benchmark is not None and position == 0:
                    _callout(
                        ax,
                        f"{benchmark.label}: {benchmark.headline} [1]",
                        ((x[result.n_pre] + x[-1]) / 2, benchmark.value)
                        if benchmark.value is not None
                        else None,
                        width=27,
                        fontsize=7.4,
                        url=benchmark.source_url,
                    )
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
        for position in range(len(estimates), axes.size):
            row, col = divmod(position, ncols)
            axes[row, col] = fig.add_subplot(grid[row, col])
            axes[row, col].set_visible(False)
            axes[row, col].set_in_layout(False)
        shared_label = fig.supylabel(ylabel, fontsize=9)
        fig._pysynth_body_y_label = (shared_label, tuple(axes.flat))
        fig._pysynth_body_y_label_text = ylabel
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
    benchmark: LiteratureBenchmark | None = None,
    references: Sequence[ReferenceComparison] | None = None,
    reference_style: str = "direct",
    note: str
    | None = "Dots are point estimates, without confidence intervals. Pretreatment RMSE describes fit, not causal credibility.",
):
    """Plot aligned ATT and pre-RMSE dot plots; return Figure, two Axes.

    Methods retain mapping order, with a shared label column and separate
    x scales. ATT includes zero; RMSE starts at zero. Every method must share
    the same observed path, ordered time index and treatment boundary. No
    error bars, significance tests, rankings or uncertainty are inferred.
    ``benchmark`` adds a sourced annotation and, if numeric, a dotted guide
    on the ATT axis only. Its averaging window must match the full post-period.
    ``references`` adds paired source/result values instead; a source diamond
    is drawn only beside its exact fitted method and on the same ATT window.
    Direct annotations are the default. Fits with several sources carry
    citation numbers beside the method name and linked comparisons below the
    axes. Absent historical fits get separate labelled rows.
    ``reference_style="strip"`` restores the table above the axes.
    """
    _annotation_style(reference_style)
    table = _estimates(estimates)
    _validate_benchmark(benchmark, next(iter(estimates.values())))
    references = _plot_references(references, next(iter(estimates.values())), benchmark)
    if reference_style == "direct":
        estimates = _reference_fits(estimates, references)
        table = _estimates(estimates)
    by_fit = _references_by_fit(references) if reference_style == "direct" else {}
    grouped = {key: refs for key, refs in by_fit.items() if len(refs) > 1}
    inline_refs = {key: refs[0] for key, refs in by_fit.items() if len(refs) == 1}
    source_numbers = _source_numbers(references)
    source_notes = _comparison_source_notes(references, grouped) if grouped else []
    # Reserve space for the caption independently of the statistical rows,
    # including on a chart that contains only the historical specification.
    source_height = sum(
        0.2 * len(fill(text, width=110).splitlines()) + 0.08 for text, _ in source_notes
    )
    # Source details already have their own height below the axes. Use an
    # 18-point pitch for the dots so the saved space shortens the page.
    row_height = 0.25 if source_notes else 0.31
    with publication_style():
        from matplotlib.ticker import FuncFormatter, MaxNLocator

        fig, body = _figure_grid(
            figsize
            or (
                7.2,
                max(2.1 if source_notes else 0, 1.55 + row_height * len(table))
                + source_height
                + sum(0.25 + 0.12 * _reference_lines(ref) for ref in inline_refs.values())
                # Either a source strip or a direct benchmark and its citation.
                + (0.75 if benchmark else 0)
                + (
                    0.4 + 0.85 * len(references) if references and reference_style == "strip" else 0
                ),
            ),
            title,
            subtitle,
            _direct_note(note, references, benchmark, include_citations=not grouped)
            if reference_style == "direct"
            else note,
            benchmark if reference_style == "strip" else None,
            references if reference_style == "strip" else (),
            source_notes=source_notes,
        )
        grid = body.subgridspec(1, 2, width_ratios=[1.2, 1], wspace=0.15)
        axes = np.array([fig.add_subplot(grid[0]), fig.add_subplot(grid[1])])
        y = np.arange(len(table), dtype=float)
        bottom_padding = 0.7
        display = []
        for row, (name, result) in enumerate(estimates.items()):
            label = fill(str((labels or {}).get(name, name)), width=26)
            if id(result) in grouped:
                keys = dict.fromkeys(
                    source_numbers[(r.citation, r.source_url)] for r in grouped[id(result)]
                )
                label += " [" + ", ".join(map(str, keys)) + "]"
            ref = inline_refs.get(id(result))
            extra = _reference_spacing(ref) if ref is not None else 0
            y[row + 1 :] += extra
            if row == len(y) - 1:
                bottom_padding += extra
            display.append(label)
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
            ax.set_ylim(y[-1] + bottom_padding, -0.5)
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
            for result, value, row in zip(estimates.values(), values, y):
                matching = _matched_references(references, result, full_window=True)
                label = (
                    _reference_number(value, max(ref.precision for ref in matching))
                    if metric == "att" and matching and reference_style == "direct"
                    else _number(value)
                )
                ax.annotate(
                    label,
                    (value, row),
                    xytext=(7, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8.5,
                    color=INK,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
                )
                ref = inline_refs.get(id(result))
                if metric == "att" and ref is not None:
                    compact = ref in matching
                    text = _reference_text(
                        ref, source_numbers[(ref.citation, ref.source_url)], compact=compact
                    )
                    artist = ax.annotate(
                        text,
                        (value, row),
                        xytext=(7, -12),
                        textcoords="offset points",
                        va="top",
                        ha="left",
                        fontsize=7.5,
                        color=LITERATURE,
                        linespacing=1.2,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
                    )
                    artist.set_in_layout(False)
                    fig._pysynth_row_annotations = getattr(fig, "_pysynth_row_annotations", []) + [
                        (artist, text)
                    ]
                    artist.set_url(ref.source_url)
            low, high = min(0, values.min()), max(0, values.max())
            if metric == "att":
                for row, result in enumerate(estimates.values()):
                    for ref in _matched_references(references, result, full_window=True):
                        ax.scatter(
                            [ref.reference],
                            [y[row]],
                            s=36,
                            marker="D",
                            facecolors="none",
                            edgecolors=LITERATURE,
                            linewidth=1.1,
                            zorder=4,
                            label=f"{ref.source_label}: {ref.label}",
                        )
                        low, high = min(low, ref.reference), max(high, ref.reference)
            if metric == "att" and reference_style == "direct":
                if benchmark is not None:
                    _callout(
                        ax,
                        f"{benchmark.label}: {benchmark.headline} [1]",
                        (benchmark.value, (y[0] + y[-1]) / 2)
                        if benchmark.value is not None
                        else None,
                        width=30,
                        fontsize=7.5,
                        url=benchmark.source_url,
                    )
            if metric == "att" and benchmark is not None and benchmark.value is not None:
                ax.axvline(
                    benchmark.value,
                    color=LITERATURE,
                    linewidth=1,
                    linestyle=(0, (2, 2)),
                    label="Literature mean ATT",
                    zorder=2.5,
                )
                low, high = min(low, benchmark.value), max(high, benchmark.value)
            span = high - low or 1.0
            ax.set_xlim(0 if metric == "pre_rmse" else low - span * 0.12, high + span * 0.37)
            ax.xaxis.set_major_locator(
                MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3)
            )
            ax.xaxis.set_major_formatter(FuncFormatter(_tick_number))
            ax.set_title(heading, loc="left", fontsize=10, fontweight="semibold", pad=12)
            ax.set_xlabel(f"{'ATT' if metric == 'att' else 'RMSE'} ({units})", fontsize=8.5)
        _finish_figure(fig)
        return fig, axes


def _svg_font_face(filename, alias, weight, characters):
    """Build an OFL webfont subset for the characters in this editable figure."""
    from base64 import b64encode
    from io import BytesIO

    from fontTools import subset
    from fontTools.ttLib import TTFont

    options = subset.Options()
    options.name_IDs = list(range(18))  # Keep authorship, copyright and license records.
    options.layout_features = ["*"]
    with TTFont(_FONT_DIR / filename, recalcTimestamp=False) as font:
        subsetter = subset.Subsetter(options=options)
        subsetter.populate(unicodes=characters)
        subsetter.subset(font)
        # A subset is a modified font. Give it its own family and PostScript
        # names while retaining all the original author and licensing records.
        style = "Regular" if weight == 400 else "Semibold"
        full_name = f"{alias} {style}"
        ps_name = full_name.replace(" ", "")
        names = {1: alias, 2: style, 3: ps_name, 4: full_name, 6: ps_name, 16: alias, 17: style}
        for record in font["name"].names:
            if record.nameID in names:
                record.string = names[record.nameID]
        font.flavor = "woff"
        buffer = BytesIO()
        font.save(buffer)
    data = b64encode(buffer.getvalue()).decode("ascii")
    return (
        f"@font-face {{ font-family: '{alias}'; font-weight: {weight}; "
        f"src: url(data:font/woff;base64,{data}) format('woff'); }}"
    )


def _embed_svg_fonts(svg):
    """Keep text editable and its measured typefaces available in standalone SVG."""
    from xml.dom import minidom

    families = {
        "Source Sans 3": ("PySynthCon Figure Sans", "SourceSans3-LICENSE.md"),
        "STIX Two Text": ("PySynthCon Figure Serif", "STIXTwoText-OFL.txt"),
    }
    document = minidom.parseString(svg)
    characters = set()
    used = set()
    for tag in ("text", "tspan"):
        for element in document.getElementsByTagName(tag):
            for node in element.childNodes:
                if node.nodeType == node.TEXT_NODE:
                    characters.update(map(ord, node.data))
            style = element.getAttribute("style")
            for family, (alias, _) in families.items():
                # Matplotlib quotes named families and writes generic families
                # unquoted. Keep original and generic fallbacks for SVG editors.
                name = f"'{family}'"
                if name in style:
                    used.add(family)
                    style = style.replace(name, f"'{alias}', {name}")
                    element.setAttribute("style", style)
    if not used:
        return svg
    rules = [
        _svg_font_face(filename, families[family][0], weight, characters)
        for family, filename, weight in _FONT_FACES
        if family in used
    ]
    style = document.createElement("style")
    style.appendChild(document.createTextNode("\n".join(rules)))
    document.getElementsByTagName("defs")[0].appendChild(style)
    metadata = document.createElement("metadata")
    licenses = [
        (_FONT_DIR / families[name][1]).read_text(encoding="utf-8") for name in sorted(used)
    ]
    metadata.appendChild(document.createTextNode("\n".join(licenses)))
    document.documentElement.appendChild(metadata)
    return document.toxml(encoding="utf-8")


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
    from io import BytesIO

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
                metadata = {"Creator": "PySynthCon", "CreationDate": None, "ModDate": None}
            elif fmt == "svg":
                metadata = {"Creator": "PySynthCon", "Date": None}
            else:
                metadata = {"Software": "PySynthCon"}
            output = BytesIO() if fmt == "svg" else destination
            figure.savefig(
                output,
                format=fmt,
                dpi=dpi,
                facecolor="white",
                bbox_inches=None,
                metadata=metadata,
            )
            if fmt == "svg":
                destination.write_bytes(_embed_svg_fonts(output.getvalue()))
            destinations.append(destination)
    return tuple(destinations)
