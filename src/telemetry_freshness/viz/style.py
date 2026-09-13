"""Figure styling for the paper.

One place defines typography, colour and output format so every figure in the
manuscript reads as one system.

Colour policy
-------------
* **Ordered levels** (update rate, latency, traffic density) use a single-hue
  sequential ramp, light to dark -- these are magnitudes, not identities, and a
  categorical palette would throw away their order.
* **Unordered categories** (encounter geometry, dataset) use a fixed categorical
  order, assigned by slot and never cycled.
* Series are always direct-labelled in addition to the legend: three of the
  categorical slots fall below 3:1 against a light surface, so identity must
  never rest on colour alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects  # noqa: F401 (registers mpl.patheffects)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba

#: Validated categorical order (adjacent-pair CVD dE 9.1, normal-vision 19.6).
CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#4a3aa7",  # violet
)

#: Single-hue sequential ramp for ordered levels.
SEQUENTIAL_HUE = ("#cfe0f5", "#8bb6e8", "#4a90d9", "#2a78d6", "#1a4f8f", "#0d2c52")

STATUS = {
    "good": "#1baf7a",
    "warning": "#eda100",
    "critical": "#e34948",
}

INK = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#8a8985",
    "grid": "#e3e2dd",
    "surface": "#ffffff",
}

FIGURES_DIR = Path("results/figures")


def use_paper_style() -> None:
    """Install the manuscript rcParams.  Call once per notebook."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": INK["surface"],
            "figure.facecolor": INK["surface"],
            "axes.facecolor": INK["surface"],
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 9,
            "axes.labelcolor": INK["secondary"],
            "axes.edgecolor": INK["grid"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": INK["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 1.0,
            "xtick.color": INK["muted"],
            "ytick.color": INK["muted"],
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "lines.solid_capstyle": "round",
            "text.color": INK["primary"],
            "pdf.fonttype": 42,  # embed as TrueType so editors can select text
            "ps.fonttype": 42,
        }
    )


def sequential_colors(n: int) -> list[str]:
    """``n`` evenly spaced steps of the sequential ramp, light -> dark."""
    if n <= 0:
        return []
    if n == 1:
        return [SEQUENTIAL_HUE[3]]
    cmap = LinearSegmentedColormap.from_list("tf_seq", SEQUENTIAL_HUE)
    return [mpl.colors.to_hex(cmap(x)) for x in np.linspace(0.12, 1.0, n)]


def categorical_colors(n: int) -> list[str]:
    """First ``n`` categorical slots, in fixed order.

    Refuses to cycle: past the validated slot count the caller must fold
    categories into "other" or facet instead of inventing hues.
    """
    if n > len(CATEGORICAL):
        raise ValueError(
            f"{n} categories exceeds the {len(CATEGORICAL)} validated slots; "
            "fold into 'other' or use small multiples"
        )
    return list(CATEGORICAL[:n])


def direct_label(
    ax, x: float, y: float, text: str, color: str, *, dx: float = 4.0,
    dy: float = 0.0, **kw
) -> None:
    """Label a series at its end, with a surface halo so it stays legible."""
    t = ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=8,
        fontweight="medium",
        color=color,
        **kw,
    )
    t.set_path_effects(
        [mpl.patheffects.withStroke(linewidth=2.5, foreground=INK["surface"])]
    )


def annotate_note(ax, text: str, *, loc: str = "lower right") -> None:
    """Small recessive caption inside the axes (assumptions, n, source)."""
    xy = {
        "lower right": (0.99, 0.02, "right", "bottom"),
        "lower left": (0.01, 0.02, "left", "bottom"),
        "upper left": (0.01, 0.98, "left", "top"),
        "upper right": (0.99, 0.98, "right", "top"),
    }[loc]
    ax.text(
        xy[0], xy[1], text, transform=ax.transAxes, ha=xy[2], va=xy[3],
        fontsize=7, color=INK["muted"],
    )


def shade_region(ax, y0: float, y1: float, *, color: str = STATUS["critical"], label=None):
    """Shade a requirement-violation band (e.g. lead time below the minimum)."""
    ax.axhspan(y0, y1, color=to_rgba(color, 0.08), zorder=0, lw=0)
    ax.axhline(y1, color=color, lw=1.0, ls=(0, (4, 3)), zorder=1)
    if label:
        ax.text(
            0.01, y1, f" {label}", transform=ax.get_yaxis_transform(),
            va="bottom", ha="left", fontsize=7, color=color,
        )


def save(fig, name: str, *, directory: Path | str = FIGURES_DIR,
         formats: Sequence[str] = ("pdf", "png")) -> list[Path]:
    """Write a figure as vector PDF plus a 300-dpi PNG."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in formats:
        p = d / f"{name}.{fmt}"
        fig.savefig(p, format=fmt)
        out.append(p)
    return out


def finish(fig, axes=None) -> None:
    """Tighten layout and trim tick clutter."""
    for ax in np.atleast_1d(axes if axes is not None else fig.axes).ravel():
        ax.tick_params(length=3, width=0.8)
    fig.tight_layout()


__all__ = [
    "CATEGORICAL", "SEQUENTIAL_HUE", "STATUS", "INK", "FIGURES_DIR",
    "use_paper_style", "sequential_colors", "categorical_colors",
    "direct_label", "annotate_note", "shade_region", "save", "finish",
    "plt",
]


# --------------------------------------------------------------- IEEE print

#: IEEE two-column widths, inches. Figures must be authored at final size so
#: that the point sizes below are the point sizes on the printed page.
IEEE_COL_IN = 3.5
IEEE_FULL_IN = 7.16


def use_ieee_style() -> None:
    """Styling for IEEE two-column print.

    Differs from :func:`use_paper_style` in that figures are drawn at their
    *final* physical size, so nothing is scaled down afterwards and 8 pt stays
    8 pt on the page.  Line weights and marker sizes come down accordingly;
    grids and spines are lightened because they reproduce heavier in print than
    on screen.
    """
    use_paper_style()
    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "axes.titlepad": 4,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.4,
            "lines.markersize": 3.5,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "savefig.dpi": 600,          # line art for print
            "savefig.pad_inches": 0.01,
        }
    )


def ieee_size(cols: int = 1, aspect: float = 0.72) -> tuple[float, float]:
    """Figure size in inches for ``cols`` IEEE columns at a given aspect."""
    w = IEEE_COL_IN if cols == 1 else IEEE_FULL_IN
    return (w, w * aspect)
