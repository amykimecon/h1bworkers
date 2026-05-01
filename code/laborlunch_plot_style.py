"""Shared plotting style for slides_laborlunch_20260507 figures."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba


FIGSIZE = (9.2, 5.1)
WIDE_FIGSIZE = (10.2, 5.1)
PANEL_FIGSIZE = (11.2, 4.8)

LEGEND_FONT_SIZE = 10
AXIS_LABEL_FONT_SIZE = 11
TICK_FONT_SIZE = 10
BASE_FONT_SIZE = 10

MARKER_SIZE = 11
MULTI_MARKER_SIZE = 9
LINE_WIDTH = 1.5
ERRORBAR_ALPHA = 0.4

PALETTE = ("#2e8b57", "#e07a5f", "#4c78a8", "#a05195", "#ffb000", "#0072b2", "#009e73")
NEUTRAL = "#555555"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": AXIS_LABEL_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "savefig.dpi": 220,
        }
    )


def color(idx: int) -> str:
    return PALETTE[idx % len(PALETTE)]


def rgba(value: str, alpha: float = ERRORBAR_ALPHA):
    return to_rgba(value, alpha)


def marker_area(marker_size: float = MARKER_SIZE) -> float:
    return marker_size * marker_size


def offsets(n_series: int, *, span: float = 0.32) -> np.ndarray:
    if n_series <= 1:
        return np.array([0.0])
    return np.linspace(-span / 2.0, span / 2.0, num=n_series)


def offset_step(n_series: int, *, max_step: float = 0.20, target_total_span: float = 0.70) -> float:
    if n_series <= 1:
        return 0.0
    return min(max_step, target_total_span / float(n_series - 1))


def errorbar_kwargs(series_color: str, marker_size: float = MARKER_SIZE, *, alpha: float = ERRORBAR_ALPHA) -> dict:
    return {
        "ecolor": rgba(series_color, alpha),
        "elinewidth": marker_size,
        "capsize": 0,
    }


def right_legend(ax: plt.Axes, *, title: str | None = None, ncol: int = 1):
    existing = ax.get_legend()
    handles, labels = ax.get_legend_handles_labels()
    pairs = [(handle, label) for handle, label in zip(handles, labels) if label and not label.startswith("_")]
    if not pairs and existing is not None:
        handles = getattr(existing, "legend_handles", getattr(existing, "legendHandles", []))
        labels = [text.get_text() for text in existing.get_texts()]
        pairs = [(handle, label) for handle, label in zip(handles, labels) if label and not label.startswith("_")]
    if not pairs:
        return None
    if existing is not None:
        existing.remove()
    handles, labels = zip(*pairs)
    return ax.legend(
        handles,
        labels,
        title=title,
        loc="best",
        frameon=True,
        framealpha=0.86,
        facecolor="white",
        edgecolor="none",
        fontsize=LEGEND_FONT_SIZE,
        ncol=ncol,
    )


def right_figure_legend(fig: plt.Figure, handles: Sequence, labels: Sequence[str], *, title: str | None = None):
    pairs = [(handle, label) for handle, label in zip(handles, labels) if label and not label.startswith("_")]
    if not pairs:
        return None
    handles, labels = zip(*pairs)
    return fig.legend(
        handles,
        labels,
        title=title,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        framealpha=0.86,
        facecolor="white",
        edgecolor="none",
        fontsize=LEGEND_FONT_SIZE,
    )


def clear_titles(fig: plt.Figure) -> None:
    fig.suptitle("")
    for ax in fig.axes:
        ax.set_title("")


def savefig(fig: plt.Figure, path: Path, *, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clear_titles(fig)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
