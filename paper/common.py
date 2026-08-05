"""Shared helpers for paper-ready figure generation.

This module contains the common plotting style and output-directory handling
used by the publication scripts in `paper/`. It does not perform any physics
computation itself; instead it keeps the manuscript-facing scripts visually
consistent while the actual calculations live under `src/`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")


GHZ = 1.0e9
DEFAULT_FIGURE_DIR = Path(__file__).resolve().parent / "figures"

_BASE_RCPARAMS = {
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "text.usetex": False,
    "mathtext.fontset": "cm",
}


def configure_matplotlib(**overrides: object) -> None:
    """Apply the default publication plotting style for manuscript figures."""
    params = dict(_BASE_RCPARAMS)
    params.update(overrides)
    matplotlib.rcParams.update(params)


def ensure_figure_dir(output_dir: str | Path | None = None) -> Path:
    """Return the output directory used for generated paper figures."""
    figure_dir = DEFAULT_FIGURE_DIR if output_dir is None else Path(output_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir
