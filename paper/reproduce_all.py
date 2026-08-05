"""Reproduce the VSGAlign figures used by the final manuscript.

This module is the top-level entry point for the paper workflow. It delegates
the physics to the individual figure scripts, which in turn call the public
APIs under `src/`. Cross-section plots are retained as optional diagnostics;
they are not included in the final 2026-08-04 revision-3 submission.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import ensure_figure_dir
from .plot_cnm_cross_sections import main as plot_cnm_cross_sections
from .plot_cross_sections import main as plot_cross_sections
from .plot_single_grain_overview import main as plot_single_grain_overview
from .plot_size_distribution import main as plot_size_distribution


def main(
    output_dir: str | Path | None = None,
    *,
    include_diagnostics: bool = False,
):
    """Generate the paper figures and, optionally, diagnostic cross sections."""
    figure_dir = ensure_figure_dir(output_dir)
    outputs = []
    outputs.extend(plot_single_grain_overview(output_dir=figure_dir))
    outputs.extend(plot_size_distribution(output_dir=figure_dir))
    if include_diagnostics:
        outputs.extend(plot_cross_sections(output_dir=figure_dir))
        outputs.append(plot_cnm_cross_sections(output_dir=figure_dir))
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce all paper figures.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated paper figures. Defaults to paper/figures.",
    )
    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="also generate cross-section plots not used in the final manuscript",
    )
    args = parser.parse_args()
    generated = main(
        output_dir=args.output_dir,
        include_diagnostics=args.include_diagnostics,
    )
    for path in generated:
        print(path)
