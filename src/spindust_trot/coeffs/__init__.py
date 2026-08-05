"""Coefficient backends.

Backends provide the dimensionless damping/excitation factors F and G
and (optionally) modified physics choices.

Implemented backends:

- `dl98`: Draine & Lazarian (1998b) baseline.
- `ahd09`: Applies the AHD09 factor-of-2 correction to IR damping.

These backends are selected by `DustGrain.compute_rotational_temperature`.
They therefore sit on the paper-production path even though the figure scripts
never call them directly.
"""

from __future__ import annotations

from typing import Literal

CoeffVersion = Literal["dl98", "ahd09"]

__all__ = [
    "CoeffVersion",
]
