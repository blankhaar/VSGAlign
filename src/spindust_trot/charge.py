"""Grain charge distributions.

DL98b (Draine & Lazarian 1998b; arXiv:astro-ph/9802239) averages several
coefficients over the grain charge distribution `f(Z_g)` (e.g. their
eqs. (19)–(20) and (38)–(41)).

This module provides a minimal representation of a discrete distribution
over integer charge states.

Within this repository the charge distribution is an input to
`spindust_trot.env.Environment`, which then feeds the DL98/AHD09 temperature
solver used by `DustGrain`. The paper scripts inherit that choice indirectly
through the `DustGrain -> params_from_dust -> emissivity/cross-section`
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ChargeDistribution",
]


@dataclass(frozen=True, slots=True)
class ChargeDistribution:
    """Discrete distribution over integer charge states Z (in units of `e`).

    Parameters
    ----------
    Z
        Integer charge states (e.g. [-1, 0, +1]).
    f
        Probabilities for each state; must be non-negative and sum to 1.
    """

    Z: NDArray[np.int_]
    f: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.Z.ndim != 1 or self.f.ndim != 1:
            raise ValueError("ChargeDistribution expects 1D arrays for Z and f.")
        if self.Z.shape != self.f.shape:
            raise ValueError("ChargeDistribution requires Z and f to have the same shape.")
        if np.any(self.f < 0):
            raise ValueError("ChargeDistribution probabilities must be non-negative.")
        total = float(np.sum(self.f))
        if not np.isfinite(total) or total <= 0:
            raise ValueError("ChargeDistribution probabilities must sum to a positive finite value.")
        # Normalize defensively (helps when passing in float32 etc).
        object.__setattr__(self, "f", self.f / total)

    @classmethod
    def delta(cls, Z0: int) -> "ChargeDistribution":
        """Return a delta-function distribution concentrated at `Z0`."""

        return cls(Z=np.array([int(Z0)], dtype=int), f=np.array([1.0], dtype=np.float64))

    def prob(self, Z0: int) -> float:
        """Return f(Z0) (0 if absent)."""

        mask = self.Z == int(Z0)
        if not np.any(mask):
            return 0.0
        return float(np.sum(self.f[mask]))

    def mean_Z2(self) -> float:
        """Return ⟨Z^2⟩."""

        return float(np.sum((self.Z.astype(np.float64) ** 2) * self.f))
