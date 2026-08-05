"""Public API for continuum absorption and cross-section calculations.

These functions provide the continuum absorption-side companions to the Sect. 5
emissivity formulas, for use with the UV/IR polarization discussion in Sect. 6
of `the companion VSG/AME manuscript`.
"""

from .continuum import (
    sigma_nu_P_parallel_continuum_general,
    sigma_nu_P_perp_continuum_general,
    sigma_nu_Q_perp_continuum_general,
    sigma_nu_total,
)
from .definitions import P_JK, Z

__all__ = [
    "P_JK",
    "Z",
    "sigma_nu_P_parallel_continuum_general",
    "sigma_nu_P_perp_continuum_general",
    "sigma_nu_Q_perp_continuum_general",
    "sigma_nu_total",
]
