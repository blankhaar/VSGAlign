"""Physical constants in cgs-Gaussian units.

All of DL98b's (Draine & Lazarian 1998b; arXiv:astro-ph/9802239) formulas
used here are most naturally evaluated in cgs-Gaussian:

- lengths in cm
- masses in g
- energies in erg
- temperatures in K
- time in s
- charge in statcoulomb (esu), so that e^2 has units erg*cm

We keep these constants in one place to reduce mistakes and make unit
assumptions explicit. The same unit system carries through the `DustGrain`
bridge and into the paper-ready emissivity and cross-section calculations.
"""

from __future__ import annotations

import math

__all__ = [
    "pi",
    "k_B",
    "c",
    "h",
    "hbar",
    "e_esu",
    "m_u",
    "m_H",
    "m_e",
    "debye_esu_cm",
    "eV_to_erg",
    "amu_to_g",
]

pi = math.pi

# Exact SI values converted to cgs where needed.
k_B = 1.380649e-16  # erg / K
c = 2.99792458e10  # cm / s
h = 6.62607015e-27  # erg*s
hbar = 1.054571817e-27  # erg*s

# Elementary charge in Gaussian-cgs (statcoulomb / esu).
e_esu = 4.803204712570263e-10  # statC

m_u = 1.66053906660e-24  # g (atomic mass unit)
m_H = 1.6735575e-24  # g (hydrogen atom mass used in DL98-style normalizations)
m_e = 9.1093837015e-28  # g

# Dipole-moment unit conversion: 1 Debye = 1e-18 statC*cm
debye_esu_cm = 1.0e-18

eV_to_erg = 1.602176634e-12  # erg / eV


def amu_to_g(mass_amu: float) -> float:
    """Convert atomic mass units to grams."""

    return mass_amu * m_u
