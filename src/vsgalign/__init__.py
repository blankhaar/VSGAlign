"""Convenience facade for the VSGAlign public API.

VSGAlign is the public project and distribution name for this repository.
The scientific implementation remains split across descriptive subpackages such
as `dust_properties`, `rotor_emissivity`, and `rotor_cross_section`, while this
module re-exports the most common user-facing entry points in one place.
"""

from __future__ import annotations

from dust_emissivity import rotor_params_from_dust
from dust_properties import DustGrain
from rotor_cross_section import sigma_nu_total
from rotor_emissivity import jnu_total
from rotor_polarized_emissivity import (
    jnuq_total,
    params_from_dust as polarized_params_from_dust,
)

__all__ = [
    "DustGrain",
    "jnu_total",
    "jnuq_total",
    "polarized_params_from_dust",
    "rotor_params_from_dust",
    "sigma_nu_total",
]
