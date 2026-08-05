"""Helpers for connecting dust-grain models to emissivity calculations.

This package bridges the Sect. 2 grain description and the Sect. 5 rotational
emissivity formulas in
`the companion VSG/AME manuscript`.
"""

from .adapter import rotor_params_from_dust, validate_mapping
from .simulate import jnu_dust, simulate_over_sizes

__all__ = [
    "jnu_dust",
    "rotor_params_from_dust",
    "simulate_over_sizes",
    "validate_mapping",
]
