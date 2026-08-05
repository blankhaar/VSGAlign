"""Public API for polarized spinning-dust emissivity calculations.

The exported functions evaluate the alignment-sensitive emissivity formulas
from Sect. 5 of `the companion VSG/AME manuscript`,
using the continuum approximations collected in Appendix E.
"""

from .compare import compare_continuum_impls, compare_discrete_vs_continuum
from .continuum_impl_b import (
    jnuq_P_parallel,
    jnuq_P_perp,
    jnuq_Q_perp,
    jnuq_total,
)
from .dust_bridge import params_from_dust
from .types import RotorPolParams

__all__ = [
    "RotorPolParams",
    "compare_continuum_impls",
    "compare_discrete_vs_continuum",
    "jnuq_P_parallel",
    "jnuq_P_perp",
    "jnuq_Q_perp",
    "jnuq_total",
    "params_from_dust",
]
