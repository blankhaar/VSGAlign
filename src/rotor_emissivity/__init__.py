"""Public API for continuum spinning-dust emissivity calculations.

The exported functions evaluate the continuum approximation to the pure
rotational emissivity described in Sect. 5.1 of
`the companion VSG/AME manuscript`, using the
branch-by-branch formulas summarized in Appendix E.
"""

from .compare import compare_grid, max_rel_err, mean_rel_err
from .impl_b import (
    jnu_P_parallel,
    jnu_P_perp,
    jnu_Q_perp,
    jnu_components,
    jnu_total,
)
from .types import RotorParams

__all__ = [
    "RotorParams",
    "compare_grid",
    "jnu_P_parallel",
    "jnu_P_perp",
    "jnu_Q_perp",
    "jnu_components",
    "jnu_total",
    "max_rel_err",
    "mean_rel_err",
]
