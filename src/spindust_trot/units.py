"""Small unit helpers.

DL98b frequently expresses sizes and temperatures in dimensionless scalings,
e.g. `a_-7 = a / 1e-7 cm` and `T_2 = T / 100 K`. We keep helpers here so
coefficient implementations can remain readable. These helpers only serve the
DL98/AHD09 temperature backend; the rest of the repository consumes the final
physical outputs through `DustGrain`.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "as_ndarray",
    "a_minus7",
    "T2",
]

T = TypeVar("T")


def as_ndarray(x: ArrayLike) -> NDArray[np.float64]:
    """Convert scalar/array-like input to a float64 ndarray without copying when possible."""

    return np.asarray(x, dtype=np.float64)


def a_minus7(a_cm: ArrayLike) -> NDArray[np.float64]:
    """Return `a / 1e-7 cm` (DL98b notation `a_-7`)."""

    return as_ndarray(a_cm) / 1.0e-7


def T2(T_K: ArrayLike) -> NDArray[np.float64]:
    """Return `T / 100 K` (DL98b notation `T_2`)."""

    return as_ndarray(T_K) / 100.0
