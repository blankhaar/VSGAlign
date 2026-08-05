"""Utilities for evaluating alignment moments used by polarized emissivity.

The cached `SigmaComputer` supports the rank-2 alignment moment needed in
Sect. 4 and then consumed by the polarized emissivity formulas in Sect. 5 of
`the companion VSG/AME manuscript`.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

from .constants import h_cgs as h, k_B_cgs as k_B


class SigmaComputer:
    """Cache the alignment moment ``sigma(J, K)`` on a discrete ``J`` grid.

    Integer ``J`` values retain the exact discrete moment used by the line-list
    reference calculation. Continuum quadratures evaluate the same moment via
    a shape-preserving interpolation between integer values. This avoids
    introducing artificial step discontinuities or interpolation overshoot
    into otherwise smooth integrands.
    """

    def __init__(self, x_ratio: float, y_param: float, B: float, A: float, T_int: float):
        self.x_ratio = x_ratio
        self.y_param = y_param
        self.B = B
        self.A = A
        self.T_int = T_int
        
        self.fac = 1.0 / (1.0 + x_ratio)
        #BL: Corrected the overall alignment sign to match the revised formalism. 04/08/26
        self.b_val = y_param * self.fac
        self.gamma = (B - A) * h / (k_B * T_int)
        
        self.S_cache = None
        self.max_J_cached = 0
        self._scaled_weight_sum = 1.0
        self._scaled_k2_sum = 0.0
        self._S_interpolator = None

    def precompute_moments(self, J_max: int):
        """Precompute the exact discrete moment vector up to ``J_max``."""
        J_max = int(J_max)
        if J_max <= 0:
            return

        if self.S_cache is None:
            self.S_cache = np.zeros(J_max + 1)
        elif J_max > self.max_J_cached:
            new_cache = np.zeros(J_max + 1)
            new_cache[: self.max_J_cached + 1] = self.S_cache
            self.S_cache = new_cache
        else:
            return

        start_J = max(1, self.max_J_cached + 1)

        denominator = self._scaled_weight_sum
        k2_numerator = self._scaled_k2_sum

        for J in range(start_J, J_max + 1):
            # Keep both cumulative sums scaled by exp(-gamma * J**2). This
            # recurrence is algebraically identical to summing all K <= J,
            # but extends the cache in O(J_max) rather than O(J_max**2).
            rescale = np.exp(-self.gamma * (2.0 * J - 1.0))
            denominator = rescale * denominator + 2.0
            k2_numerator = rescale * k2_numerator + 2.0 * J**2
            mean_k2_over_j2 = k2_numerator / (denominator * J**2)
            self.S_cache[J] = self.y_param * 0.5 * (3.0 * mean_k2_over_j2 - 1.0)

        self._scaled_weight_sum = denominator
        self._scaled_k2_sum = k2_numerator
        self.max_J_cached = J_max

        grid = np.arange(self.max_J_cached + 1, dtype=float)
        self._S_interpolator = PchipInterpolator(grid, self.S_cache, extrapolate=True)

    def _mean_alignment(self, J: float) -> float:
        """Return the cached K-averaged alignment at integer or continuous J."""
        if J <= 0:
            return 0.0

        upper_J = int(np.ceil(J))
        if upper_J > self.max_J_cached:
            self.precompute_moments(upper_J + 100)

        integer_J = int(J)
        if float(J).is_integer():
            return float(self.S_cache[integer_J])

        return float(self._S_interpolator(float(J)))

    def coefficients(self, J):
        """Return the affine coefficients used by the cached sigma model."""
        if J <= 0:
            return 0.0, 0.0

        S_val = self._mean_alignment(float(J))
        return -S_val * self.fac, self.b_val

    def __call__(self, J, K):
        """Evaluate ``sigma(J, K)`` with vectorized support in ``K``."""
        if J <= 0:
            return 0.0

        S_val = self._mean_alignment(float(J))

        K_arr = np.atleast_1d(K)
        x = np.divide(K_arr, J, out=np.zeros_like(K_arr, dtype=float), where=J!=0)
        P2 = 0.5 * (3 * x**2 - 1)

        term1 = self.b_val * P2
        term2 = -S_val * self.fac

        res = term1 + term2

        if np.ndim(K) == 0:
            return res.item()
        return res
