"""Discrete polarized line list for the Sect. 5.1 rotational branches.

This module keeps the explicit line-by-line version of the polarized
pure-rotational emissivity from Sect. 5.1, especially Eq. (17b),
Eq. (21), and Eqs. (22a)-(22c), using the two-temperature population model
from Eq. (4). It serves as the reference model against which the continuum
approximation in Appendix E is validated.
"""

from typing import Dict

import numpy as np

from discrete_emissivity.rotor_lines import _energy, calc_partition_sum, estimate_sufficient_Jmax
from rotor_emissivity.constants import c_cgs, h_cgs
from rotor_emissivity.partition import _get_beta_gamma

from .constants import calc_P_chi, w_P, w_Q
from .types import RotorPolParams

LINE_DTYPE_POL = [('nu', 'f8'), ('weight_q', 'f8'), ('branch', 'U10')]


def polarized_line_list(p: RotorPolParams, Jmax: int = None) -> np.ndarray:
    """Build the discrete polarized line list of Sect. 5.1.

    The returned lines resolve the branch structure of Eqs. (22a)-(22c) before
    any continuum approximation is taken.
    """
    if Jmax is None:
        Jmax = estimate_sufficient_Jmax(p)

    beta, gamma = _get_beta_gamma(p)
    Z_val = calc_partition_sum(p, Jmax)
    P_chi = calc_P_chi(p.chi)

    A_pre_const = (64 * np.pi**4) / (3 * h_cgs * c_cgs**3)
    h_over_4pi = h_cgs / (4 * np.pi)
    B_val = p.B
    A_val = p.A
    lines = []

    for J in range(0, Jmax + 1):
        pop_J = (p.n / Z_val) * (2 * J + 1) * np.exp(-beta * J * (J + 1))

        for K in range(0, J + 1):
            pop_JK = pop_J * np.exp(gamma * K**2)
            g_K = 1.0 if K == 0 else 2.0
            current_n = pop_JK * g_K
            sig = p.get_sigma(J, K)
            base_fac = -current_n * sig * P_chi * h_over_4pi

            def add_line(nu, A_val_coeff, w_branch, label):
                if nu <= 0:
                    return
                lines.append((nu, base_fac * nu * w_branch * A_val_coeff, label))

            if J >= 1 and K <= (J - 1):
                nu_P_par = _energy(J, K, B_val, A_val) - _energy(J - 1, K, B_val, A_val)
                if nu_P_par > 0:
                    A_P_par = (
                        A_pre_const
                        * nu_P_par**3
                        * (p.mu_par**2 / 2.0)
                        * (1.0 - (K / J) ** 2)
                    )
                    add_line(nu_P_par, A_P_par, w_P, "P_parallel")

            if J >= 1 and (K + 1) <= (J - 1):
                nu_P_plus = _energy(J, K, B_val, A_val) - _energy(J - 1, K + 1, B_val, A_val)
                if nu_P_plus > 0:
                    A_P_plus = (
                        A_pre_const
                        * nu_P_plus**3
                        * (p.mu_perp**2 / 8.0)
                        * (1.0 - K / J) ** 2
                    )
                    add_line(nu_P_plus, A_P_plus, w_P, "P_perp")

            if J >= 1:
                K_minus = K - 1
                if abs(K_minus) <= (J - 1):
                    nu_P_minus = _energy(J, K, B_val, A_val) - _energy(
                        J - 1, abs(K_minus), B_val, A_val
                    )
                    if nu_P_minus > 0:
                        A_P_minus = (
                            A_pre_const
                            * nu_P_minus**3
                            * (p.mu_perp**2 / 8.0)
                            * (1.0 + K / J) ** 2
                        )
                        add_line(nu_P_minus, A_P_minus, w_P, "P_perp")

            if (K + 1) <= J:
                nu_Q = _energy(J, K, B_val, A_val) - _energy(J, K + 1, B_val, A_val)
                if nu_Q > 0:
                    A_Q = (
                        A_pre_const
                        * nu_Q**3
                        * (p.mu_perp**2 / 4.0)
                        * (1.0 - (K / J) ** 2)
                    )
                    add_line(nu_Q, A_Q, w_Q, "Q_perp")

            if K >= 1:
                nu_Q_minus = _energy(J, K, B_val, A_val) - _energy(J, K - 1, B_val, A_val)
                if nu_Q_minus > 0:
                    A_Q_minus = (
                        A_pre_const
                        * nu_Q_minus**3
                        * (p.mu_perp**2 / 4.0)
                        * (1.0 - (K / J) ** 2)
                    )
                    add_line(nu_Q_minus, A_Q_minus, w_Q, "Q_perp")

    if not lines:
        return np.zeros(0, dtype=LINE_DTYPE_POL)

    return np.array(lines, dtype=LINE_DTYPE_POL)


def spectrum_binned_pol(line_data: np.ndarray, nu_grid: np.ndarray) -> Dict[str, np.ndarray]:
    """Bin the polarized line list onto a frequency grid."""
    if len(nu_grid) < 2:
        raise ValueError("Grid too small")

    edges = np.zeros(len(nu_grid) + 1)
    edges[0] = nu_grid[0] - (nu_grid[1] - nu_grid[0]) / 2
    edges[-1] = nu_grid[-1] + (nu_grid[-1] - nu_grid[-2]) / 2
    edges[1:-1] = (nu_grid[1:] + nu_grid[:-1]) / 2
    dnu = np.diff(edges)

    results = {
        'total': np.zeros_like(nu_grid),
        'P_parallel': np.zeros_like(nu_grid),
        'P_perp': np.zeros_like(nu_grid),
        'Q_perp': np.zeros_like(nu_grid),
    }

    for branch in np.unique(line_data['branch']):
        subset = line_data[line_data['branch'] == branch]
        hist, _ = np.histogram(subset['nu'], bins=edges, weights=subset['weight_q'])
        j_nu = hist / dnu
        if branch in results:
            results[branch] += j_nu
        results['total'] += j_nu

    return results
