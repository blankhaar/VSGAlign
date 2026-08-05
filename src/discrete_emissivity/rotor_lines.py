"""Discrete rotational line builder for Sect. 5.1 of the paper.

This module keeps the explicit line-by-line version of the pure rotational
emissivity discussed in Sect. 5.1 of
`the companion VSG/AME manuscript`. The level
populations follow the two-temperature model of Eq. (4), while the emitted
branches follow Eqs. (19)-(22). The continuum modules in `rotor_emissivity`
approximate the same physics via Appendix E.
"""

import numpy as np

from rotor_emissivity.constants import c_cgs, h_cgs
from rotor_emissivity.partition import Z_numeric, _get_beta_gamma
from rotor_emissivity.types import RotorParams

LINE_DTYPE = [('nu', 'f8'), ('weight', 'f8'), ('branch', 'U10')]


def _energy(J, K, B, A):
    """Return ``E/h`` for the symmetric-top rotor of Sect. 2.1, Eq. (1a)."""
    return B * J * (J + 1) - (B - A) * K**2


def estimate_sufficient_Jmax(p: RotorParams, epsilon: float = 1e-9) -> int:
    """Estimate a discrete cutoff that captures the thermal emissivity tail."""
    beta, _ = _get_beta_gamma(p)
    if beta <= 0:
        return 1000

    target_exponent = 25.0
    Jmax = np.sqrt(target_exponent / beta)
    return int(np.ceil(Jmax))


def calc_partition_sum(p: RotorParams, Jmax: int = None) -> float:
    """Compute the discrete partition sum used by the line populations."""
    if Jmax is None:
        Jmax = estimate_sufficient_Jmax(p)
    return Z_numeric(p, Jmax=Jmax)


def line_list(p: RotorParams, Jmax: int = None, branches: str = "all") -> np.ndarray:
    """Generate the discrete rotational line list for Sect. 5.1.

    Parameters
    ----------
    p
        Rotor parameters defining Eq. (4) and the branch strengths.
    Jmax
        Truncation of the upper-state sum. If omitted, an adaptive estimate is
        used.
    branches
        Either ``"all"`` or one of ``"P_parallel"``, ``"P_perp"``, or
        ``"Q_perp"``.
    """
    if Jmax is None:
        Jmax = estimate_sufficient_Jmax(p)

    valid_branches = {"all", "P_parallel", "P_perp", "Q_perp"}
    if branches not in valid_branches:
        raise ValueError(f"Unknown branch selector: {branches}")

    beta, gamma = _get_beta_gamma(p)
    Z_val = calc_partition_sum(p, Jmax)

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

            def add_line(nu, A_val_coeff, label):
                if nu <= 0:
                    return
                if branches != "all" and label != branches:
                    return
                weight = (h_over_4pi * nu) * current_n * A_val_coeff
                lines.append((nu, weight, label))

            if J >= 1 and K <= (J - 1):
                nu_P_par = _energy(J, K, B_val, A_val) - _energy(J - 1, K, B_val, A_val)
                if nu_P_par > 0:
                    A_P_par = (
                        A_pre_const
                        * nu_P_par**3
                        * (p.mu_par**2 / 2.0)
                        * (1.0 - (K / J) ** 2)
                    )
                    add_line(nu_P_par, A_P_par, "P_parallel")

            if J >= 1 and (K + 1) <= (J - 1):
                nu_P_plus = _energy(J, K, B_val, A_val) - _energy(J - 1, K + 1, B_val, A_val)
                if nu_P_plus > 0:
                    A_P_plus = (
                        A_pre_const
                        * nu_P_plus**3
                        * (p.mu_perp**2 / 8.0)
                        * (1.0 - K / J) ** 2
                    )
                    add_line(nu_P_plus, A_P_plus, "P_perp")

            # The ``abs(K - 1)`` target folds the symmetric ``K = 0 -> -1``
            # transition into the positive-``K`` bookkeeping without iterating
            # signed ``K`` explicitly.
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
                        add_line(nu_P_minus, A_P_minus, "P_perp")

            # For oblate grains with ``B > A``, the Q branch emits toward larger
            # ``|K|``, so only the ``K -> K + 1`` path appears in the unpolarized
            # line list.
            if (K + 1) <= J:
                nu_Q = _energy(J, K, B_val, A_val) - _energy(J, K + 1, B_val, A_val)
                if nu_Q > 0:
                    A_Q = (
                        A_pre_const
                        * nu_Q**3
                        * (p.mu_perp**2 / 4.0)
                        * (1.0 - (K / J) ** 2)
                    )
                    add_line(nu_Q, A_Q, "Q_perp")

    if not lines:
        return np.zeros(0, dtype=LINE_DTYPE)

    return np.array(lines, dtype=LINE_DTYPE)
