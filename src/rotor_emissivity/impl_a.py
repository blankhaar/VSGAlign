"""Continuum emissivity implementation based on the Appendix E ``x = K/J`` form.

This module evaluates the unpolarized pure-rotational emissivity of Sect. 5.1
for an oblate symmetric-top grain. The branch-specific integrals correspond to
the continuum formulas in Appendix E of
`the companion VSG/AME manuscript`:

- ``jnu_P_parallel``: Eq. (E.9a)
- ``jnu_P_perp``: Eq. (E.12a)
- ``jnu_Q_perp``: Eq. (E.14a)
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import exp1

from .constants import c_cgs
from .partition import Z, _get_beta_gamma
from .types import RotorParams


def jnu_P_parallel(nu, p: RotorParams):
    """Compute the parallel P-branch emissivity.

    This is the continuum version of the Sect. 5.1 parallel branch, related to
    Eq. (20a) and evaluated in the Appendix E form of Eq. (E.9a).
    """
    nu = np.asarray(nu)
    scalar_input = nu.ndim == 0
    if scalar_input:
        nu = nu[None]

    beta, gamma = _get_beta_gamma(p)
    z_val = Z(p)
    const_fac = (4 * np.pi**3) / (3 * c_cgs**3)
    pop_fac = (p.n * p.mu_par**2) / z_val
    inv_2B = 1.0 / (2 * p.B)

    res = np.zeros_like(nu, dtype=float)

    for i, freq in enumerate(nu):
        if freq <= 0:
            continue

        J_nu = freq * inv_2B
        y = gamma * J_nu**2

        # Write the integral in a scaled form so the growing ``exp(y x^2)``
        # term is absorbed into the outer Boltzmann factor.
        def integrand_scaled(x):
            return (1.0 - x**2) * np.exp(-y * (1.0 - x**2))

        val_scaled, _ = quad(
            integrand_scaled,
            0.0,
            1.0,
            epsabs=p.quad_epsabs,
            epsrel=p.quad_epsrel,
        )

        res[i] = (
            const_fac
            * pop_fac
            * (freq**6 / p.B**3)
            * np.exp(-(beta - gamma) * J_nu**2)
            * val_scaled
        )

    if scalar_input:
        return res[0]
    return res


def jnu_P_perp(nu, p: RotorParams):
    """Compute the perpendicular P-branch emissivity.

    This evaluates the two ``Delta K = +/- 1`` channels of Sect. 5.1, Eq. (20b),
    in the continuum form of Appendix Eq. (E.12a).
    """
    nu = np.asarray(nu)
    scalar_input = nu.ndim == 0
    if scalar_input:
        nu = nu[None]

    beta, gamma = _get_beta_gamma(p)
    z_val = Z(p)
    delta = p.B - p.A
    const_fac = (np.pi**3) / (3 * c_cgs**3)
    pop_fac = (p.n * p.mu_perp**2) / z_val

    res = np.zeros_like(nu, dtype=float)

    for i, freq in enumerate(nu):
        if freq <= 0:
            continue

        def integrand(x):
            D_plus = p.B + delta * x
            D_minus = p.B - delta * x
            exp_factor = beta - gamma * x**2

            arg_plus = -(freq**2) * exp_factor / (4.0 * D_plus**2)
            arg_minus = -(freq**2) * exp_factor / (4.0 * D_minus**2)

            term_plus = ((1.0 - x) ** 2 / D_plus**3) * np.exp(arg_plus)
            term_minus = ((1.0 + x) ** 2 / D_minus**3) * np.exp(arg_minus)
            return term_plus + term_minus

        val_int, _ = quad(
            integrand,
            0.0,
            1.0,
            epsabs=p.quad_epsabs,
            epsrel=p.quad_epsrel,
        )
        res[i] = const_fac * pop_fac * freq**6 * val_int

    if scalar_input:
        return res[0]
    return res


def jnu_Q_perp(nu, p: RotorParams):
    """Compute the perpendicular Q-branch emissivity.

    This is the continuum version of the Sect. 5.1 Q branch, related to
    Eq. (20c) and implemented in the Appendix Eq. (E.14a) form.
    """
    nu = np.asarray(nu)
    scalar_input = nu.ndim == 0
    if scalar_input:
        nu = nu[None]

    beta, gamma = _get_beta_gamma(p)
    z_val = Z(p)
    delta = p.B - p.A
    const_fac = (4 * np.pi**3) / (3 * c_cgs**3)
    pop_fac = (p.n * p.mu_perp**2) / (delta * z_val)
    inv_2Delta = 1.0 / (2 * delta)

    res = np.zeros_like(nu, dtype=float)

    for i, freq in enumerate(nu):
        if freq <= 0:
            continue

        K_nu = freq * inv_2Delta
        z = beta * K_nu**2

        if z > 100.0:
            bracket = 1.0 / z - 2.0 / z**2 + 6.0 / z**3
        elif z > 0.0:
            bracket = 1.0 - z * np.exp(z) * exp1(z)
        else:
            bracket = 1.0

        res[i] = (
            const_fac
            * pop_fac
            * freq**4
            * (np.exp(-(beta - gamma) * K_nu**2) / beta)
            * bracket
        )

    if scalar_input:
        return res[0]
    return res


def jnu_components(nu, p: RotorParams):
    """Return the three Sect. 5.1 branch contributions separately."""
    return jnu_P_parallel(nu, p), jnu_P_perp(nu, p), jnu_Q_perp(nu, p)


def jnu_total_with_branches(
    nu,
    p: RotorParams,
    *,
    include_p_parallel: bool = True,
    include_p_perp: bool = True,
    include_q_perp: bool = True,
):
    """Sum any subset of the branch emissivities from Sect. 5.1."""
    j_par, j_perp, j_q = jnu_components(nu, p)
    total = 0.0
    if include_p_parallel:
        total = total + j_par
    if include_p_perp:
        total = total + j_perp
    if include_q_perp:
        total = total + j_q
    return total


def jnu_total_p_only(nu, p: RotorParams):
    """Return the combined P-branch emissivity only."""
    return jnu_total_with_branches(nu, p, include_q_perp=False)


def jnu_total(nu, p: RotorParams):
    """Return the total pure-rotational emissivity of Sect. 5.1, Eq. (19)."""
    return jnu_total_with_branches(nu, p, include_q_perp=True)
