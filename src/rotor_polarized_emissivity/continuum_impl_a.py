"""Polarized continuum emissivity using the Appendix E ``x = K/J`` form.

This module evaluates the polarized pure-rotational emissivity of Sect. 5.1
for aligned grains. The branch-specific continuum formulas correspond to
Appendix E of `the companion VSG/AME manuscript`:

- ``jnuq_P_parallel``: Eq. (E.9b)
- ``jnuq_P_perp``: Eq. (E.12b)
- ``jnuq_Q_perp``: Eq. (E.14b)
"""

import numpy as np
from scipy.integrate import quad

from rotor_emissivity.partition import Z, _get_beta_gamma

from .constants import calc_P_chi, w_P, w_Q, c_cgs
from .types import RotorPolParams


def _common_precalc(p: RotorPolParams):
    beta, gamma = _get_beta_gamma(p)
    return beta, gamma, Z(p), calc_P_chi(p.chi)


def jnuq_P_parallel(nu, p: RotorPolParams, impl="a", mode="continuum"):
    """Compute the polarized parallel P branch.

    This is the Sect. 5.1 parallel polarized emissivity, Eq. (22a), evaluated
    in the continuum form of Appendix Eq. (E.9b).
    """
    nu = np.asarray(nu)
    scalar_input = nu.ndim == 0
    if scalar_input:
        nu = nu[None]

    beta, gamma, z_val, P_chi = _common_precalc(p)
    const_term = (
        -P_chi
        * w_P
        * (16 * np.pi**3)
        / (3 * c_cgs**3)
        * (p.mu_par**2 / 2.0)
        * (1.0 / p.B)
    )
    n_over_Z = p.n / z_val

    res = np.zeros_like(nu, dtype=float)

    for i, freq in enumerate(nu):
        if freq <= 0:
            continue

        J_nu = freq / (2 * p.B)
        pre_fac = (
            const_term
            * freq**4
            * J_nu
            * (n_over_Z * 2 * J_nu)
            * np.exp(-beta * J_nu**2)
        )

        def integrand(x):
            K_val = x * J_nu
            return p.get_sigma(J_nu, K_val) * (1.0 - x**2) * np.exp(gamma * K_val**2)

        val, _ = quad(
            integrand,
            0.0,
            1.0,
            epsabs=p.quad_epsabs,
            epsrel=p.quad_epsrel,
        )
        res[i] = pre_fac * val

    if scalar_input:
        return res[0]
    return res


def jnuq_P_perp(nu, p: RotorPolParams, impl="a", mode="continuum"):
    """Compute the polarized perpendicular P branch.

    This evaluates the two ``Delta K = +/- 1`` channels in Sect. 5.1, Eq. (22b),
    using the continuum form of Appendix Eq. (E.12b).
    """
    nu = np.asarray(nu)
    scalar_input = nu.ndim == 0
    if scalar_input:
        nu = nu[None]

    beta, gamma, z_val, P_chi = _common_precalc(p)
    delta = p.B - p.A
    const_term = (
        -P_chi
        * w_P
        * (16 * np.pi**3)
        / (3 * c_cgs**3)
        * (p.mu_perp**2 / 8.0)
        * (1.0 / p.B)
    )
    n_over_Z = p.n / z_val

    res = np.zeros_like(nu, dtype=float)

    for i, freq in enumerate(nu):
        if freq <= 0:
            continue

        K_max_plus = freq / (2 * (p.B + delta))
        K_max_minus = freq / (2 * p.A)

        def compute_channel(sign_fac, K_max):
            def integrand(x):
                K_val = K_max * x
                if sign_fac == -1:
                    J_val = (freq - 2 * delta * K_val) / (2 * p.B)
                    honl = (1.0 - K_val / J_val) ** 2
                else:
                    J_val = (freq + 2 * delta * K_val) / (2 * p.B)
                    honl = (1.0 + K_val / J_val) ** 2

                pop_term = (2 * J_val) * np.exp(-beta * J_val**2 + gamma * K_val**2)
                return p.get_sigma(J_val, K_val) * pop_term * honl

            val, _ = quad(
                integrand,
                0.0,
                1.0,
                epsabs=p.quad_epsabs,
                epsrel=p.quad_epsrel,
            )
            return val * K_max

        total_val = compute_channel(-1, K_max_plus) + compute_channel(1, K_max_minus)
        res[i] = const_term * freq**4 * n_over_Z * total_val

    if scalar_input:
        return res[0]
    return res


def jnuq_Q_perp(nu, p: RotorPolParams, impl="a", mode="continuum"):
    """Compute the polarized perpendicular Q branch.

    This is the Sect. 5.1 Q-branch polarized emissivity, Eq. (22c), written in
    the continuum form of Appendix Eq. (E.14b).
    """
    nu = np.asarray(nu)
    scalar_input = nu.ndim == 0
    if scalar_input:
        nu = nu[None]

    beta, gamma, z_val, P_chi = _common_precalc(p)
    delta = p.B - p.A
    const_term = (
        -P_chi
        * w_Q
        * (16 * np.pi**3)
        / (3 * c_cgs**3)
        * (p.mu_perp**2 / 4.0)
        * (1.0 / delta)
    )
    n_over_Z = p.n / z_val

    res = np.zeros_like(nu, dtype=float)

    for i, freq in enumerate(nu):
        if freq <= 0:
            continue

        K_nu = freq / (2 * delta)
        pre_fac = const_term * freq**4 * n_over_Z

        sqrt_beta = np.sqrt(beta)
        u_min = sqrt_beta * K_nu
        u_max = np.sqrt(u_min**2 + 40.0)

        def integrand(u):
            J = u / sqrt_beta
            exp_val = -beta * J**2 + gamma * K_nu**2
            pop_J = 2 * J * np.exp(exp_val)
            sig = p.get_sigma(J, K_nu)
            honl = 1.0 - (K_nu / J) ** 2
            return pop_J * sig * honl / sqrt_beta

        val, _ = quad(
            integrand,
            u_min,
            u_max,
            epsabs=p.quad_epsabs,
            epsrel=p.quad_epsrel,
        )
        res[i] = pre_fac * val

    if scalar_input:
        return res[0]
    return res


def jnuq_total(nu, p: RotorPolParams, impl="a", mode="continuum"):
    """Return the full polarized emissivity of Sect. 5.1, Eq. (21)."""
    return jnuq_P_parallel(nu, p) + jnuq_P_perp(nu, p) + jnuq_Q_perp(nu, p)
