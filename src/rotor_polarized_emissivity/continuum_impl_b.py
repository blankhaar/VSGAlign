"""Alternative polarized continuum evaluation for the Appendix E formulas.

Implementation B computes the same polarized pure-rotational emissivity as
`rotor_polarized_emissivity.continuum_impl_a`, but with direct ``J(K)`` or
``K`` integrations rather than the scaled ``x = K/J`` representation.
"""

import numpy as np
from scipy.integrate import quad

from rotor_emissivity.partition import Z, _get_beta_gamma

from .constants import calc_P_chi, w_P, w_Q, c_cgs
from .types import RotorPolParams


def _common_precalc(p: RotorPolParams):
    beta, gamma = _get_beta_gamma(p)
    return beta, gamma, Z(p), calc_P_chi(p.chi)


def jnuq_P_parallel(nu, p: RotorPolParams, impl="b", mode="continuum"):
    """Compute the polarized parallel P branch.

    This is the Sect. 5.1 parallel polarized emissivity, Eq. (22a), written in
    the continuum form of Appendix Eq. (E.9b).
    """
    nu = np.asarray(nu)
    scalar_input = nu.ndim == 0
    if scalar_input:
        nu = nu[None]

    beta, gamma, z_val, P_chi = _common_precalc(p)
    if np.isinf(z_val):
        return np.zeros_like(nu, dtype=float)

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

        def integrand(K):
            sig = p.get_sigma(J_nu, K)
            honl = 1.0 - (K / J_nu) ** 2
            pop = 2 * J_nu * np.exp(-beta * J_nu**2 + gamma * K**2)
            return sig * honl * pop

        val, _ = quad(
            integrand,
            0.0,
            J_nu,
            epsabs=p.quad_epsabs,
            epsrel=p.quad_epsrel,
        )
        res[i] = const_term * freq**4 * n_over_Z * val

    if scalar_input:
        return res[0]
    return res


def jnuq_P_perp(nu, p: RotorPolParams, impl="b", mode="continuum"):
    """Compute the polarized perpendicular P branch.

    This evaluates the two ``Delta K = +/- 1`` channels in Sect. 5.1, Eq. (22b),
    using the continuum form of Appendix Eq. (E.12b).
    """
    nu = np.asarray(nu)
    scalar_input = nu.ndim == 0
    if scalar_input:
        nu = nu[None]

    beta, gamma, z_val, P_chi = _common_precalc(p)
    if np.isinf(z_val):
        return np.zeros_like(nu, dtype=float)

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

        def integrand_plus(K):
            J_val = (freq - 2 * delta * K) / (2 * p.B)
            pop = 2 * J_val * np.exp(-beta * J_val**2 + gamma * K**2)
            sig = p.get_sigma(J_val, K)
            honl = (1.0 - K / J_val) ** 2
            return sig * pop * honl

        def integrand_minus(K):
            J_val = (freq + 2 * delta * K) / (2 * p.B)
            pop = 2 * J_val * np.exp(-beta * J_val**2 + gamma * K**2)
            sig = p.get_sigma(J_val, K)
            honl = (1.0 + K / J_val) ** 2
            return sig * pop * honl

        val_plus, _ = quad(
            integrand_plus,
            0.0,
            K_max_plus,
            epsabs=p.quad_epsabs,
            epsrel=p.quad_epsrel,
        )
        val_minus, _ = quad(
            integrand_minus,
            0.0,
            K_max_minus,
            epsabs=p.quad_epsabs,
            epsrel=p.quad_epsrel,
        )
        res[i] = const_term * freq**4 * n_over_Z * (val_plus + val_minus)

    if scalar_input:
        return res[0]
    return res


def jnuq_Q_perp(nu, p: RotorPolParams, impl="b", mode="continuum"):
    """Compute the polarized perpendicular Q branch.

    This is the Sect. 5.1 Q-branch polarized emissivity, Eq. (22c), written in
    the continuum form of Appendix Eq. (E.14b).
    """
    nu = np.asarray(nu)
    scalar_input = nu.ndim == 0
    if scalar_input:
        nu = nu[None]

    beta, gamma, z_val, P_chi = _common_precalc(p)
    if np.isinf(z_val):
        return np.zeros_like(nu, dtype=float)

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

        sqrt_beta = np.sqrt(beta)
        u_min = sqrt_beta * K_nu
        # The population falls as exp(-u^2). Truncating after an additional
        # exponent of 40 discards less than exp(-40) of the local Gaussian
        # scale and avoids the fragile infinite-interval transformation used
        # by QUADPACK.
        u_max = np.sqrt(u_min**2 + 40.0)

        def integrand(u):
            J = u / sqrt_beta
            pop = 2 * J * np.exp(-beta * J**2 + gamma * K_nu**2)
            sig = p.get_sigma(J, K_nu)
            honl = 1.0 - (K_nu / J) ** 2
            return sig * pop * honl / sqrt_beta

        val, _ = quad(
            integrand,
            u_min,
            u_max,
            epsabs=p.quad_epsabs,
            epsrel=p.quad_epsrel,
        )
        res[i] = const_term * freq**4 * n_over_Z * val

    if scalar_input:
        return res[0]
    return res


def jnuq_total(nu, p: RotorPolParams, impl="b", mode="continuum"):
    """Return the full polarized emissivity of Sect. 5.1, Eq. (21)."""
    return jnuq_P_parallel(nu, p) + jnuq_P_perp(nu, p) + jnuq_Q_perp(nu, p)
