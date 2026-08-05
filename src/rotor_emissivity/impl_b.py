"""Alternative continuum emissivity evaluation for the Appendix E formulas.

Implementation B evaluates the same Sect. 5.1 branch emissivities as
`rotor_emissivity.impl_a`, but with a different choice of integration
variables. The underlying paper mapping is unchanged:

- ``jnu_P_parallel``: Appendix Eq. (E.9a)
- ``jnu_P_perp``: Appendix Eq. (E.12a)
- ``jnu_Q_perp``: Appendix Eq. (E.14a)
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import dawsn, exp1

from .constants import c_cgs, h_cgs, k_B_cgs
from .partition import Z
from .types import RotorParams


def _get_alpha_gamma(p: RotorParams):
    """Return the dimensionless parameters used in the implementation-B notes."""
    alpha = h_cgs * p.B / (k_B_cgs * p.T_rot)
    delta = p.B - p.A
    gamma = h_cgs * delta / (k_B_cgs * p.T_int)
    return alpha, gamma


def _R_nu(nu):
    """Return the common radiative prefactor ``16 pi^3 nu^4 / (3 c^3)``."""
    return (16 * np.pi**3 * nu**4) / (3 * c_cgs**3)


def _F_parallel(s):
    """Evaluate the auxiliary function for the parallel P branch."""
    if s < 1e-4:
        return 2.0 / 3.0 - 2.0 / 15.0 * s

    sqrt_s = np.sqrt(s)
    D_val = dawsn(sqrt_s)
    return (2 * s + 1) / (2 * s * sqrt_s) * D_val - 1.0 / (2 * s)


def jnu_P_parallel(nu, p: RotorParams):
    """Compute the parallel P-branch emissivity.

    This branch corresponds to Sect. 5.1, Eq. (20a), written in the analytic
    continuum form of Appendix Eq. (E.9a).
    """
    nu = np.asarray(nu)
    scalar_input = nu.ndim == 0
    if scalar_input:
        nu = nu[None]

    alpha, gamma = _get_alpha_gamma(p)
    z_val = Z(p)
    if np.isinf(z_val):
        return np.zeros_like(nu, dtype=float)

    inv_2B = 1.0 / (2 * p.B)
    n_over_Z = p.n / z_val
    mu_fac = p.mu_par**2 / 2.0

    res = np.zeros_like(nu, dtype=float)

    for i, freq in enumerate(nu):
        if freq <= 0:
            continue

        J_nu = freq * inv_2B
        s_nu = gamma * J_nu**2
        res[i] = (
            _R_nu(freq)
            * mu_fac
            * n_over_Z
            * (2 * J_nu**2 / p.B)
            * np.exp(-(alpha - gamma) * J_nu**2)
            * _F_parallel(s_nu)
        )

    if scalar_input:
        return res[0]
    return res


def jnu_P_perp(nu, p: RotorParams):
    """Compute the perpendicular P-branch emissivity.

    This evaluates the ``Delta K = +/- 1`` channels of Sect. 5.1, Eq. (20b),
    using the continuum representation of Appendix Eq. (E.12a).
    """
    nu = np.asarray(nu)
    scalar_input = nu.ndim == 0
    if scalar_input:
        nu = nu[None]

    alpha, gamma = _get_alpha_gamma(p)
    z_val = Z(p)
    if np.isinf(z_val):
        return np.zeros_like(nu, dtype=float)

    delta = p.B - p.A
    n_over_Z = p.n / z_val
    mu_fac = p.mu_perp**2 / 8.0

    res = np.zeros_like(nu, dtype=float)

    for i, freq in enumerate(nu):
        if freq <= 0:
            continue

        def integrand_plus(x):
            D = p.B + delta * x
            J_plus = freq / (2 * D)
            exp_arg = -(alpha - gamma) * J_plus**2 - gamma * J_plus**2 * (1 - x**2)
            return freq**2 * np.exp(exp_arg) * (1 - x) ** 2 / (2 * D**3)

        def integrand_minus(x):
            D = p.B - delta * x
            J_minus = freq / (2 * D)
            exp_arg = -(alpha - gamma) * J_minus**2 - gamma * J_minus**2 * (1 - x**2)
            return freq**2 * np.exp(exp_arg) * (1 + x) ** 2 / (2 * D**3)

        val_plus, _ = quad(
            integrand_plus,
            0.0,
            1.0,
            epsabs=p.quad_epsabs,
            epsrel=p.quad_epsrel,
        )
        val_minus, _ = quad(
            integrand_minus,
            0.0,
            1.0,
            epsabs=p.quad_epsabs,
            epsrel=p.quad_epsrel,
        )
        res[i] = _R_nu(freq) * mu_fac * n_over_Z * (val_plus + val_minus)

    if scalar_input:
        return res[0]
    return res


def jnu_Q_perp(nu, p: RotorParams):
    """Compute the perpendicular Q-branch emissivity.

    This is the Sect. 5.1 Q branch, Eq. (20c), written in the continuum form
    of Appendix Eq. (E.14a).
    """
    nu = np.asarray(nu)
    scalar_input = nu.ndim == 0
    if scalar_input:
        nu = nu[None]

    alpha, gamma = _get_alpha_gamma(p)
    z_val = Z(p)
    if np.isinf(z_val):
        return np.zeros_like(nu, dtype=float)

    delta = p.B - p.A
    n_over_Z = p.n / z_val
    mu_fac = p.mu_perp**2 / 4.0
    inv_2Delta = 1.0 / (2 * delta)

    res = np.zeros_like(nu, dtype=float)

    for i, freq in enumerate(nu):
        if freq <= 0:
            continue

        K_nu = freq * inv_2Delta
        z_nu = alpha * K_nu**2
        if z_nu > 100.0:
            bracket = (1.0 / z_nu - 2.0 / z_nu**2 + 6.0 / z_nu**3) / alpha
        elif z_nu > 0.0:
            bracket = (1.0 - z_nu * np.exp(z_nu) * exp1(z_nu)) / alpha
        else:
            bracket = 1.0 / alpha

        res[i] = (
            _R_nu(freq)
            * mu_fac
            * n_over_Z
            * (np.exp(-(alpha - gamma) * K_nu**2) / delta)
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
