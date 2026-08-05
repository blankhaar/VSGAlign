"""DL98b coefficient backend.

Implements the dimensionless damping and excitation factors appearing in
DL98b (Draine & Lazarian 1998b; arXiv:astro-ph/9802239):

- Damping factors F_n, F_i, F_p, F_IR: DL98b eqs. (19), (20), (25), (30)–(33)
- Excitation factors G_n, G_i, G_p, G_IR: DL98b eqs. (38)–(43), (44)–(46)
- Optional additional excitation terms G_pe and G_H2: DL98b eqs. (47)–(48)
- Systematic torque term G_s (H2 formation): DL98b eqs. (49)–(54)

These are then used by the single-temperature solver (DL98b eqs. (55)–(59)).

Repository role
---------------
This module is the numerical core behind `DustGrain.compute_rotational_temperature`.
Its outputs do not appear directly in the paper scripts, but they determine the
`T_rot` values used in the current paper's Sect. 2.2 population model and
therefore feed every publication figure based on `DustGrain`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import erf

from ..charge import ChargeDistribution
from ..constants import c, eV_to_erg, e_esu, hbar, k_B, m_H, m_e, pi
from ..env import Environment, IonSpecies, NeutralSpecies
from ..grain import GrainModel
from ..units import T2, a_minus7, as_ndarray

__all__ = [
    "DL98FG",
    "compute_FG",
    "tau_H",
    "tau_ed",
    "omega_T",
]


def omega_T(I_g_cm2: ArrayLike, T_gas_K: ArrayLike) -> NDArray[np.float64]:
    """Thermal rms rotation rate ω_T (DL98b eq. (13)).

    DL98b eq. (13) quotes a numerical approximation for a spherical grain.
    The underlying relation is the Maxwellian moment:

      ⟨ω^2⟩ = 3 k_B T / I  →  ω_T ≡ √⟨ω^2⟩ = sqrt(3 k_B T / I).

    This function implements the exact expression in cgs.
    """

    I = as_ndarray(I_g_cm2)
    T = as_ndarray(T_gas_K)
    return np.sqrt(3.0 * k_B * T / I)


def tau_H(geom, env: Environment, grain: GrainModel) -> NDArray[np.float64]:
    """Fiducial damping time τ_H for pure-H sticky collisions (DL98b eqs. (15)–(16))."""

    # Ref: DL98b eq. (15). Uses ξ, ρ, a, a_x, n_H, T.
    a = geom.a_cm
    ax = geom.a_x_cm
    xi = geom.xi
    rho = float(grain.rho_g_cm3)
    nH = float(env.n_H)
    T = float(env.T_gas)

    return (4.0 * xi * rho * a**5 / (5.0 * nH * m_H * ax**4)) * np.sqrt(
        (pi * m_H) / (8.0 * k_B * T)
    )


def tau_ed(I_g_cm2: ArrayLike, mu2_cgs: ArrayLike, T_gas_K: ArrayLike) -> NDArray[np.float64]:
    """Electric dipole damping time τ_ed (DL98b eq. (34)).

    Implemented in the exact (non-approximate) form:

      τ_ed = 3 I^2 c^3 / (4 μ^2 k_B T)

    where μ is the electric dipole moment magnitude in cgs (esu*cm).
    """

    I = as_ndarray(I_g_cm2)
    mu2 = as_ndarray(mu2_cgs)
    T = as_ndarray(T_gas_K)
    if T.ndim == 0:
        T = np.full_like(I, float(T))
    out = np.full_like(I, np.inf, dtype=np.float64)
    mask = mu2 > 0
    out[mask] = 3.0 * I[mask] ** 2 * c**3 / (4.0 * mu2[mask] * k_B * T[mask])
    return out


def _g1(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Helper g1(x) (DL98b eq. (22))."""

    return np.where(x < 0.0, 1.0 - x, np.exp(-x))


def _g2(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Helper g2(x) (DL98b eq. (42))."""

    return np.where(x < 0.0, 1.0 - x + 0.5 * x**2, np.exp(-x))


def _epsilon2(alpha_cm3: float, a_s_cm: NDArray[np.float64], T_gas_K: float) -> NDArray[np.float64]:
    """ε_n^2 (DL98b eq. (23))."""

    return (e_esu**2 * alpha_cm3) / (2.0 * a_s_cm**4 * k_B * T_gas_K)


def _phi(a_s_cm: NDArray[np.float64], T_gas_K: float) -> NDArray[np.float64]:
    """φ (DL98b eq. (24))."""

    return np.sqrt((2.0 * e_esu**2) / (a_s_cm * k_B * T_gas_K))


def _mean_Z2_for_sizes(a_cm: NDArray[np.float64], env: Environment) -> NDArray[np.float64]:
    cd = env.charge_distribution
    if isinstance(cd, ChargeDistribution):
        return np.full_like(a_cm, cd.mean_Z2(), dtype=np.float64)
    # Potentially size-dependent: evaluate per element.
    return np.array([env.charge_dist(float(ai)).mean_Z2() for ai in a_cm], dtype=np.float64)


def _F_n(
    a_cm: NDArray[np.float64], geom, env: Environment, charge_dist: ChargeDistribution
) -> NDArray[np.float64]:
    """Neutral collision damping F_n (DL98b eq. (19))."""

    a_s = geom.a_s_cm
    T = float(env.T_gas)
    out = np.zeros_like(a_cm)

    Z = charge_dist.Z.astype(np.float64)
    fZ = charge_dist.f.astype(np.float64)
    absZ = np.abs(Z)
    Z2 = Z**2

    for sp in env.neutrals:
        sp: NeutralSpecies
        pref = (sp.n_cm3 / env.n_H) * np.sqrt(sp.mass_g / m_H)
        eps2 = _epsilon2(sp.polarizability_cm3, a_s, T)
        eps = np.sqrt(eps2)

        term = np.zeros_like(a_cm)
        for Zi, fi, aZi, Zi2 in zip(Z, fZ, absZ, Z2, strict=False):
            # Ref: DL98b eq. (19) bracket term.
            term += fi * (np.exp(-Zi2 * eps2) + aZi * eps * np.sqrt(pi) * erf(aZi * eps))
        out += pref * term

    return out


def _F_i(
    a_cm: NDArray[np.float64], geom, env: Environment, charge_dist: ChargeDistribution
) -> NDArray[np.float64]:
    """Ion collision damping F_i (DL98b eq. (20))."""

    a_s = geom.a_s_cm
    T = float(env.T_gas)
    phi = _phi(a_s, T)

    Z = charge_dist.Z.astype(np.int64)
    fZ = charge_dist.f.astype(np.float64)
    f0 = charge_dist.prob(0)

    out = np.zeros_like(a_cm)

    # Z_g != 0 term (DL98b eq. (20), first sum).
    mask_nonzero = Z != 0
    if np.any(mask_nonzero):
        Znz = Z[mask_nonzero].astype(np.float64)
        fnz = fZ[mask_nonzero]
        for ion in env.ions:
            ion: IonSpecies
            pref = (ion.n_cm3 / env.n_H) * np.sqrt(ion.mass_g / m_H)
            x = (Znz[:, None] * ion.Z * e_esu**2) / (a_s[None, :] * k_B * T)
            # Vectorized over charge states then summed.
            g = _g1(x)
            out += pref * np.sum(fnz[:, None] * g, axis=0)

    # Z_g = 0 term (DL98b eq. (20), second sum).
    if f0 > 0:
        for ion in env.ions:
            pref = (ion.n_cm3 / env.n_H) * np.sqrt(ion.mass_g / m_H)
            out += f0 * pref * (1.0 + 0.5 * np.sqrt(pi) * ion.Z * phi)

    return out


def _F_p(a_cm: NDArray[np.float64], geom, env: Environment, mu2_cgs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Plasma drag damping F_p (DL98b eq. (25)).

    Notes
    -----
    The log form and cutoffs are defined by DL98b eqs. (26)–(28). For the
    log structure we follow the (dimensionally consistent) form used in
    DL98b Appendix B.4 (see their discussion around plasma drag), which is
    equivalent to the intended meaning of eq. (25)/(43).
    """

    a_s = geom.a_s_cm
    ax = geom.a_x_cm
    I = geom.I_g_cm2

    T = float(env.T_gas)
    n_e = float(env.n_e)
    cos2 = float(env.cos2_psi)

    # If there is no plasma, the coefficient vanishes (the sums in DL98b eq. (25)/(43)
    # are over ion species). Also avoid Debye-length division by zero.
    if (not env.ions) or (n_e <= 0.0) or np.all(mu2_cgs == 0.0):
        return np.zeros_like(a_cm)

    # Ref: DL98b eq. (26): Debye length.
    lambda_D = np.sqrt(k_B * T / (4.0 * pi * n_e * e_esu**2))

    out = np.zeros_like(a_cm)
    for ion in env.ions:
        ion: IonSpecies
        pref = (ion.n_cm3 * ion.Z**2 / env.n_H) * np.sqrt(ion.mass_g / m_H)
        # Ref: DL98b eq. (27): b_omega = sqrt(I / m_i)
        b_omega = np.sqrt(I / ion.mass_g)
        # Ref: DL98b eq. (28): b_q = I * sqrt(2 kT / m_i) / hbar
        b_q = I * np.sqrt(2.0 * k_B * T / ion.mass_g) / hbar

        b_max = np.minimum(b_q, lambda_D)
        log_term = np.log(b_omega / a_s) + cos2 * np.log(b_max / b_omega)

        out += (
            (2.0 * e_esu**2 * mu2_cgs) / (3.0 * ax**4 * (k_B * T) ** 2) * pref * log_term
        )

    return out


def _F_IR(a_cm: NDArray[np.float64], geom, env: Environment) -> NDArray[np.float64]:
    """Infrared damping F_IR (DL98b eqs. (30)–(33))."""

    a7 = a_minus7(a_cm)
    T2v = T2(env.T_gas)
    nH = float(env.n_H)
    chi = float(env.chi)

    # Ref: DL98b eq. (30)
    F_c = (60.8 / a7) * chi ** (2.0 / 3.0) * (20.0 / nH) * (T2v ** -0.5) * (a_cm / geom.a_x_cm) ** 4
    # Ref: DL98b eq. (31)
    F_q = 4.49 * (a7**0.5) * chi * (20.0 / nH) * (T2v ** -0.5) * (a_cm / geom.a_x_cm) ** 4
    # Ref: DL98b eq. (33)
    return np.minimum(F_q, F_c)


def _G_n(
    a_cm: NDArray[np.float64], geom, env: Environment, charge_dist: ChargeDistribution
) -> NDArray[np.float64]:
    """Neutral collision excitation G_n (DL98b eq. (38))."""

    a_s = geom.a_s_cm
    T = float(env.T_gas)
    T_ev = float(env.T_dust)

    out = np.zeros_like(a_cm)

    Z = charge_dist.Z.astype(np.float64)
    fZ = charge_dist.f.astype(np.float64)
    absZ = np.abs(Z)
    Z2 = Z**2

    for sp in env.neutrals:
        sp: NeutralSpecies
        pref = (sp.n_cm3 / (2.0 * env.n_H)) * np.sqrt(sp.mass_g / m_H)
        eps2 = _epsilon2(sp.polarizability_cm3, a_s, T)

        term = np.zeros_like(a_cm)
        for Zi2, fi, aZi in zip(Z2, fZ, absZ, strict=False):
            # Ref: DL98b eq. (38) curly braces.
            term += fi * (
                np.exp(-Zi2 * eps2)
                + 2.0 * Zi2 * eps2
                + (T_ev / T)
                * (
                    np.exp(-Zi2 * eps2 * (T / T_ev))
                    + 2.0 * Zi2 * eps2 * (T / T_ev)
                )
            )
        out += pref * term

    return out


def _G_i_in(
    a_cm: NDArray[np.float64], geom, env: Environment, charge_dist: ChargeDistribution
) -> NDArray[np.float64]:
    """Incoming-ion excitation G_i^(in) (DL98b eq. (40))."""

    a_s = geom.a_s_cm
    T = float(env.T_gas)
    phi = _phi(a_s, T)

    Z = charge_dist.Z.astype(np.int64)
    fZ = charge_dist.f.astype(np.float64)
    f0 = charge_dist.prob(0)

    out = np.zeros_like(a_cm)

    mask_nonzero = Z != 0
    if np.any(mask_nonzero):
        Znz = Z[mask_nonzero].astype(np.float64)
        fnz = fZ[mask_nonzero]
        for ion in env.ions:
            ion: IonSpecies
            pref = (ion.n_cm3 / (2.0 * env.n_H)) * np.sqrt(ion.mass_g / m_H)
            x = (Znz[:, None] * ion.Z * e_esu**2) / (a_s[None, :] * k_B * T)
            out += pref * np.sum(fnz[:, None] * _g2(x), axis=0)

    if f0 > 0:
        for ion in env.ions:
            pref = (ion.n_cm3 / (2.0 * env.n_H)) * np.sqrt(ion.mass_g / m_H)
            out += f0 * pref * (1.0 + (3.0 * np.sqrt(pi) / 4.0) * phi + 0.5 * phi**2)

    return out


def _G_i_ev(
    a_cm: NDArray[np.float64], geom, env: Environment, charge_dist: ChargeDistribution
) -> NDArray[np.float64]:
    """Evaporation-after-ion excitation G_i^(ev) (DL98b eq. (41))."""

    a_s = geom.a_s_cm
    T = float(env.T_gas)
    T_ev = float(env.T_dust)
    phi = _phi(a_s, T)

    Z = charge_dist.Z.astype(np.int64)
    fZ = charge_dist.f.astype(np.float64)
    f0 = charge_dist.prob(0)

    out = np.zeros_like(a_cm)

    mask_nonzero = Z != 0
    if np.any(mask_nonzero):
        Znz = Z[mask_nonzero].astype(np.float64)
        fnz = fZ[mask_nonzero]
        absZ = np.abs(Znz)
        Z2 = Znz**2

        for ion in env.ions:
            ion: IonSpecies
            pref = (ion.n_cm3 / (2.0 * env.n_H)) * np.sqrt(ion.mass_g / m_H)

            # Ref: DL98b eq. (23) but with the polarizability of the neutral after capture.
            eps2 = _epsilon2(ion.neutral_polarizability_cm3, a_s, T)
            eps = np.sqrt(eps2)

            x = (Znz[:, None] * ion.Z * e_esu**2) / (a_s[None, :] * k_B * T)
            g1v = _g1(x)

            denom = np.exp(-Z2[:, None] * eps2[None, :]) + absZ[:, None] * eps[None, :] * np.sqrt(pi) * erf(
                absZ[:, None] * eps[None, :]
            )
            numer = np.exp(-Z2[:, None] * eps2[None, :]) + 2.0 * Z2[:, None] * eps2[None, :]
            ratio = numer / denom

            out += pref * np.sum(fnz[:, None] * g1v * (T_ev / T) * ratio, axis=0)

    if f0 > 0:
        for ion in env.ions:
            pref = (ion.n_cm3 / (2.0 * env.n_H)) * np.sqrt(ion.mass_g / m_H)
            out += f0 * pref * (T_ev / T) * (1.0 + 0.5 * np.sqrt(pi) * phi)

    return out


def _G_p(a_cm: NDArray[np.float64], geom, env: Environment, mu2_cgs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Plasma drag excitation G_p (DL98b eq. (43))."""

    # Same functional form as F_p in DL98b (compare eq. (25) and eq. (43)).
    return _F_p(a_cm=a_cm, geom=geom, env=env, mu2_cgs=mu2_cgs)


def _G_IR(a_cm: NDArray[np.float64], geom, env: Environment) -> NDArray[np.float64]:
    """Infrared excitation G_IR (DL98b eqs. (44)–(46))."""

    a7 = a_minus7(a_cm)
    T2v = T2(env.T_gas)
    nH = float(env.n_H)
    chi = float(env.chi)

    # Ref: DL98b eq. (44)
    G_c = (7.34 / a7) * (a_cm / geom.a_x_cm) ** 4 * chi ** (5.0 / 6.0) * (20.0 / nH) * (T2v ** -1.5)
    # Ref: DL98b eq. (45)
    G_q = (2.11 / (a7 ** 0.25)) * (a_cm / geom.a_x_cm) ** 4 * chi * (20.0 / nH) * (T2v ** -1.5)
    # Ref: DL98b eq. (46)
    return np.minimum(G_c, G_q)


def _G_pe(a_cm: NDArray[np.float64], geom, env: Environment, charge_dist: ChargeDistribution) -> NDArray[np.float64]:
    """Photoelectric excitation G_pe (DL98b eq. (47)).

    If no photoelectron providers are configured in the environment, this returns 0.
    """

    if env.photoelectron_rate is None:
        return np.zeros_like(a_cm)
    if env.photoelectron_mean_energy_erg is None:
        mean_energy = lambda _a, _Z: 0.0  # noqa: E731
    else:
        mean_energy = env.photoelectron_mean_energy_erg

    a_s = geom.a_s_cm
    ax = geom.a_x_cm
    T = float(env.T_gas)

    Z = charge_dist.Z.astype(int)
    fZ = charge_dist.f.astype(np.float64)

    out = np.zeros_like(a_cm)
    # Prefactor from DL98b eq. (47).
    pref = m_e / (4.0 * env.n_H * np.sqrt(8.0 * pi * m_H * k_B * T) * ax**2 * k_B * T)

    for Zi, fi in zip(Z, fZ, strict=False):
        Ndot = np.array([env.photoelectron_rate(float(ai), int(Zi)) for ai in a_cm], dtype=np.float64)
        Emean = np.array([mean_energy(float(ai), int(Zi)) for ai in a_cm], dtype=np.float64)
        out += fi * Ndot * (Emean + ((Zi + 1) * e_esu**2) / a_s)

    return pref * out


def _G_H2_random(a_cm: NDArray[np.float64], geom, env: Environment) -> NDArray[np.float64]:
    """Random H2 formation excitation G_H2 (DL98b eq. (48))."""

    gamma = float(env.h2_random_gamma)
    if gamma <= 0:
        return np.zeros_like(a_cm)
    y = float(env.y)
    Ef = float(env.h2_random_Ef_eV) * eV_to_erg
    J2 = float(env.h2_random_J2)
    T = float(env.T_gas)

    ax = geom.a_x_cm
    return (
        (gamma / 4.0)
        * (1.0 - y)
        * (Ef / (k_B * T))
        * (1.0 + (J2 * hbar**2) / (2.0 * m_H * Ef * ax**2))
    )


def _G_s_H2_systematic(
    a_cm: NDArray[np.float64],
    geom,
    env: Environment,
    F_total: NDArray[np.float64],
    tauH: NDArray[np.float64],
    mu2_cgs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Systematic H2 formation torques: compute G_s (DL98b eqs. (49)–(54))."""

    gamma = float(env.h2_systematic_gamma)
    if gamma <= 0:
        return np.zeros_like(a_cm)

    # Torque Γ_s (DL98b eq. (52)).
    nH0 = float(env.n_H0)  # n(H)
    Ef = float(env.h2_systematic_Ef_eV) * eV_to_erg
    T = float(env.T_gas)
    Nr = np.array([env.recombination_sites(float(ai)) for ai in a_cm], dtype=np.float64)

    Gamma_s = (
        (4.0 * gamma * nH0 * geom.a_x_cm**2 * geom.a_s_cm) / np.sqrt(Nr)
    ) * np.sqrt(pi * m_H * Ef * k_B * T)

    # Steady ω_s approximation (DL98b eq. (50)).
    I = geom.I_g_cm2
    mu2 = mu2_cgs
    # Avoid division by zero for very small torques: Γ_s→0 implies G_s→0.
    out = np.zeros_like(a_cm)
    mask = Gamma_s > 0
    if not np.any(mask):
        return out

    Gamma = Gamma_s[mask]
    tauH_m = tauH[mask]
    I_m = I[mask]
    F_m = F_total[mask]
    mu2_m = mu2[mask]

    omega0 = (Gamma * tauH_m) / (I_m * F_m)
    corr = 1.0 + (4.0 * mu2_m / (9.0 * Gamma * c**3)) * omega0**3
    omega_s = omega0 * corr ** (-1.0 / 3.0)

    # Dimensionless excitation from systematic torque (DL98b eq. (51)).
    denom = env.n_H * np.sqrt(2.0 * pi * m_H * k_B * T) * 4.0 * geom.a_x_cm[mask] ** 4 * k_B * T
    out[mask] = (I_m * Gamma * omega_s) / denom
    return out


@dataclass(frozen=True, slots=True)
class DL98FG:
    """Container for DL98b F/G terms (arrays over grain size)."""

    a_cm: NDArray[np.float64]
    mean_Z2: NDArray[np.float64]
    mu2_cgs: NDArray[np.float64]

    F_n: NDArray[np.float64]
    F_i: NDArray[np.float64]
    F_p: NDArray[np.float64]
    F_IR: NDArray[np.float64]
    F_total: NDArray[np.float64]

    G_n: NDArray[np.float64]
    G_i_in: NDArray[np.float64]
    G_i_ev: NDArray[np.float64]
    G_i: NDArray[np.float64]
    G_p: NDArray[np.float64]
    G_IR: NDArray[np.float64]
    G_pe: NDArray[np.float64]
    G_H2: NDArray[np.float64]
    G_s: NDArray[np.float64]
    G_total: NDArray[np.float64]


def compute_FG(a_cm: ArrayLike, env: Environment, grain: GrainModel) -> DL98FG:
    """Compute DL98b dimensionless damping/excitation terms for size(s) `a`.

    This is the backend entry point used by `DustGrain` before the results are
    collapsed into a single rotational temperature for the rest of the codebase.
    """

    a = np.atleast_1d(as_ndarray(a_cm))

    cd = env.charge_distribution
    if isinstance(cd, ChargeDistribution):
        # Vectorized path: single charge distribution for all sizes.
        geom = grain.geometry(a)
        meanZ2 = np.full_like(a, cd.mean_Z2(), dtype=np.float64)
        mu2 = grain.mu2_cgs(a, meanZ2)  # Ref: DL98b eq. (11)

        F_n = _F_n(a, geom, env, cd)  # Ref: DL98b eq. (19)
        F_i = _F_i(a, geom, env, cd)  # Ref: DL98b eq. (20)
        F_p = _F_p(a, geom, env, mu2)  # Ref: DL98b eq. (25)
        F_IR = _F_IR(a, geom, env)  # Ref: DL98b eqs. (30)–(33)
        F_total = F_n + F_i + F_p + F_IR  # Ref: DL98b eq. (18)

        G_n = _G_n(a, geom, env, cd)  # Ref: DL98b eq. (38)
        G_i_in = _G_i_in(a, geom, env, cd)  # Ref: DL98b eq. (40)
        G_i_ev = _G_i_ev(a, geom, env, cd)  # Ref: DL98b eq. (41)
        G_i = G_i_in + G_i_ev  # Ref: DL98b eq. (39)
        G_p = _G_p(a, geom, env, mu2)  # Ref: DL98b eq. (43)
        G_IR = _G_IR(a, geom, env)  # Ref: DL98b eqs. (44)–(46)

        # Additional (impulsive) excitation terms derived by DL98b but not included in their
        # default single-T closure G (see DL98b eq. (56)).
        G_pe = _G_pe(a, geom, env, cd)  # Ref: DL98b eq. (47)
        G_H2 = _G_H2_random(a, geom, env)  # Ref: DL98b eq. (48)

        # Systematic torques (H2 formation) are included in DL98b's G definition.
        tauH = tau_H(geom, env, grain)  # Ref: DL98b eq. (15)
        G_s = _G_s_H2_systematic(a, geom, env, F_total, tauH, mu2)  # Ref: DL98b eq. (51)

        G_total = G_n + G_i + G_p + G_IR + G_s  # Ref: DL98b eq. (56)

        return DL98FG(
            a_cm=a,
            mean_Z2=meanZ2,
            mu2_cgs=mu2,
            F_n=F_n,
            F_i=F_i,
            F_p=F_p,
            F_IR=F_IR,
            F_total=F_total,
            G_n=G_n,
            G_i_in=G_i_in,
            G_i_ev=G_i_ev,
            G_i=G_i,
            G_p=G_p,
            G_IR=G_IR,
            G_pe=G_pe,
            G_H2=G_H2,
            G_s=G_s,
            G_total=G_total,
        )

    # Size-dependent f(Z_g): correct but not vectorized; loop over sizes.
    meanZ2 = np.empty_like(a)
    mu2 = np.empty_like(a)
    F_n = np.empty_like(a)
    F_i = np.empty_like(a)
    F_p = np.empty_like(a)
    F_IR = np.empty_like(a)
    F_total = np.empty_like(a)
    G_n = np.empty_like(a)
    G_i_in = np.empty_like(a)
    G_i_ev = np.empty_like(a)
    G_i = np.empty_like(a)
    G_p = np.empty_like(a)
    G_IR = np.empty_like(a)
    G_pe = np.empty_like(a)
    G_H2 = np.empty_like(a)
    G_s = np.empty_like(a)
    G_total = np.empty_like(a)

    for idx, ai in enumerate(a):
        cd_i = env.charge_dist(float(ai))
        geom_i = grain.geometry(np.array([ai], dtype=np.float64))
        meanZ2_i = cd_i.mean_Z2()
        mu2_i = float(grain.mu2_cgs(np.array([ai], dtype=np.float64), meanZ2_i)[0])

        meanZ2[idx] = meanZ2_i
        mu2[idx] = mu2_i

        F_n[idx] = float(_F_n(np.array([ai]), geom_i, env, cd_i)[0])
        F_i[idx] = float(_F_i(np.array([ai]), geom_i, env, cd_i)[0])
        F_p[idx] = float(_F_p(np.array([ai]), geom_i, env, np.array([mu2_i]))[0])
        F_IR[idx] = float(_F_IR(np.array([ai]), geom_i, env)[0])
        F_total[idx] = F_n[idx] + F_i[idx] + F_p[idx] + F_IR[idx]

        G_n[idx] = float(_G_n(np.array([ai]), geom_i, env, cd_i)[0])
        G_i_in[idx] = float(_G_i_in(np.array([ai]), geom_i, env, cd_i)[0])
        G_i_ev[idx] = float(_G_i_ev(np.array([ai]), geom_i, env, cd_i)[0])
        G_i[idx] = G_i_in[idx] + G_i_ev[idx]
        G_p[idx] = float(_G_p(np.array([ai]), geom_i, env, np.array([mu2_i]))[0])
        G_IR[idx] = float(_G_IR(np.array([ai]), geom_i, env)[0])
        G_pe[idx] = float(_G_pe(np.array([ai]), geom_i, env, cd_i)[0])
        G_H2[idx] = float(_G_H2_random(np.array([ai]), geom_i, env)[0])

        tauH_i = float(tau_H(geom_i, env, grain)[0])
        G_s[idx] = float(
            _G_s_H2_systematic(
                np.array([ai]), geom_i, env, np.array([F_total[idx]]), np.array([tauH_i]), np.array([mu2_i])
            )[0]
        )
        G_total[idx] = G_n[idx] + G_i[idx] + G_p[idx] + G_IR[idx] + G_s[idx]

    return DL98FG(
        a_cm=a,
        mean_Z2=meanZ2,
        mu2_cgs=mu2,
        F_n=F_n,
        F_i=F_i,
        F_p=F_p,
        F_IR=F_IR,
        F_total=F_total,
        G_n=G_n,
        G_i_in=G_i_in,
        G_i_ev=G_i_ev,
        G_i=G_i,
        G_p=G_p,
        G_IR=G_IR,
        G_pe=G_pe,
        G_H2=G_H2,
        G_s=G_s,
        G_total=G_total,
    )
