"""`DustGrain` model compatible with `ism-newang` dust-properties interface.

Design goal
-----------
Mirror the external `dust_properties.dust_alignment.DustGrain` API while using
the DL98/AHD09 single-temperature rotational solver implemented in this repo.

Paper connection
----------------
The geometry, rotational constants, and temperature mapping are the code-side
representation of the oblate symmetric-top grain model introduced in Sect. 2 of
`the companion VSG/AME manuscript`, especially
Eqs. (1a)-(1c), Eq. (3), and Eq. (4).

Compatibility targets include the attributes/methods consumed by the internal
bridges (`dust_emissivity.adapter`, `rotor_polarized_emissivity.dust_bridge`),
namely:

- attributes: `Arot`, `Brot`, `mu_par`, `mu_perp`, `T_rot`, `n_H`
- methods: `critical_temperature`, `x_ratio`, `y_param`, `sigma_JK`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from spindust_trot.charge import ChargeDistribution
from spindust_trot.coeffs import CoeffVersion
from spindust_trot.coeffs import dl98 as dl98_coeffs
from spindust_trot.constants import e_esu, hbar, k_B, pi
from spindust_trot.env import Environment

# Constants kept intentionally aligned with the external implementation.
H_BAR = hbar  # erg s
C_LIGHT = 2.99792458e10  # cm/s
KB = k_B  # erg/K
DEBYE = 1.0e-18  # esu cm
h_CGS = H_BAR * 2.0 * np.pi


@dataclass(frozen=True, slots=True)
class _OblateTemperatureGeometry:
    """Geometry container expected by `spindust_trot.coeffs.dl98` helper functions."""

    a_cm: np.ndarray
    a_s_cm: np.ndarray
    a_x_cm: np.ndarray
    xi: np.ndarray
    mass_g: np.ndarray
    I_g_cm2: np.ndarray


@dataclass(frozen=True, slots=True)
class _RhoProxy:
    """Minimal object exposing `rho_g_cm3` for `dl98.tau_H`."""

    rho_g_cm3: float


def _surface_area_oblate(a: float, c: float) -> float:
    """Surface area of an oblate spheroid (a = b >= c)."""

    if a <= 0.0 or c <= 0.0:
        raise ValueError("Semi-axes must be positive.")
    if np.isclose(a, c):
        return 4.0 * pi * a**2
    e = np.sqrt(max(0.0, 1.0 - (c * c) / (a * a)))
    if e < 1.0e-12:
        return 4.0 * pi * a**2
    return 2.0 * pi * a * a * (1.0 + ((1.0 - e * e) / e) * np.arctanh(e))


def _excitation_equivalent_radius_oblate(a: float, c: float) -> float:
    """Compute `a_x` from the defining surface integral.

    By definition (DL98b eq. (3)):
      4 pi a_x^4 = integral r^2 dS
    where `r` is distance from the center to a surface element.

    For an oblate spheroid (`a=b>=c`) this integral has a closed-form
    reduction to elementary functions, so no quadrature is needed.
    """
    if a <= 0.0 or c <= 0.0:
        raise ValueError("Semi-axes must be positive.")

    q = c / a
    if q > 1.0 + 1.0e-12:
        raise ValueError("Expected oblate geometry (c <= a).")
    q = min(max(q, 1.0e-300), 1.0)

    if np.isclose(q, 1.0, rtol=0.0, atol=1.0e-12):
        return float(a)

    q2 = q * q
    e2 = 1.0 - q2
    sqrt_e = np.sqrt(e2)

    # Shape factor J such that a_x = a * J^(1/4), derived from
    # integral_{surface} r^2 dS for an oblate spheroid.
    asinh_term = np.arcsinh(sqrt_e / q)
    J = (2.0 + q2) / 8.0 + (q2 * (4.0 + q2) / (8.0 * sqrt_e)) * asinh_term
    return float(a * (J ** 0.25))


class DustGrain:
    """Compatibility dust grain model with integrated rotational-temperature computer.

    Core constructor parameters intentionally match the external implementation.
    The resulting object supplies the grain properties consumed by the Sect. 5
    emissivity and Sect. 4 alignment interfaces in this repository.

    Parameters
    ----------
    a_eff : float
        Effective radius [cm].
    rho : float
        Grain density [g cm^-3].
    s : float
        Shape factor c/a for an oblate spheroid.
    T_gas : float
        Gas temperature [K].
    n_H : float
        Hydrogen number density [cm^-3].
    U : float
        Radiation field scale factor relative to ISRF.
    mu_anis : float | None
        Dipole anisotropy. If None, inferred from shape as in external code.

    Additional keyword-only parameters are compatibility adapters for the
    `spindust_trot` temperature solver and default to CNM-like values.
    """

    def __init__(
        self,
        a_eff: float = 0.5e-7,
        rho: float = 3.4,
        s: float = 0.5,
        T_gas: float = 100.0,
        n_H: float = 30.0,
        U: float = 1.0,
        mu_anis: float | None = None,
        *,
        beta_debye: float = 0.3,
        T_dust: float = 20.0,
        x_H: float = 0.0012,
        x_M: float = 0.0003,
        y: float = 0.0,
        he_fraction: float = 0.1,
        cos2_psi: float = 1.0 / 3.0,
        charge_distribution: ChargeDistribution | None = None,
        charge_centroid_fraction: float = 0.0,
        coeff_version: CoeffVersion = "ahd09",
        recompute_t_rot: bool = True,
    ):
        self.a_eff = float(a_eff)
        self.rho = float(rho)
        self.s = float(s)
        self.T_gas = float(T_gas)
        self.n_H = float(n_H)
        self.U = float(U)
        self.mu_anis = mu_anis

        # Additional environment parameters used by the rotational-temperature solver.
        self.T_dust = float(T_dust)
        self.x_H = float(x_H)
        self.x_M = float(x_M)
        self.y = float(y)
        self.he_fraction = float(he_fraction)
        self.cos2_psi = float(cos2_psi)
        self.charge_distribution = charge_distribution or ChargeDistribution.delta(0)
        self.charge_centroid_fraction = float(charge_centroid_fraction)
        self.coeff_version: CoeffVersion = coeff_version

        # Geometry from the external DustGrain definition.
        self.a = self.a_eff * (self.s) ** (-1.0 / 3.0)
        self.c = self.s * self.a

        self.V = (4.0 / 3.0) * np.pi * self.a**2 * self.c
        self.mass = self.rho * self.V

        # Moments of inertia for oblate spheroid.
        self.I_par = 0.4 * self.mass * self.a**2
        self.I_perp = 0.2 * self.mass * (self.a**2 + self.c**2)

        # Rotational constants in Hz (compatible with external implementation).
        self.Arot = H_BAR / (4.0 * np.pi * self.I_par)
        self.Brot = H_BAR / (4.0 * np.pi * self.I_perp)

        # Number of atoms and intrinsic dipole.
        self.N_atoms = 40.0 * (self.a_eff / 5e-8) ** 3
        self.beta_debye = float(beta_debye)
        beta_cgs = self.beta_debye * DEBYE
        self.mu_sq = (beta_cgs**2) * self.N_atoms

        # Dipole anisotropy split.
        if self.mu_anis is not None:
            f_par = (2.0 * self.mu_anis + 1.0) / 3.0
            f_par = max(0.0, min(1.0, f_par))
            self.mu_par_sq = self.mu_sq * f_par
            self.mu_perp_sq = self.mu_sq * (1.0 - f_par)
            self.mu_anis = float(self.mu_anis)
        else:
            self.mu_anis = (self.s * self.s - 1.0) / (self.s * self.s + 2.0)
            self.mu_perp_sq = self.mu_sq * 2.0 / (self.s * self.s + 2.0)
            self.mu_par_sq = self.mu_sq * self.s * self.s / (self.s * self.s + 2.0)

        self.mu_perp = float(np.sqrt(max(0.0, self.mu_perp_sq)))
        self.mu_par = float(np.sqrt(max(0.0, self.mu_par_sq)))

        # Temperature computer output (set below).
        self.T_rot = self.T_gas
        self._last_temperature_debug: dict[str, float] | None = None
        if recompute_t_rot:
            self.compute_rotational_temperature(coeff_version=self.coeff_version, update=True)

    def _build_environment(self) -> Environment:
        return Environment(
            n_H=self.n_H,
            T_gas=self.T_gas,
            T_dust=self.T_dust,
            chi=self.U,
            y=self.y,
            x_H=self.x_H,
            x_M=self.x_M,
            he_fraction=self.he_fraction,
            n_e=None,
            cos2_psi=self.cos2_psi,
            charge_distribution=self.charge_distribution,
        )

    def _temperature_geometry(self) -> _OblateTemperatureGeometry:
        a_arr = np.array([self.a_eff], dtype=np.float64)

        # Surface-equivalent radius from exact oblate-spheroid area.
        S = _surface_area_oblate(self.a, self.c)
        a_s = np.sqrt(S / (4.0 * pi))

        # Excitation-equivalent radius from the defining integral.
        a_x = _excitation_equivalent_radius_oblate(self.a, self.c)

        # Use largest principal moment for DL98-style xi normalization.
        I_major = max(self.I_par, self.I_perp)
        xi = I_major / (0.4 * self.mass * self.a_eff**2)

        return _OblateTemperatureGeometry(
            a_cm=a_arr,
            a_s_cm=np.array([a_s], dtype=np.float64),
            a_x_cm=np.array([a_x], dtype=np.float64),
            xi=np.array([xi], dtype=np.float64),
            mass_g=np.array([self.mass], dtype=np.float64),
            I_g_cm2=np.array([I_major], dtype=np.float64),
        )

    def _total_mu2_for_temperature(self, geom: _OblateTemperatureGeometry, env: Environment) -> np.ndarray:
        # Mean-square charge from the supplied charge distribution.
        mean_z2 = env.charge_dist(self.a_eff).mean_Z2()
        mu_charge2 = (self.charge_centroid_fraction * e_esu * geom.a_x_cm[0]) ** 2 * mean_z2
        return np.array([self.mu_sq + mu_charge2], dtype=np.float64)

    def compute_rotational_temperature(
        self,
        *,
        coeff_version: CoeffVersion | None = None,
        update: bool = True,
        return_debug: bool = False,
    ) -> float | tuple[float, dict[str, float]]:
        """Compute `T_rot` with DL98/AHD09 single-temperature closure.

        Returns
        -------
        float or (float, dict)
            Rotational temperature [K], optionally with diagnostic terms.
        """

        version = self.coeff_version if coeff_version is None else coeff_version
        if version not in ("dl98", "ahd09"):
            raise KeyError(f"Unknown coeff_version={version!r}")

        env = self._build_environment()
        geom = self._temperature_geometry()
        a = geom.a_cm

        charge_dist = env.charge_dist(self.a_eff)
        mu2 = self._total_mu2_for_temperature(geom, env)

        # DL98 coefficients with custom geometry.
        F_n = dl98_coeffs._F_n(a, geom, env, charge_dist)[0]  # type: ignore[attr-defined]
        F_i = dl98_coeffs._F_i(a, geom, env, charge_dist)[0]  # type: ignore[attr-defined]
        F_p = dl98_coeffs._F_p(a, geom, env, mu2)[0]  # type: ignore[attr-defined]
        F_IR = dl98_coeffs._F_IR(a, geom, env)[0]  # type: ignore[attr-defined]

        if version == "ahd09":
            # AHD09 (eq. 151/153): IR damping is 2x DL98 for the same IR spectrum.
            F_IR *= 2.0

        F_total = F_n + F_i + F_p + F_IR

        G_n = dl98_coeffs._G_n(a, geom, env, charge_dist)[0]  # type: ignore[attr-defined]
        G_i_in = dl98_coeffs._G_i_in(a, geom, env, charge_dist)[0]  # type: ignore[attr-defined]
        G_i_ev = dl98_coeffs._G_i_ev(a, geom, env, charge_dist)[0]  # type: ignore[attr-defined]
        G_i = G_i_in + G_i_ev
        G_p = dl98_coeffs._G_p(a, geom, env, mu2)[0]  # type: ignore[attr-defined]
        G_IR = dl98_coeffs._G_IR(a, geom, env)[0]  # type: ignore[attr-defined]

        tau_H = dl98_coeffs.tau_H(geom, env, _RhoProxy(self.rho))[0]
        G_s = dl98_coeffs._G_s_H2_systematic(  # type: ignore[attr-defined]
            a, geom, env, np.array([F_total]), np.array([tau_H]), mu2
        )[0]
        G_total = G_n + G_i + G_p + G_IR + G_s

        tau_ed = dl98_coeffs.tau_ed(geom.I_g_cm2, mu2, self.T_gas)[0]
        dipole_term = 0.0 if not np.isfinite(tau_ed) else (20.0 * tau_H) / (3.0 * tau_ed)

        # DL98 eq. (58), scalar form.
        A_term = (G_total / (F_total * F_total)) * dipole_term
        omega2 = (2.0 / (1.0 + np.sqrt(1.0 + A_term))) * (G_total / F_total) * (
            3.0 * k_B * self.T_gas / geom.I_g_cm2[0]
        )
        t_rot = float(geom.I_g_cm2[0] * omega2 / (3.0 * k_B))

        debug = {
            "a_eff_cm": self.a_eff,
            "a_eff_A": self.a_eff * 1.0e8,
            "a_s_over_a_eff": float(geom.a_s_cm[0] / self.a_eff),
            "a_x_over_a_eff": float(geom.a_x_cm[0] / self.a_eff),
            "xi": float(geom.xi[0]),
            "F_n": float(F_n),
            "F_i": float(F_i),
            "F_p": float(F_p),
            "F_IR": float(F_IR),
            "F_total": float(F_total),
            "G_n": float(G_n),
            "G_i": float(G_i),
            "G_p": float(G_p),
            "G_IR": float(G_IR),
            "G_s": float(G_s),
            "G_total": float(G_total),
            "tau_H_s": float(tau_H),
            "tau_ed_s": float(tau_ed),
            "T_rot_K": float(t_rot),
        }

        if update:
            self.T_rot = t_rot
            self.coeff_version = version
            self._last_temperature_debug = debug

        if return_debug:
            return t_rot, debug
        return t_rot

    def temperature_debug(self) -> dict[str, float]:
        """Return the latest rotational-temperature diagnostics."""

        if self._last_temperature_debug is None:
            _, debug = self.compute_rotational_temperature(update=False, return_debug=True)
            return debug
        return dict(self._last_temperature_debug)

    def critical_temperature(self) -> float:
        """Critical internal temperature following the external DustGrain model."""

        a_eff_A = self.a_eff * 1.0e8
        if a_eff_A <= 0.0:
            return 0.0
        if a_eff_A < 5.0:
            return 1.0e4 * (a_eff_A**-2.7)
        return 1.8e3 * (a_eff_A**-1.6)

    def absorption_rate(self) -> float:
        """Absorption rate [s^-1] following external dust-properties scaling."""

        a_A = self.a_eff * 1.0e8
        return 5.0e-7 * self.U * (a_A / 10.0) ** 3

    def collision_rate(self) -> float:
        """Collision rate [s^-1] following external dust-properties scaling."""

        a_A = self.a_eff * 1.0e8
        return 1.4e-7 * (self.n_H / 30.0) * np.sqrt(self.T_gas / 100.0) * (a_A / 10.0) ** 2

    def depolarization_rate(self) -> float:
        """Depolarization rate [s^-1] using the external scaling and `T_rot`."""

        a_A = self.a_eff * 1.0e8
        return 5.6e-9 * (self.T_rot / 100.0) * (self.rho / 3.0) ** (-2.0) * (a_A / 5.0) ** (-7.0)

    def x_ratio(self) -> float:
        """Ratio x used in alignment models.

        This intentionally matches the currently used external behavior:
        `x = depolarization_rate / absorption_rate`.
        """

        abs_r = self.absorption_rate()
        if abs_r == 0.0:
            return float("inf")
        return self.depolarization_rate() / abs_r

    def y_param(self, w: float = 1.0) -> float:
        """Parameter y = sqrt(2/10) * mu_anis * w."""

        return float(np.sqrt(0.2) * self.mu_anis * w)

    def sigma_JK(self, J: int, K: Any, T_int: float | None = None, w: float = 1.0):
        """Compute alignment efficiency sigma(J, K), interface-compatible with external code."""

        if J == 0:
            if np.isscalar(K):
                return 0.0
            return np.zeros_like(np.asarray(K, dtype=float))

        if T_int is None:
            T_int = self.critical_temperature()

        x_rat = self.x_ratio()
        y_par = self.y_param(w=w)

        gamma = (self.Brot - self.Arot) * h_CGS / (KB * T_int)

        K_vec = np.arange(0, J + 1, dtype=np.float64)
        exponent = gamma * K_vec**2
        max_exp = exponent[-1]
        rho_scaled = np.exp(exponent - max_exp)

        weights = np.full(len(K_vec), 2.0, dtype=np.float64)
        weights[0] = 1.0

        sum_rho = np.sum(rho_scaled * weights)
        x_vec = K_vec / J
        P2_vec = 0.5 * (3.0 * x_vec**2 - 1.0)
        avg_term = np.sum(y_par * P2_vec * rho_scaled * weights) / sum_rho

        K_in = np.asarray(K, dtype=np.float64)
        x_in = K_in / J
        P2_in = 0.5 * (3.0 * x_in**2 - 1.0)
        #BL: Corrected the overall alignment sign to match the revised formalism. 04/08/26
        sigma = (y_par * P2_in - avg_term) / (1.0 + x_rat)

        if np.isscalar(K):
            return float(np.asarray(sigma))
        return sigma

    def to_rotor_compat_dict(self, *, n_dust: float = 1.0) -> dict[str, float]:
        """Return a dictionary with fields expected by the VSGAlign bridge layer."""

        return {
            "n": float(n_dust),
            "A": float(self.Arot),
            "B": float(self.Brot),
            "mu_par": float(self.mu_par),
            "mu_perp": float(self.mu_perp),
            "T_rot": float(self.T_rot),
            "T_int": float(self.critical_temperature()),
        }
