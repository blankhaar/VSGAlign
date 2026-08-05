"""Grain geometry and dipole-moment model.

DL98b (Draine & Lazarian 1998b; arXiv:astro-ph/9802239) derives most
coefficients for *spherical* grains, but then introduces geometric
substitutions for non-spherical (disk/cylinder) grains via:

- surface-equivalent radius a_s (DL98b eq. (2)),
- excitation-equivalent radius a_x (DL98b eq. (3)),
- inertia factor ξ = I / (0.4 M a^2).

Appendix A of DL98b provides explicit a_s, a_x, ξ for cylinders and disks
(their eqs. (A1)–(A8)).

This module implements those geometric factors and the DL98b dipole-moment
model μ^2 (DL98b eq. (11)).

Repository role
---------------
This package is not called directly by the paper figure scripts. Instead, it
is used by `dust_properties.dust_alignment.DustGrain`, which wraps these
DL98/AHD09 ingredients into the grain object consumed by the emissivity and
cross-section code.

Paper relation
--------------
In `the companion VSG/AME manuscript`, these geometric
ingredients provide the support needed for the size-dependent rotational
temperature input discussed in Sect. 2.2 below Eq. (4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import debye_esu_cm, e_esu, m_u, pi
from .units import as_ndarray, a_minus7

__all__ = [
    "GrainGeometry",
    "GrainModel",
]


@dataclass(frozen=True, slots=True)
class GrainGeometry:
    """Size-dependent geometric factors used by DL98b.

    All lengths are in cm.
    """

    a_cm: NDArray[np.float64]
    a_s_cm: NDArray[np.float64]
    a_x_cm: NDArray[np.float64]
    xi: NDArray[np.float64]
    mass_g: NDArray[np.float64]
    I_g_cm2: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GrainModel:
    """Geometric and dipole-moment assumptions for grains.

    Parameters match the DL98b default choices unless stated otherwise.
    The resulting geometry is one of the inputs used by the temperature solver
    that ultimately supplies `DustGrain.T_rot`.
    """

    rho_g_cm3: float = 2.0
    # DL98b uses d = 3.35e-8 cm (graphite interlayer spacing) as a fiducial thickness.
    thickness_cm: float = 3.35e-8
    # Shape transitions (DL98b often takes a1=0, a2=6e-8 cm so smallest grains are disks).
    a1_cm: float = 0.0
    a2_cm: float = 6.0e-8

    # Mean mass per atom for a carbonaceous grain (DL98b: m ≈ 9.25 amu for C:H≈3:1).
    mean_atom_mass_amu: float = 9.25
    # Intrinsic dipole per atom (Debye). DL98b uses β0≈0.4 D (their §3.4 and eq. (11)).
    beta_debye: float = 0.4
    # Charge-centroid displacement fraction |ε| (DL98b: ε≈0.1).
    epsilon_charge_centroid: float = 0.1

    def geometry(self, a_cm: ArrayLike) -> GrainGeometry:
        """Compute a_s, a_x, ξ, M, I for the given size(s) a (cm).

        Notes
        -----
        - DL98b Appendix A eqs. (A1)–(A4): cylinder (a < a1)
        - DL98b Appendix A eqs. (A5)–(A8): disk (a1 < a < a2)
        - DL98b Appendix A: sphere (a > a2): a_s=a_x=a, ξ=1
        """

        a = as_ndarray(a_cm)
        d = float(self.thickness_cm)

        # Preallocate arrays.
        a_s = np.empty_like(a)
        a_x = np.empty_like(a)
        xi = np.empty_like(a)

        is_cyl = a < float(self.a1_cm)
        is_disk = (~is_cyl) & (a < float(self.a2_cm))
        is_sph = ~is_cyl & ~is_disk

        # Cylinders: diameter d, length 2b (DL98b App. A eq. (A1)–(A4)).
        if np.any(is_cyl):
            a_c = a[is_cyl]
            b = (8.0 * a_c**3) / (3.0 * d**2)  # (A1)
            a_s[is_cyl] = np.sqrt((b * d) / 2.0 + (d**2) / 8.0)  # (A2)
            a_x[is_cyl] = (
                (b**3 * d) / 6.0
                + (b**2 * d**2) / 8.0
                + (b * d**3) / 8.0
                + (d**4) / 64.0
            ) ** 0.25  # (A3)
            xi[is_cyl] = (160.0 / 27.0) * (a_c / d) ** 4  # (A4)

        # Disks: thickness d, radius b (DL98b App. A eq. (A5)–(A8)).
        if np.any(is_disk):
            a_d = a[is_disk]
            b = np.sqrt((4.0 * a_d**3) / (3.0 * d))  # (A5)
            a_s[is_disk] = np.sqrt((b**2) / 2.0 + (b * d) / 2.0)  # (A6)
            a_x[is_disk] = (
                (b**4) / 4.0
                + (b**3 * d) / 2.0
                + (b**2 * d**2) / 8.0
                + (b * d**3) / 24.0
            ) ** 0.25  # (A7)
            xi[is_disk] = (5.0 * a_d) / (3.0 * d)  # (A8)

        # Spheres: a_s=a_x=a, ξ=1.
        if np.any(is_sph):
            a_s[is_sph] = a[is_sph]
            a_x[is_sph] = a[is_sph]
            xi[is_sph] = 1.0

        # Mass M = (4π/3) ρ a^3.
        mass = (4.0 * pi / 3.0) * float(self.rho_g_cm3) * a**3
        # Largest principal moment I = ξ (0.4 M a^2) by definition of ξ.
        I = xi * 0.4 * mass * a**2

        return GrainGeometry(a_cm=a, a_s_cm=a_s, a_x_cm=a_x, xi=xi, mass_g=mass, I_g_cm2=I)

    def mean_atom_count(self, a_cm: ArrayLike) -> NDArray[np.float64]:
        """Estimate number of atoms N in a grain of size a (DL98b §3.1).

        DL98b gives an approximate scaling N ≈ 545 a_-7^3 for ρ≈2 g cm^-3 and
        mean atomic mass ≈9.25 amu. Here we compute the same quantity from the
        explicit mass and mean atom mass.
        """

        a = as_ndarray(a_cm)
        mass = (4.0 * pi / 3.0) * float(self.rho_g_cm3) * a**3
        m_atom = float(self.mean_atom_mass_amu) * m_u
        return mass / m_atom

    def mu2_cgs(self, a_cm: ArrayLike, mean_Z2: ArrayLike) -> NDArray[np.float64]:
        """Return μ^2 in (esu*cm)^2 using the DL98b dipole model (DL98b eq. (11)).

        Parameters
        ----------
        a_cm
            Volume-equivalent radius a in cm.
        mean_Z2
            Mean square grain charge ⟨Z^2⟩ for that size/environment.

        Notes
        -----
        DL98b eq. (11) is expressed in Debye; we implement the same model but
        return μ^2 in cgs units for direct use in DL98b plasma-drag and dipole-
        damping expressions. In the current repository this quantity influences
        the temperature solver, which then feeds the Sect. 5 emissivity
        calculations through `DustGrain`.
        """

        geom = self.geometry(a_cm)
        a7 = a_minus7(geom.a_cm)
        z2 = as_ndarray(mean_Z2)

        # Charge-centroid term: (ε Z e a_x)^2, averaged to ⟨Z^2⟩.
        mu_charge_debye2 = ((self.epsilon_charge_centroid * e_esu * geom.a_x_cm) / debye_esu_cm) ** 2 * z2

        # Intrinsic term: μ_i ≈ sqrt(N) β (DL98b eq. (9)), so μ_i^2 ≈ N β^2.
        N = self.mean_atom_count(geom.a_cm)
        mu_intrinsic_debye2 = N * float(self.beta_debye) ** 2

        mu2_debye2 = mu_charge_debye2 + mu_intrinsic_debye2
        return mu2_debye2 * (debye_esu_cm**2)
