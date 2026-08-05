"""Environment definitions for spinning-dust rotation.

The DL98b coefficients (Draine & Lazarian 1998b; arXiv:astro-ph/9802239)
depend on gas temperature, density, chemical/ion composition, and the
radiation field intensity (for IR emission terms).

We keep the *environment* definition deliberately lightweight:

- you may provide explicit neutral/ion species lists; otherwise, a simple
  default H/H2/He + (H+, M+) composition is constructed from fractions.
- grain charge distributions and photoelectric emission rates are treated
  as *inputs* (the detailed charging model is outside the scope of this
  single-T implementation).

Within this repository an `Environment` is typically assembled indirectly by
`dust_properties.dust_alignment.DustGrain`. The paper scripts then inherit the
chosen CNM-like defaults through the `DustGrain` objects they instantiate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from .charge import ChargeDistribution
from .constants import amu_to_g, m_H

__all__ = [
    "NeutralSpecies",
    "IonSpecies",
    "Environment",
    "environment_preset",
]


@dataclass(frozen=True, slots=True)
class NeutralSpecies:
    """Neutral collision partner for DL98b sums (e.g. H, H2, He).

    Parameters
    ----------
    name
        Display name.
    n_cm3
        Number density (cm^-3).
    mass_g
        Mass (g).
    polarizability_cm3
        Static electric polarizability (cm^3), used in DL98b eq. (23).
    """

    name: str
    n_cm3: float
    mass_g: float
    polarizability_cm3: float


@dataclass(frozen=True, slots=True)
class IonSpecies:
    """Ion collision partner for DL98b sums (e.g. H+, C+ treated as 'M+').

    The ion-impact formulas (DL98b eqs. (20), (40)–(41)) and plasma-drag
    formulas (DL98b eqs. (25), (43)) require ion density, mass, and charge.

    For `G_i^(ev)` (DL98b eq. (41)), one also needs the polarizability of
    the neutral species produced after neutralization of the incoming ion.
    """

    name: str
    n_cm3: float
    mass_g: float
    Z: int
    neutral_polarizability_cm3: float


ChargeDistProvider = Callable[[float], ChargeDistribution]
PhotoRateProvider = Callable[[float, int], float]
PhotoEnergyProvider = Callable[[float, int], float]
RecombSitesProvider = Callable[[float], float]


def _default_recomb_sites(a_cm: float) -> float:
    # DL98b eq. (54): N_r = 1 + (a / (2e-7 cm))^2
    return 1.0 + (a_cm / (2.0e-7)) ** 2


@dataclass(frozen=True, slots=True)
class Environment:
    """Gas + radiation environment required by the DL98b single-T model.

    Parameters
    ----------
    n_H
        Hydrogen nuclei density `n_H` in cm^-3 (DL98b "H nucleon density").
    T_gas
        Gas kinetic temperature in K.
    T_dust
        Grain temperature in K (used as evaporation temperature `T_ev` in
        DL98b excitation terms; see DL98b eq. (38) and eqs. (41)).
        If None, defaults to `T_gas`.
    chi
        Radiation field intensity scaling `χ = u_*/u_ISRF` (DL98b eqs. (30)–(33),
        (44)–(46)).
    y
        Molecular fraction parameter `y = 2 n(H2) / n_H` (DL98b eq. (48) and
        systematic-torque terms). Must be in [0, 1].
    x_H
        Ion fraction for H+ (taken as `n(H+)/n_H`).
    x_M
        Ion fraction for a representative metal ion M+ (taken as `n(M+)/n_H`).
    he_fraction
        Helium abundance by number relative to `n_H` (DL98b often uses 0.1).
    neutrals, ions
        Optional explicit species lists. If omitted, a default composition is built
        from `y`, `x_H`, `x_M`, and `he_fraction`.
    n_e
        Electron density (cm^-3) used for the Debye length in plasma drag.
        If None, defaults to `n_e = Σ_i Z_i n_i` (singly-ionized default: (x_H+x_M)*n_H).
    cos2_psi
        Value of cos^2 Ψ (angle between μ and ω) used in plasma drag logs.
        DL98b uses 1/3 for random orientations.
    charge_distribution
        Provider for the grain charge distribution f(Z_g). The default assumes a
        neutral grain: delta(Z=0).
    photoelectron_rate
        Provider for photoelectron emission rate Ṅ_pe(Z_g) in s^-1 (DL98b eq. (47)).
        If None, photoelectric excitation is treated as zero unless overridden.
    photoelectron_mean_energy_erg
        Provider for mean photoelectron kinetic energy at infinity ⟨E_pe⟩ in erg.
        If None, treated as 0 unless overridden.
    h2_random_gamma
        γ for random H2 formation excitation (DL98b eq. (48)).
    h2_random_Ef_eV
        Translational kinetic energy E_f (eV) for random H2 formation (DL98b eq. (48)).
    h2_random_J2
        Mean ⟨J(J+1)⟩ for nascent H2 (DL98b eq. (48)).
    h2_systematic_gamma
        γ for systematic H2-formation torques (DL98b eq. (52)).
    h2_systematic_Ef_eV
        Translational kinetic energy parameter E_f (eV) in DL98b eq. (52).
    recombination_sites
        Provider for N_r(a) (DL98b eq. (54) default).
    """

    n_H: float
    T_gas: float
    T_dust: float | None = None
    chi: float = 1.0

    y: float = 0.0
    x_H: float = 0.0
    x_M: float = 0.0
    he_fraction: float = 0.1

    neutrals: tuple[NeutralSpecies, ...] = ()
    ions: tuple[IonSpecies, ...] = ()
    n_e: float | None = None

    cos2_psi: float = 1.0 / 3.0

    # Either a constant distribution (common in tests / simplified runs) or a provider.
    charge_distribution: ChargeDistribution | ChargeDistProvider = field(
        default_factory=lambda: ChargeDistribution.delta(0)
    )

    photoelectron_rate: PhotoRateProvider | None = None
    photoelectron_mean_energy_erg: PhotoEnergyProvider | None = None

    h2_random_gamma: float = 0.0
    h2_random_Ef_eV: float = 0.2
    h2_random_J2: float = 0.0

    h2_systematic_gamma: float = 0.0
    h2_systematic_Ef_eV: float = 0.2
    recombination_sites: RecombSitesProvider = _default_recomb_sites

    def __post_init__(self) -> None:
        if not (0.0 <= self.y <= 1.0):
            raise ValueError("Environment.y must be in [0, 1].")
        if self.T_dust is None:
            object.__setattr__(self, "T_dust", float(self.T_gas))

        if not self.neutrals:
            object.__setattr__(self, "neutrals", tuple(_build_default_neutrals(self)))
        if not self.ions:
            object.__setattr__(self, "ions", tuple(_build_default_ions(self)))

        if self.n_e is None:
            ne = sum(float(ion.Z) * float(ion.n_cm3) for ion in self.ions)
            object.__setattr__(self, "n_e", ne)

    def charge_dist(self, a_cm: float) -> ChargeDistribution:
        """Return the grain charge distribution f(Z_g) for size `a_cm`."""

        cd = self.charge_distribution
        if isinstance(cd, ChargeDistribution):
            return cd
        return cd(float(a_cm))

    @property
    def n_H2(self) -> float:
        """Return n(H2) in cm^-3 from `y = 2 n(H2) / n_H`."""

        return 0.5 * self.y * self.n_H

    @property
    def n_Hplus(self) -> float:
        return self.x_H * self.n_H

    @property
    def n_H0(self) -> float:
        """Return neutral atomic hydrogen density n(H).

        Using:
          n_H = n(H) + 2 n(H2) + n(H+)
        and y = 2 n(H2) / n_H, x_H = n(H+)/n_H.
        """

        return self.n_H * (1.0 - self.y - self.x_H)


def _build_default_neutrals(env: Environment) -> list[NeutralSpecies]:
    # Default polarizabilities (cm^3). These matter only for charged-grain
    # induced-dipole terms in the DL98 backend; users may override them when a
    # different physical environment is needed.
    alpha_H = 0.667e-24
    alpha_H2 = 0.802e-24
    alpha_He = 0.205e-24

    neutrals: list[NeutralSpecies] = []
    nH0 = env.n_H0
    if nH0 > 0:
        neutrals.append(
            NeutralSpecies(name="H", n_cm3=nH0, mass_g=m_H, polarizability_cm3=alpha_H)
        )
    nH2 = env.n_H2
    if nH2 > 0:
        neutrals.append(
            NeutralSpecies(
                name="H2",
                n_cm3=nH2,
                mass_g=amu_to_g(2.01588),
                polarizability_cm3=alpha_H2,
            )
        )
    nHe = env.he_fraction * env.n_H
    if nHe > 0:
        neutrals.append(
            NeutralSpecies(
                name="He",
                n_cm3=nHe,
                mass_g=amu_to_g(4.002602),
                polarizability_cm3=alpha_He,
            )
        )
    return neutrals


def _build_default_ions(env: Environment) -> list[IonSpecies]:
    # Representative ion list: H+ and "M+" (use carbon as a proxy).
    # These defaults are sufficient for the publication figures, but callers
    # can supply an explicit species list when reproducing a different medium.
    alpha_H = 0.667e-24
    alpha_C = 1.76e-24

    ions: list[IonSpecies] = []
    nHp = env.x_H * env.n_H
    if nHp > 0:
        ions.append(
            IonSpecies(
                name="H+",
                n_cm3=nHp,
                mass_g=m_H,
                Z=1,
                neutral_polarizability_cm3=alpha_H,
            )
        )
    nMp = env.x_M * env.n_H
    if nMp > 0:
        ions.append(
            IonSpecies(
                name="M+",
                n_cm3=nMp,
                mass_g=amu_to_g(12.0107),
                Z=1,
                neutral_polarizability_cm3=alpha_C,
            )
        )
    return ions


_PRESETS: Mapping[str, dict[str, float]] = {
    # Hoang, Draine & Lazarian (2010; arXiv:1003.2638) Table 1 values are used
    # as convenient defaults for CLI usage/testing. They are broadly consistent
    # with the classic DL98b phase definitions.
    "CNM": dict(n_H=30.0, T_gas=100.0, T_dust=20.0, chi=1.0, x_H=0.0012, x_M=0.0003, y=0.0),
    "WNM": dict(n_H=0.4, T_gas=6000.0, T_dust=20.0, chi=1.0, x_H=0.1, x_M=0.0003, y=0.0),
    "WIM": dict(n_H=0.1, T_gas=8000.0, T_dust=20.0, chi=1.0, x_H=0.99, x_M=0.001, y=0.0),
    "RN": dict(n_H=1.0e3, T_gas=100.0, T_dust=40.0, chi=1000.0, x_H=0.001, x_M=0.0002, y=0.01),
    "PDR": dict(n_H=1.0e5, T_gas=1000.0, T_dust=80.0, chi=30000.0, x_H=0.0001, x_M=0.0002, y=0.01),
}


def environment_preset(name: str) -> Environment:
    """Construct an `Environment` from a named preset.

    Preset names currently include: CNM, WNM, WIM, RN, PDR.
    """

    key = name.strip().upper()
    if key not in _PRESETS:
        raise KeyError(f"Unknown preset {name!r}. Available: {sorted(_PRESETS)}")
    return Environment(**_PRESETS[key])
