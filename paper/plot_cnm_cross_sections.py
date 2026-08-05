"""
Paper figure: CNM cross sections per H nucleon comparison.

Compares three absorption cross sections (cm^2 per H) vs frequency:
  (i)   spinning dust (rotational): size-distribution-integrated using
        rotor_cross_section with first-principles T_rot from DustGrain,
  (ii)  vibrational dust opacity: Planck diffuse-ISM power-law fit,
  (iii) free-free absorption: fiducial CNM electron fraction & temperature.

Adapted from nano_notes/paper/cnm_cross_section_comparison.py to use the
ism-newang rotor_cross_section machinery and DustGrain's first-principles
rotational temperature.

Output: paper/figures/cnm_cross_sections_comparison.pdf

Repository relation
-------------------
This script is the clearest example of the full production path:
`spindust_trot` computes the rotational-temperature ingredients used by
`DustGrain`, `params_from_dust` maps them onto the public rotor parameter set,
and `rotor_cross_section` evaluates the continuum spinning-dust absorption.

Paper relation
--------------
It supports the cross-section and absorption comparison discussed alongside
Sect. 6 of `the companion VSG/AME manuscript`.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from dust_properties.dust_alignment import DustGrain
from rotor_polarized_emissivity.dust_bridge import params_from_dust
from rotor_cross_section.continuum import sigma_nu_total
from .common import configure_matplotlib, ensure_figure_dir

configure_matplotlib(
    **{
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 17,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
    }
)

h_cgs = 6.62607015e-27   # erg s
k_B_cgs = 1.380649e-16   # erg K^-1
c_cgs = 2.99792458e10    # cm s^-1


@dataclass(frozen=True)
class CNMFiducials:
    """Cold neutral medium gas parameters."""
    n_H: float = 30.0      # cm^-3
    T_e: float = 100.0     # K
    x_e: float = 3.0e-4    # n_e / n_H
    Z_i: int = 1


@dataclass(frozen=True)
class DustOpacityFiducials:
    """Planck Intermediate Results XVII diffuse-gas vibrational opacity."""
    sigma_353: float = 7.1e-27  # cm^2 per H at 353 GHz
    beta_mm: float = 1.53


# Size-distribution parameters shared by the spinning-dust contribution.
A_MIN_CM = 3.5e-8   # 3.5 Angstrom
A_MAX_CM = 15.0e-8  # 15 Angstrom (= 1.5 nm)
A0_CM = 3.5e-8
SIGMA_LN = 0.4
C_LN = 1.3e-6       # normalization [cm^-1 H^-1] in dn/d(ln a)


def log_normal_dnda(a_cm, a0_cm=A0_CM, sigma=SIGMA_LN):
    """Return the log-normal grain-size distribution used in the CNM comparison."""
    if a_cm <= 0:
        return 0.0
    log_term = np.log(a_cm / a0_cm)
    return (C_LN / a_cm) * np.exp(-(log_term**2) / (2 * sigma**2))


def sigma_spin_per_H(nu_hz, T_gas=100.0, n_H=30.0, N_a=30):
    """Return the size-integrated spinning-dust cross section per H nucleon."""
    a_grid = np.logspace(np.log10(A_MIN_CM), np.log10(A_MAX_CM), N_a)
    nu_hz = np.asarray(nu_hz, dtype=float)
    result = np.zeros_like(nu_hz)

    def sigma_grain(a_cm):
        # Publication path: DL98/AHD09 temperature backend -> `DustGrain` ->
        # `params_from_dust` -> public cross-section solver.
        grain = DustGrain(a_eff=a_cm, T_gas=T_gas, n_H=n_H, mu_anis=None)
        p = params_from_dust(grain, chi=0.0)
        return sigma_nu_total(nu_hz, p)

    sig_prev = sigma_grain(a_grid[0])
    dnda_prev = log_normal_dnda(a_grid[0])

    for i in range(1, N_a):
        a_curr = a_grid[i]
        sig_curr = sigma_grain(a_curr)
        dnda_curr = log_normal_dnda(a_curr)

        da = a_curr - a_grid[i - 1]
        result += 0.5 * (sig_prev * dnda_prev + sig_curr * dnda_curr) * da

        sig_prev = sig_curr
        dnda_prev = dnda_curr

        print(f'    grain {i}/{N_a-1}: a = {a_curr*1e8:.1f} A, '
              f'T_rot = {DustGrain(a_eff=a_curr, T_gas=T_gas, n_H=n_H, mu_anis=None).T_rot:.1f} K')

    return result


def sigma_dust_vibrational_per_H(nu_GHz, d=DustOpacityFiducials()):
    """Planck diffuse-ISM dust opacity per H: power law from 353 GHz."""
    return d.sigma_353 * (np.asarray(nu_GHz, dtype=float) / 353.0) ** d.beta_mm


def gaunt_factor_oster(nu_GHz, T_e):
    """Radio Gaunt factor: g_ff = max(1, 1.5 ln T - ln nu_GHz - 5.307)."""
    return np.maximum(1.0, 1.5 * np.log(T_e) - np.log(np.asarray(nu_GHz)) - 5.307)


def sigma_freefree_per_H(nu_hz, cnm=CNMFiducials()):
    """
    Free-free absorption cross section per H nucleon.
    alpha_ff = 3.692e8 T^{-1/2} Z^2 n_e n_i nu^{-3} (1 - exp(-hnu/kT)) g_ff
    sigma_ff = alpha_ff / n_H
    """
    nu = np.asarray(nu_hz, dtype=float)
    nu_GHz = nu * 1e-9

    n_e = cnm.x_e * cnm.n_H
    n_i = n_e

    g_ff = gaunt_factor_oster(nu_GHz, cnm.T_e)
    alpha = (3.692e8 * cnm.T_e**(-0.5) * cnm.Z_i**2 * n_e * n_i
             * nu**(-3) * (1.0 - np.exp(-h_cgs * nu / (k_B_cgs * cnm.T_e))) * g_ff)
    return alpha / cnm.n_H


def main(output_dir: str | Path | None = None):
    """Generate the CNM cross-section comparison figure for the paper."""
    figure_dir = ensure_figure_dir(output_dir)
    warnings.filterwarnings('ignore', category=UserWarning)
    from scipy.integrate import IntegrationWarning
    warnings.filterwarnings('ignore', category=IntegrationWarning)

    nu_GHz = np.logspace(np.log10(0.1), np.log10(500.0), 200)
    nu_hz = nu_GHz * 1e9

    print('Computing spinning dust cross section (first-principles T_rot) ...')
    sigma_spin = sigma_spin_per_H(nu_hz)

    print('Computing vibrational dust opacity ...')
    sigma_vib = sigma_dust_vibrational_per_H(nu_GHz)

    print('Computing free-free cross section ...')
    sigma_ff = sigma_freefree_per_H(nu_hz)

    from matplotlib.ticker import FixedLocator

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 4.8))
    ax.plot(nu_GHz, sigma_spin, color='black', lw=1.8, label='Spinning dust (rot.)')
    ax.plot(nu_GHz, sigma_vib, color='tab:orange', lw=1.8, label='Dust (vibrational)')
    ax.plot(nu_GHz, sigma_ff, color='tab:blue', lw=1.8, label=r'Free$-$free')

    ax.set_xscale('log')
    ax.set_xlim(0.1, 500.0)
    ax.set_xlabel(r'$\nu$ [GHz]')
    ax.set_ylabel(r'$\sigma$ [cm$^{2}$ H$^{-1}$]')

    ax.set_ylim(0.0, 0.5e-26)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(17)

    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100]))

    ax.legend(frameon=False, loc='best')
    fig.tight_layout()

    out_path = figure_dir / 'cnm_cross_sections_comparison.pdf'
    fig.savefig(out_path)
    print(f'Saved {out_path}')
    plt.close(fig)
    return out_path


if __name__ == '__main__':
    main()
