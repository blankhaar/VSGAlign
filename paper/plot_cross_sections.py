"""
Paper figure: spinning dust absorption cross sections.

Total cross section for 3.5, 5, 7 and 10 Angstrom grains (shape-derived
dipole, T_rot=100 K, T_int from grain size), overlaid with the vibrational
absorption cross section C_abs^vib for each grain size (dashed, same color).

Repository relation
-------------------
This script exercises the same `DustGrain -> params_from_dust` bridge as the
emissivity figures, but sends the resulting parameters into the public
cross-section API in `src/rotor_cross_section`.

Paper relation
--------------
It supports the absorption-side discussion connected to Sect. 6 of
`the companion VSG/AME manuscript`.
"""

from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace

from dust_properties.dust_alignment import DustGrain
from rotor_polarized_emissivity.dust_bridge import params_from_dust
from rotor_cross_section.continuum import sigma_nu_total
from .common import GHZ, configure_matplotlib, ensure_figure_dir

configure_matplotlib()

# ---------------------------------------------------------------------------
# Grain sizes and colors
# ---------------------------------------------------------------------------
GRAIN_SIZES_ANGSTROM = [3.5, 5.0, 7.0, 10.0]
GRAIN_COLORS = {3.5: 'C0', 5.0: 'C1', 7.0: 'C2', 10.0: 'C3'}

c_cm = 2.99792458e10


def vibrational_cross_section(nu, a_cm):
    """Return the approximate vibrational absorption cross section used in the figure."""
    lam_cm = c_cm / nu
    lam_um = lam_cm * 1e4
    term_lam = (100.0 / lam_um)**2
    term_a = (a_cm / 1e-7)**3
    return 9e-19 * term_lam * term_a


def _find_scale(data):
    """Return (exponent, scale) so that data/scale has O(1) max."""
    peak = np.max(np.abs(data))
    if peak == 0:
        return 0, 1.0
    exp = int(np.floor(np.log10(peak)))
    return exp, 10.0**exp


def main(output_dir: str | Path | None = None):
    """Generate the per-grain rotational and vibrational cross-section figures."""
    figure_dir = ensure_figure_dir(output_dir)
    warnings.filterwarnings('ignore', category=UserWarning)
    from scipy.integrate import IntegrationWarning
    warnings.filterwarnings('ignore', category=IntegrationWarning)

    nu_grid = np.logspace(np.log10(0.1e9), np.log10(500.0e9), 400)

    sigma_spin = {}
    sigma_vib = {}
    for a_ang in GRAIN_SIZES_ANGSTROM:
        a_cm = a_ang * 1e-8
        print(f'  Computing a = {a_ang} A ...')
        # `DustGrain` carries the DL98/AHD09 temperature machinery, while the
        # cross section itself is evaluated by the public continuum solver.
        grain = DustGrain(a_eff=a_cm, T_gas=100.0, mu_anis=None)
        p = params_from_dust(grain, chi=0.0)
        p = replace(p, T_rot=100.0)

        sigma_spin[a_ang] = sigma_nu_total(nu_grid, p)
        sigma_vib[a_ang] = vibrational_cross_section(nu_grid, a_cm)

    fig, ax = plt.subplots(figsize=(5, 3.8))

    for a_ang in GRAIN_SIZES_ANGSTROM:
        color = GRAIN_COLORS[a_ang]
        label = rf'{a_ang} $\AA$'
        ax.plot(nu_grid / GHZ, sigma_spin[a_ang] ,
                color=color, ls='-', label=label)
        ax.plot(nu_grid / GHZ, sigma_vib[a_ang] ,
                color=color, ls='--')

    ax.set_xlabel(r'$\nu$ [GHz]')
    ax.set_ylabel(r'$\sigma$ [cm$^2$]')
    ax.set_xlim(0, 140)
    ax.legend(fontsize=10, frameon=False)
    ax.grid(False)

    fname = figure_dir / 'cross_sections.pdf'
    fig.savefig(fname)
    print(f'  Saved {fname}')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.8))

    for a_ang in GRAIN_SIZES_ANGSTROM:
        color = GRAIN_COLORS[a_ang]
        label = rf'{a_ang} $\AA$'
        ax.semilogx(nu_grid / GHZ, sigma_spin[a_ang] ,
                     color=color, ls='-', label=label)
        ax.semilogx(nu_grid / GHZ, sigma_vib[a_ang] ,
                     color=color, ls='--')

    ax.set_xlabel(r'$\nu$ [GHz]')
    ax.set_ylabel(r'$\sigma$ [cm$^2$]')
    ax.set_xlim(0.7, 500)
    ymax_spin = max(np.max(sigma_spin[a] ) for a in GRAIN_SIZES_ANGSTROM)
    ax.set_ylim(bottom=0, top=ymax_spin * 1.1)
    ax.legend(fontsize=10, frameon=False)
    ax.grid(False)

    fname_log = figure_dir / 'cross_sections_logx.pdf'
    fig.savefig(fname_log)
    print(f'  Saved {fname_log}')
    plt.close(fig)
    print('Done.')
    return [fname, fname_log]


if __name__ == '__main__':
    main()
