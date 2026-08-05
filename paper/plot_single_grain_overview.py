"""
Paper figure: single-grain emissivity overview.

Produces eta_I(a), eta_Q(a), and p_Q = Q/I for a single grain
for three dipole configurations: parallel, perpendicular, shape-derived.

Individual figures: one per dipole type (black, no legend).
Combined figure:   all three dipoles overlaid in 3 shared panels.

Repository relation
-------------------
The script uses `DustGrain` to obtain the size-dependent rotational constants
and rotational temperature, then `params_from_dust` to convert that grain
model into the public rotor parameterization consumed by the emissivity code in
`src/rotor_emissivity` and `src/rotor_polarized_emissivity`.

Paper relation
--------------
This script visualizes the Sect. 5 pure-rotational emissivity pipeline from
`the companion VSG/AME manuscript` for a single grain.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace

from dust_properties.dust_alignment import DustGrain
from rotor_polarized_emissivity.dust_bridge import params_from_dust
from rotor_emissivity.impl_b import jnu_P_parallel, jnu_P_perp, jnu_Q_perp as jnu_Q
from rotor_polarized_emissivity.continuum_impl_b import (
    jnuq_P_parallel, jnuq_P_perp, jnuq_Q_perp as jnuq_Q,
)
from .common import GHZ, configure_matplotlib, ensure_figure_dir

configure_matplotlib()

# ---------------------------------------------------------------------------
# Dipole configurations
# ---------------------------------------------------------------------------
DIPOLE_CONFIGS = {
    'parallel': dict(mu_anis=1.0, label=r'parallel'),
    'perpendicular': dict(mu_anis=-0.5, label=r'perpendicular'),
    'shape': dict(mu_anis=None, label=r'shape-derived'),
}

# Colors for the combined figure
DIPOLE_COLORS = {
    'parallel': 'C0',
    'perpendicular': 'C1',
    'shape': 'C2',
}

NU_MAX_GHZ = 80.0


def compute_single_grain(a_angstrom=5.0, T_rot=100.0, mu_anis=None,
                          nu_min=1.0e9, nu_max=80.0e9, n_nu=400, sigma=None):
    """Return the Sect. 5 single-grain spectra for one dipole configuration."""
    a_cm = a_angstrom * 1e-8
    nu_grid = np.linspace(nu_min, nu_max, n_nu)

    # `DustGrain` pulls in the DL98/AHD09 temperature solver through
    # `spindust_trot`, while `params_from_dust` maps the result onto the public
    # emissivity API used by the manuscript-facing code.
    grain = DustGrain(a_eff=a_cm, T_gas=100.0, mu_anis=mu_anis)
    p = params_from_dust(grain, chi=np.pi / 2, sigma=sigma)
    p = replace(p, T_rot=float(T_rot))

    print(f"  a={a_angstrom} A, mu_anis={mu_anis}, "
          f"A={p.A:.3e} Hz, B={p.B:.3e} Hz, "
          f"T_rot={p.T_rot:.0f} K, T_int={p.T_int:.0f} K")

    I_nu = (jnu_P_parallel(nu_grid, p)
            + jnu_P_perp(nu_grid, p)
            + jnu_Q(nu_grid, p))
    Q_nu = (jnuq_P_parallel(nu_grid, p)
            + jnuq_P_perp(nu_grid, p)
            + jnuq_Q(nu_grid, p))

    with np.errstate(divide='ignore', invalid='ignore'):
        p_Q = np.where(I_nu > 0, Q_nu / I_nu, 0.0)
    return nu_grid, I_nu, Q_nu, p_Q


def _find_scale(data):
    """Return (exponent, scale) so that data/scale has O(1) max."""
    peak = np.max(np.abs(data))
    if peak == 0:
        return 0, 1.0
    exp = int(np.floor(np.log10(peak)))
    return exp, 10.0**exp


# ---------------------------------------------------------------------------
# Individual figures (one per dipole type)
# ---------------------------------------------------------------------------
def plot_individual(
    a_angstrom=5.0,
    T_rot=100.0,
    output_dir: str | Path | None = None,
    *,
    n_nu: int = 400,
    sigma=None,
):
    """Produce one 3-panel figure per dipole configuration."""
    figure_dir = ensure_figure_dir(output_dir)
    saved_paths: list[Path] = []

    for key, cfg in DIPOLE_CONFIGS.items():
        print(f'  Computing {key} ...')
        nu, I, Q, pQ = compute_single_grain(
            a_angstrom=a_angstrom,
            T_rot=T_rot,
            mu_anis=cfg['mu_anis'],
            n_nu=n_nu,
            sigma=sigma,
        )

        exp_I, sc_I = _find_scale(I)
        exp_Q, sc_Q = _find_scale(Q)

        fig, axes = plt.subplots(3, 1, figsize=(5, 6.5), sharex=True,
                                 gridspec_kw={'hspace': 0.08})
        ax_I, ax_Q, ax_p = axes

        ax_I.plot(nu / GHZ, I / sc_I, 'k-')
        ax_Q.plot(nu / GHZ, Q / sc_Q, 'k-')
        ax_p.plot(nu / GHZ, pQ * 100, 'k-')

        ax_I.set_ylabel(rf'$\eta_I(a) \times 10^{{{-exp_I}}}$')
        ax_Q.set_ylabel(rf'$\eta_Q(a) \times 10^{{{-exp_Q}}}$')
        ax_p.set_ylabel(r'$p_Q$ [\%]')
        ax_p.set_xlabel(r'$\nu$ [GHz]')

        for ax in axes:
            ax.grid(False)
            ax.set_xlim(0, NU_MAX_GHZ)
        for ax in (ax_I, ax_Q):
            ax.tick_params(labelbottom=False)

        fname = figure_dir / f'single_grain_{key}_a{a_angstrom}.pdf'
        fig.savefig(fname)
        print(f'  Saved {fname}')
        plt.close(fig)
        saved_paths.append(fname)

    return saved_paths


# ---------------------------------------------------------------------------
# Combined figure: all dipole types overlaid in 3 panels
# ---------------------------------------------------------------------------
def plot_combined(
    a_angstrom=5.0,
    T_rot=100.0,
    output_dir: str | Path | None = None,
    *,
    n_nu: int = 400,
    sigma=None,
):
    """3 rows (eta_I, eta_Q, p_Q) with all dipole curves overlaid."""
    figure_dir = ensure_figure_dir(output_dir)
    keys = list(DIPOLE_CONFIGS.keys())

    # Pre-compute all data
    data = {}
    for key in keys:
        cfg = DIPOLE_CONFIGS[key]
        print(f'  Computing {key} (combined) ...')
        data[key] = compute_single_grain(
            a_angstrom=a_angstrom,
            T_rot=T_rot,
            mu_anis=cfg['mu_anis'],
            n_nu=n_nu,
            sigma=sigma,
        )

    # Common scale per row
    all_I = np.concatenate([data[k][1] for k in keys])
    all_Q = np.concatenate([data[k][2] for k in keys])
    exp_I, sc_I = _find_scale(all_I)
    exp_Q, sc_Q = _find_scale(all_Q)

    fig, axes = plt.subplots(
        3, 1, figsize=(5, 6.5), sharex=True,
        gridspec_kw={'hspace': 0.08})
    ax_I, ax_Q, ax_p = axes

    for key in keys:
        cfg = DIPOLE_CONFIGS[key]
        color = DIPOLE_COLORS[key]
        nu, I, Q, pQ = data[key]

        ax_I.plot(nu / GHZ, I / sc_I, color=color, label=cfg['label'])
        ax_Q.plot(nu / GHZ, Q / sc_Q, color=color)
        ax_p.plot(nu / GHZ, pQ * 100, color=color)

    ax_I.set_ylabel(rf'$\eta_I(a) \times 10^{{{-exp_I}}}$')
    ax_Q.set_ylabel(rf'$\eta_Q(a) \times 10^{{{-exp_Q}}}$')
    ax_p.set_ylabel(r'$p_Q$ [%]')
    ax_p.set_xlabel(r'$\nu$ [GHz]')

    ax_I.legend(fontsize=10, frameon=False)

    for ax in axes:
        ax.grid(False)
        ax.set_xlim(0, NU_MAX_GHZ)
    for ax in (ax_I, ax_Q):
        ax.tick_params(labelbottom=False)

    fname = figure_dir / f'single_grain_combined_a{a_angstrom}.pdf'
    fig.savefig(fname)
    print(f'  Saved {fname}')
    plt.close(fig)
    return fname


def main(
    output_dir: str | Path | None = None,
    *,
    a: float = 5.0,
    T: float = 100.0,
    n_nu: int = 400,
    sigma=None,
):
    """Generate the single-grain overview figures used in the paper workflow."""
    print(f'Generating single-grain overview figures (a={a} A, T_rot={T} K)')
    print('--- Individual figures ---')
    saved_paths = plot_individual(
        a_angstrom=a,
        T_rot=T,
        output_dir=output_dir,
        n_nu=n_nu,
        sigma=sigma,
    )
    print('--- Combined figure ---')
    saved_paths.append(
        plot_combined(
            a_angstrom=a,
            T_rot=T,
            output_dir=output_dir,
            n_nu=n_nu,
            sigma=sigma,
        )
    )
    print('Done.')
    return saved_paths


if __name__ == '__main__':
    main()
