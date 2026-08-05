"""
Paper figure: size-distribution-integrated emissivity.

Produces eta_I, eta_Q, and p_Q = Q/I integrated over a log-normal grain
size distribution for three dipole configurations: parallel, perpendicular,
shape-derived.

Individual figures: one per dipole type (black, no legend).
Combined figure:   all three dipoles overlaid in 3 shared panels.

Repository relation
-------------------
This script is the size-distribution counterpart of
`paper.plot_single_grain_overview`: it constructs `DustGrain` models, bridges
them through `params_from_dust`, and then integrates the public emissivity
functions over a grain-size distribution.

Paper relation
--------------
This figure corresponds to the size-integrated Sect. 5 emissivity calculations
described in `the companion VSG/AME manuscript`.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from dust_properties.dust_alignment import DustGrain
from rotor_polarized_emissivity.dust_bridge import params_from_dust
from rotor_emissivity.impl_b import jnu_total
from rotor_polarized_emissivity.continuum_impl_b import jnuq_total
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

NU_MAX_GHZ = 70.0


# ---------------------------------------------------------------------------
# Log-normal size distribution
# ---------------------------------------------------------------------------
def log_normal_dnda(a_cm, a0_cm=3.5e-8, sigma=0.4):
    """Return the log-normal grain-size distribution used in the paper runs."""
    if a_cm <= 0:
        return 0.0
    const = 1.3e-6
    log_term = np.log(a_cm / a0_cm)
    exponent = -(log_term**2) / (2 * sigma**2)
    return (const / a_cm) * np.exp(exponent)


# ---------------------------------------------------------------------------
# Compute size-distribution-integrated emissivity
# ---------------------------------------------------------------------------
def compute_size_distribution(mu_anis=None, T_gas=100.0,
                              nu_min=1.0e9, nu_max=140.0e9, n_nu=300,
                              a_min=3.5e-8, a_max=10.0e-8, N_a=50):
    """Return the Sect. 5 spectra integrated over the adopted size distribution."""
    nu_grid = np.linspace(nu_min, nu_max, n_nu)
    a_grid = np.logspace(np.log10(a_min), np.log10(a_max), N_a)

    J_I = np.zeros_like(nu_grid)
    J_Q = np.zeros_like(nu_grid)

    def get_j_nu_grain(a_val):
        # This is the same production path used elsewhere in the repository:
        # `spindust_trot` -> `DustGrain` -> `params_from_dust` -> public
        # emissivity functions.
        grain = DustGrain(a_eff=a_val, T_gas=T_gas, mu_anis=mu_anis)
        p = params_from_dust(grain, chi=np.pi / 2)
        j_I = jnu_total(nu_grid, p)
        j_Q = jnuq_total(nu_grid, p)
        return j_I, j_Q

    # Trapezoidal integration over grain sizes
    j_I_prev, j_Q_prev = get_j_nu_grain(a_grid[0])
    dnda_prev = log_normal_dnda(a_grid[0])

    for i in range(1, N_a):
        a_curr = a_grid[i]
        j_I_curr, j_Q_curr = get_j_nu_grain(a_curr)
        dnda_curr = log_normal_dnda(a_curr)

        da = a_curr - a_grid[i - 1]

        J_I += 0.5 * (j_I_prev * dnda_prev + j_I_curr * dnda_curr) * da
        J_Q += 0.5 * (j_Q_prev * dnda_prev + j_Q_curr * dnda_curr) * da

        j_I_prev, j_Q_prev = j_I_curr, j_Q_curr
        dnda_prev = dnda_curr

    with np.errstate(divide='ignore', invalid='ignore'):
        p_Q = np.where(J_I > 0, J_Q / J_I, 0.0)

    return nu_grid, J_I, J_Q, p_Q


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
def plot_individual(data, output_dir: str | Path | None = None):
    """Produce one 3-panel figure per dipole configuration."""
    figure_dir = ensure_figure_dir(output_dir)
    saved_paths: list[Path] = []

    for key, cfg in DIPOLE_CONFIGS.items():
        nu, I, Q, pQ = data[key]

        exp_I, sc_I = _find_scale(I)
        exp_Q, sc_Q = _find_scale(Q)

        fig, axes = plt.subplots(3, 1, figsize=(5, 6.5), sharex=True,
                                 gridspec_kw={'hspace': 0.08})
        ax_I, ax_Q, ax_p = axes

        ax_I.plot(nu / GHZ, I / sc_I, 'k-')
        ax_Q.plot(nu / GHZ, Q / sc_Q, 'k-')
        ax_p.plot(nu / GHZ, pQ * 100, 'k-')

        ax_I.set_ylabel(rf'$\eta_I \times 10^{{{-exp_I}}}$')
        ax_Q.set_ylabel(rf'$\eta_Q \times 10^{{{-exp_Q}}}$')
        ax_p.set_ylabel(r'$p_Q$ [%]')
        ax_p.set_xlabel(r'$\nu$ [GHz]')

        ax_p.set_ylim(bottom=-3.5)

        for ax in axes:
            ax.grid(False)
            ax.set_xlim(0, NU_MAX_GHZ)
        for ax in (ax_I, ax_Q):
            ax.tick_params(labelbottom=False)

        fname = figure_dir / f'size_dist_{key}.pdf'
        fig.savefig(fname)
        print(f'  Saved {fname}')
        plt.close(fig)
        saved_paths.append(fname)

    return saved_paths


# ---------------------------------------------------------------------------
# Combined figure: all dipole types overlaid in 3 panels
# ---------------------------------------------------------------------------
def plot_combined(data, output_dir: str | Path | None = None):
    """3 rows (eta_I, eta_Q, p_Q) with all dipole curves overlaid."""
    figure_dir = ensure_figure_dir(output_dir)
    keys = list(DIPOLE_CONFIGS.keys())

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

    ax_I.set_ylabel(rf'$\eta_I \times 10^{{{-exp_I}}}$')
    ax_Q.set_ylabel(rf'$\eta_Q \times 10^{{{-exp_Q}}}$')
    ax_p.set_ylabel(r'$p_Q$ [%]')
    ax_p.set_xlabel(r'$\nu$ [GHz]')

    ax_I.legend(fontsize=10, frameon=False)
    ax_p.set_ylim(bottom=-3.5)

    for ax in axes:
        ax.grid(False)
        ax.set_xlim(0, NU_MAX_GHZ)
    for ax in (ax_I, ax_Q):
        ax.tick_params(labelbottom=False)

    fname = figure_dir / 'size_dist_combined.pdf'
    fig.savefig(fname)
    print(f'  Saved {fname}')
    plt.close(fig)
    return fname


def main(output_dir: str | Path | None = None):
    """Generate the size-distribution emissivity figures for the paper."""
    data = {}
    for key, cfg in DIPOLE_CONFIGS.items():
        print(f'  Computing {key} ...')
        data[key] = compute_size_distribution(mu_anis=cfg['mu_anis'])

    print('--- Individual figures ---')
    saved_paths = plot_individual(data, output_dir=output_dir)
    print('--- Combined figure ---')
    saved_paths.append(plot_combined(data, output_dir=output_dir))
    print('Done.')
    return saved_paths


if __name__ == '__main__':
    main()
