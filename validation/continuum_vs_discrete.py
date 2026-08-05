from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from dust_properties import DustGrain
from rotor_polarized_emissivity import compare_discrete_vs_continuum, params_from_dust
from rotor_polarized_emissivity.continuum_impl_a import (
    jnuq_P_parallel,
    jnuq_P_perp,
    jnuq_Q_perp,
    jnuq_total,
)
from rotor_polarized_emissivity.discrete import polarized_line_list, spectrum_binned_pol


GHZ = 1.0e9
BRANCHES = (
    ("total", jnuq_total, "Total"),
    ("P_parallel", jnuq_P_parallel, "P_parallel"),
    ("P_perp", jnuq_P_perp, "P_perp"),
    ("Q_perp", jnuq_Q_perp, "Q_perp"),
)


def generate_report(
    output_dir: str | Path,
    *,
    a_angstrom: float = 5.0,
    t_gas: float = 100.0,
    sigma: float = 0.1,
    n_nu: int = 100,
    jmax: int | None = None,
) -> dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    grain = DustGrain(a_eff=a_angstrom * 1e-8, T_gas=t_gas)
    params = params_from_dust(grain, chi=np.pi / 2, sigma=sigma)

    lines = polarized_line_list(params, Jmax=jmax)
    nu_grid = np.linspace(np.min(lines["nu"]), np.max(lines["nu"]), n_nu)
    discrete_spec = spectrum_binned_pol(lines, nu_grid)
    metrics = compare_discrete_vs_continuum(
        params,
        nu_grid,
        discrete_spec,
        impl="a",
        floor_rel=0.05,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.flatten()

    for ax, (branch, continuum_fn, title) in zip(axes, BRANCHES):
        continuum_values = continuum_fn(nu_grid, params)
        ax.plot(nu_grid / GHZ, discrete_spec[branch], label="Discrete", alpha=0.8)
        ax.plot(nu_grid / GHZ, continuum_values, "--", label="Continuum", lw=1.5)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("Emissivity")

    axes[0].legend(frameon=False)
    axes[2].set_xlabel("Frequency [GHz]")
    axes[3].set_xlabel("Frequency [GHz]")
    fig.suptitle(f"Continuum vs discrete validation ({a_angstrom:.1f} A, T_gas={t_gas:.0f} K)")
    fig.tight_layout()

    stem = f"continuum_vs_discrete_a{a_angstrom:.1f}_T{t_gas:.0f}"
    plot_path = output_path / f"{stem}.png"
    metrics_path = output_path / f"{stem}_metrics.json"

    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    serializable_metrics = {key: {name: float(value) for name, value in value_dict.items()} for key, value_dict in metrics.items()}
    metrics_path.write_text(json.dumps(serializable_metrics, indent=2))

    return {
        "plot_path": plot_path,
        "metrics_path": metrics_path,
        "metrics": serializable_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a continuum-vs-discrete validation plot.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/validation"))
    parser.add_argument("--a-angstrom", type=float, default=5.0)
    parser.add_argument("--t-gas", type=float, default=100.0)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--n-nu", type=int, default=100)
    parser.add_argument("--jmax", type=int, default=None)
    args = parser.parse_args()

    result = generate_report(
        args.output_dir,
        a_angstrom=args.a_angstrom,
        t_gas=args.t_gas,
        sigma=args.sigma,
        n_nu=args.n_nu,
        jmax=args.jmax,
    )

    print(result["plot_path"])
    print(result["metrics_path"])


if __name__ == "__main__":
    main()
