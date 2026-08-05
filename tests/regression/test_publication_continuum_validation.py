import numpy as np

from dust_properties.dust_alignment import DustGrain
from rotor_polarized_emissivity import compare_discrete_vs_continuum, params_from_dust
from rotor_polarized_emissivity.discrete import polarized_line_list, spectrum_binned_pol


def test_continuum_approximation_for_5A_grain():
    """
    Publication-facing validation: the continuum approximation should track the
    discrete polarized spectrum for a representative 5 Angstrom grain near the
    spectral peak.
    """
    grain = DustGrain(a_eff=5e-8, T_gas=100.0)
    params = params_from_dust(grain, chi=np.pi / 2, sigma=0.1)

    lines = polarized_line_list(params)
    nu_grid = np.linspace(np.min(lines["nu"]), np.max(lines["nu"]), 100)
    discrete_spec = spectrum_binned_pol(lines, nu_grid)

    metrics = compare_discrete_vs_continuum(
        params,
        nu_grid,
        discrete_spec,
        impl="a",
        floor_rel=0.05,
    )

    assert metrics["total"]["max_rel_err"] < 0.2
