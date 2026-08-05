
import numpy as np

from rotor_polarized_emissivity.types import RotorPolParams
from rotor_polarized_emissivity import discrete, compare
from dust_properties.dust_alignment import DustGrain
from rotor_polarized_emissivity.dust_bridge import params_from_dust

def test_discrete_vs_continuum_agreement_large_grain():
    """The 5 Angstrom discrete spectrum tracks its continuum limit."""
    dg = DustGrain(a_eff=5e-8, T_gas=100)
    p = params_from_dust(dg, chi=np.pi / 2, sigma=0.1)

    lines = discrete.polarized_line_list(p)
    assert len(lines) > 0

    nu_min = np.min(lines["nu"])
    nu_max = np.max(lines["nu"])
    nu_grid = np.linspace(nu_min, nu_max, 100)
    d_spec = discrete.spectrum_binned_pol(lines, nu_grid)

    res = compare.compare_discrete_vs_continuum(
        p,
        nu_grid,
        d_spec,
        impl="a",
        floor_rel=0.05,
    )
    mre = res["total"]["max_rel_err"]
    assert mre < 0.2, f"Discrepancy too high: {mre}"


def test_discrete_vs_continuum_sign_flip():
    """Changing the alignment sign flips every polarized line weight."""
    dg = DustGrain(a_eff=10e-8, T_gas=100)

    p_pos = params_from_dust(dg, chi=np.pi / 2, sigma=0.1)
    p_neg = params_from_dust(dg, chi=np.pi / 2, sigma=-0.1)

    # This invariant does not require the roughly nine-million-state adaptive
    # line list of a 10 Angstrom grain. A fixed cutoff exercises every branch
    # while keeping the regression bounded for laptops and CI runners.
    d_lines_pos = discrete.polarized_line_list(p_pos, Jmax=120)
    d_lines_neg = discrete.polarized_line_list(p_neg, Jmax=120)

    assert len(d_lines_pos) == len(d_lines_neg)
    assert np.array_equal(d_lines_pos["nu"], d_lines_neg["nu"])
    assert np.array_equal(d_lines_pos["branch"], d_lines_neg["branch"])
    assert np.allclose(d_lines_pos["weight_q"], -d_lines_neg["weight_q"])
