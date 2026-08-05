
import pytest
import numpy as np
from dust_properties.dust_alignment import DustGrain
from dust_emissivity.simulate import jnu_dust
from discrete_emissivity import rotor_lines, spectrum, compare
from dust_emissivity.adapter import rotor_params_from_dust

def test_dust_smoke():
    """
    Smoke test comparing discrete vs continuum for a physical dust grain.
    Note: Real dust grains have tiny spacing (B small).
    High J values -> Continuum should be very good.
    However, discrete calc needs huge Jmax.
    We test on a VERY small grain (high B) to keep Jmax manageable for test.
    """
    # 5 Angstrom grain.
    # a = 5e-8 cm.
    # I ~ a^5. B ~ 1/I ~ a^-5.
    # If a=1e-5 (0.1um), B ~ 1e5 Hz. Jpeak ~ 1e4. Too slow.
    # If a=5e-8 (0.5nm), B ~ 1e9 Hz. Jpeak ~ 100. Fast.
    dg = DustGrain(a_eff=5e-8, T_gas=100.0)
    p = rotor_params_from_dust(dg)
    
    # nu_grid around peak.
    # Peak J ~ 100. nu ~ 2 B J ~ 2e11 Hz.
    nu_grid = np.linspace(1e11, 5e11, 20)
    
    # Discrete
    lines = rotor_lines.line_list(p, Jmax=300)
    d_spec = spectrum.spectrum_binned(lines, nu_grid)
    
    # Continuum
    c_val = jnu_dust(dg, nu_grid, impl="a", branch="total")
    
    # Check
    err = compare.rel_err(d_spec['total'], c_val)
    mean = np.mean(err)
    print(f"Mean error (dust): {mean}")
    
    assert mean < 0.5 # Generous tolerance for binned vs continuum on coarse grid
