
import pytest
import numpy as np
from dust_properties.dust_alignment import DustGrain
from dust_emissivity.adapter import rotor_params_from_dust, validate_mapping
from rotor_emissivity.types import RotorParams

def test_adapter_basic():
    dg = DustGrain(a_eff=1e-5, T_gas=100.0)
    p = rotor_params_from_dust(dg)
    
    validate_mapping(p)
    
    # Check values
    assert np.isfinite(p.T_rot)
    assert p.T_rot > 0.0
    assert p.T_rot == pytest.approx(dg.T_rot)
    assert p.n == 1.0 # Default
    assert p.A > 0
    assert p.B > 0
    assert p.mu_par > 0

def test_adapter_n_dust():
    dg = DustGrain()
    p = rotor_params_from_dust(dg, n_dust=123.0)
    assert p.n == 123.0

def test_adapter_oblate_check():
    # DustGrain defaults to s=0.5 (oblate), so B > A.
    dg = DustGrain(s=0.5)
    p = rotor_params_from_dust(dg)
    assert p.B > p.A
    
    # What if we force prolate? s > 1.
    # DustGrain logic: if s > 1 (prolate), a < c.
    # I_par (along c) = 2/5 M a^2
    # I_perp (along a) = 1/5 M (a^2 + c^2)
    # If prolate, c > a.
    # I_perp = 1/5 M (a^2 + c^2) > 1/5 M (a^2 + a^2) = 2/5 M a^2 = I_par ?
    # Let's check. c = s*a. I_perp = 0.2 M a^2 (1 + s^2).
    # I_par = 0.4 M a^2.
    # I_perp > I_par if 0.2(1+s^2) > 0.4 => 1+s^2 > 2 => s^2 > 1 => s > 1.
    # So for prolate grain, I_perp > I_par.
    # B = h / I_perp. A = h / I_par.
    # If I_perp > I_par, then B < A.
    # Our Rotor implementation assumes B > A.
    # So prolate grains should fail validation.
    
    with pytest.raises(ValueError, match="Expected oblate geometry"):
        _ = DustGrain(s=2.0)
