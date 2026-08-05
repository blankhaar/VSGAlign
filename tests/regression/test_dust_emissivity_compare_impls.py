
import pytest
import numpy as np
from dust_properties.dust_alignment import DustGrain
from dust_emissivity.simulate import jnu_dust
from rotor_emissivity.compare import max_rel_err


def test_dust_a_vs_b():
    """
    Compare implementations A and B for a typical dust grain.
    Use a subclass to enforce T_int = T_rot for stability (beta > gamma).
    """
    class StableDustGrain(DustGrain):
        def critical_temperature(self):
            return self.T_rot

    dg = StableDustGrain(a_eff=1e-5, s=0.5, T_gas=100.0)
    # Fundamental freq ~ 0.01 Hz. Peak ~ 2e5 Hz.
    # Test around peak-ish.
    nu_grid = np.geomspace(1e4, 1e7, 10)
    
    # Calculate branchwise
    ja_tot = jnu_dust(dg, nu_grid, impl="a", branch="total")
    jb_tot = jnu_dust(dg, nu_grid, impl="b", branch="total")
    
    # Check errors
    err = max_rel_err(ja_tot, jb_tot)
    print(f"Max rel err (total): {err}")
    assert err < 1e-3

def test_dust_small_grain():
    """
    Smaller grain. Stable.
    """
    class StableDustGrain(DustGrain):
        def critical_temperature(self):
            return self.T_rot

    dg = StableDustGrain(a_eff=1e-6, T_gas=50.0)
    # Peak higher ~ 60 GHz for a=1e-7. For a=1e-6:
    # I ~ a^5 (factor 1e-10). B ~ 1e10 higher?
    # No, I ~ 1e-5^5 = 1e-25. I ~ 1e-6^5 = 1e-30.
    # B ~ 1e-2 vs 1e3.
    # Peak ~ 1e3 * 1e4 ~ 1e7 Hz (10 MHz).
    nu_grid = np.geomspace(1e5, 1e8, 20)
    
    ja = jnu_dust(dg, nu_grid, impl="a")
    jb = jnu_dust(dg, nu_grid, impl="b")
    
    err = max_rel_err(ja, jb)
    assert err < 1e-3

    
def test_dust_sanity():
    """
    Test positivity and finite values.
    """
    dg = DustGrain()
    nu = 100e9 # 100 GHz
    
    val = jnu_dust(dg, np.array([nu]))
    assert val[0] > 0
    assert np.isfinite(val[0])
