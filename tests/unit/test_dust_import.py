
import pytest
from dust_properties.dust_alignment import DustGrain

def test_dust_grain_instantiation():
    """
    Verify we can import and instantiate DustGrain.
    """
    dg = DustGrain()
    assert dg.a_eff > 0
    assert dg.T_gas > 0
    
    # Check dipole moments are calculated (in CGS/esu cm)
    # Expect ~ Debye range (1e-18)
    assert dg.mu_par >= 0
    assert dg.mu_perp >= 0
    
    # Just check not None
    assert isinstance(dg.mu_par, float)
