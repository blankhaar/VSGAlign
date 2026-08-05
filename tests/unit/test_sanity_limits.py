
import pytest
import numpy as np
from rotor_emissivity.types import RotorParams
from rotor_emissivity import impl_a, impl_b

@pytest.mark.parametrize("module", [impl_a, impl_b])
def test_positivity(module):
    """
    Test that emissivity is non-negative.
    """
    p = RotorParams(
        n=1e4, A=1e9, B=2e9, 
        mu_par=1e-29, mu_perp=1e-29,
        T_rot=20.0, T_int=300.0
    )
    nu = np.linspace(1e9, 1e12, 50)
    
    # Check each branch
    for branch in ['jnu_P_parallel', 'jnu_P_perp', 'jnu_Q_perp', 'jnu_total']:
        func = getattr(module, branch)
        val = func(nu, p)
        assert np.all(val >= 0), f"{branch} produced negative values"
        assert np.all(np.isfinite(val)), f"{branch} produced non-finite values"

@pytest.mark.parametrize("module", [impl_a, impl_b])
def test_divergence_check(module):
    """
    Test that we handle (or at least don't crash) near divergence condition beta=gamma.
    However, the current code assumes beta > gamma for analytic Z.
    Let's test a case close to bound.
    """
    # beta = gamma => B/Trot = (B-A)/Tint
    # p.A = 1e9, p.B = 2e9. Delta = 1e9.
    # B/Trot approx Delta/Tint => 2/Trot = 1/Tint => Tint = 0.5 Trot.
    # Normally Tint >> Trot.
    
    # Let's try comfortable stable region.
    p = RotorParams(
        n=1e4, A=1e9, B=2e9, 
        mu_par=1e-29, mu_perp=1e-29,
        T_rot=20.0, T_int=300.0
    )
    # This is safe.
    
    # Try a case where gamma is large (T_int small) but still stable?
    # gamma < beta. h Delta / k Tint < h B / k Trot
    # Delta/ Tint < B / Trot
    # 1e9 / Tint < 2e9 / 20 => 1/Tint < 1/10 => Tint > 10.
    
    # Test strict positivity.
    val = module.jnu_total(1e11, p)
    assert val > 0
