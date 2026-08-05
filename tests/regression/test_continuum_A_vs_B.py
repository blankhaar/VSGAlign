
import pytest
import numpy as np
from rotor_polarized_emissivity.types import RotorPolParams
from rotor_polarized_emissivity import continuum_impl_a, continuum_impl_b, compare

def test_continuum_equivalence_heavy_rotor():
    # Heavy rotor parameters
    p = RotorPolParams(
        n=1.0,
        A=1e9, B=2e9,
        mu_par=1e-18, mu_perp=1e-18,
        T_rot=100.0, T_int=100.0,
        sigma=0.1, chi=1.5
    )
    nu_grid = np.linspace(10e9, 500e9, 50)
    
    res = compare.compare_continuum_impls(p, nu_grid)
    
    for branch, metrics in res.items():
        assert metrics['max_rel_err'] < 1e-3, f"{branch} mismatch: {metrics['max_rel_err']}"

def test_continuum_equivalence_sigma_func():
    # Variable sigma
    def sig(J, K):
        return 0.1 * (K/J if J>0 else 0)
        
    p = RotorPolParams(
        n=1.0,
        A=1e9, B=2e9,
        mu_par=1e-18, mu_perp=1e-18,
        T_rot=100.0, T_int=100.0,
        sigma=sig, chi=1.5
    )
    nu_grid = np.linspace(10e9, 500e9, 50)
    
    res = compare.compare_continuum_impls(p, nu_grid)
    for branch, metrics in res.items():
        assert metrics['max_rel_err'] < 1e-3, f"{branch} mismatch with function sigma: {metrics['max_rel_err']}"
