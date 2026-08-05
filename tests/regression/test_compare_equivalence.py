
import pytest
import numpy as np
from rotor_emissivity.types import RotorParams
from rotor_emissivity import compare

def test_compare_equivalence_typical():
    """
    Test that A and B agree for typical parameters.
    """
    p = RotorParams(
        n=1e4, A=1e9, B=2e9, 
        mu_par=1e-29, mu_perp=1e-29,
        T_rot=20.0, T_int=300.0
    )
    nu_grid = np.geomspace(1e9, 1e12, 20)
    
    res = compare.compare_grid(p, nu_grid)
    
    # Relax tolerance slightly for integration diffs
    tol = 1e-3
    
    for k, metrics in res['metrics'].items():
        err = metrics['max_rel_err']
        print(f"Branch {k}: err={err}")
        assert err < tol, f"Branch {k} disagreement: {err} > {tol}"

def test_compare_equivalence_high_T():
    """
    Test equality at higher temperatures.
    """
    p = RotorParams(
        n=1e4, A=1e9, B=2e9, 
        mu_par=1e-29, mu_perp=1e-29,
        T_rot=100.0, T_int=1000.0
    )
    nu_grid = np.geomspace(1e9, 1e13, 20)
    res = compare.compare_grid(p, nu_grid)
    tol = 1e-3
    for k, metrics in res['metrics'].items():
        assert metrics['max_rel_err'] < tol

