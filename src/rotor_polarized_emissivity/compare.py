"""Validation helpers for polarized emissivity implementations.

These routines compare alternative numerical evaluations of the polarized
continuum formulas in Appendix E and their agreement with the discrete
line-by-line reference model.
"""

import numpy as np
from typing import Dict
from .types import RotorPolParams
from . import continuum_impl_a, continuum_impl_b

def max_rel_err(a, b, floor=1e-40):
    """Return the maximum relative error against a reference array."""
    denom = np.maximum(np.abs(b), floor)
    return np.max(np.abs(a - b) / denom)

def l2_rel_err(a, b, floor=1e-40):
    """Return the RMS relative error against a reference array."""
    denom = np.maximum(np.abs(b), floor)
    return np.sqrt(np.mean(((a - b) / denom)**2))

def compare_continuum_impls(p: RotorPolParams, nu_grid: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compare implementations A and B on a shared frequency grid."""
    results = {}
    branches = ['total', 'P_parallel', 'P_perp', 'Q_perp']
    
    for branch in branches:
        func_name = f"jnuq_{branch}"
        func_a = getattr(continuum_impl_a, func_name)
        func_b = getattr(continuum_impl_b, func_name)
        
        val_a = func_a(nu_grid, p)
        val_b = func_b(nu_grid, p)
        
        peak = np.max(np.abs(val_a))
        floor = peak * 1e-8 if peak > 0 else 1e-40
        
        mre = max_rel_err(val_a, val_b, floor)
        l2 = l2_rel_err(val_a, val_b, floor)
        
        results[branch] = {"max_rel_err": mre, "l2_rel_err": l2}
        
    return results

def compare_discrete_vs_continuum(p: RotorPolParams, nu_grid: np.ndarray, 
                                  d_spec: Dict[str, np.ndarray],
                                  impl="a", floor_rel=1e-6) -> Dict[str, Dict[str, float]]:
    """Compare a discrete polarized spectrum against a continuum approximation."""
    mod = continuum_impl_a if impl == "a" else continuum_impl_b
    
    results = {}
    branches = ['total', 'P_parallel', 'P_perp', 'Q_perp']
    
    for branch in branches:
        func_name = f"jnuq_{branch}"
        func_c = getattr(mod, func_name)
        val_c = func_c(nu_grid, p)
        
        if branch not in d_spec:
            continue
            
        val_d = d_spec[branch]
        
        peak = np.max(np.abs(val_d))
        floor = peak * floor_rel if peak > 0 else 1e-40
        
        mre = max_rel_err(val_d, val_c, floor)
        results[branch] = {"max_rel_err": mre}
        
    return results
