"""Comparison helpers for the two continuum emissivity implementations.

The routines in this module validate that the alternative numerical
evaluations of the Appendix E emissivity formulas agree on a shared frequency
grid.
"""

import numpy as np
from typing import Dict, Any
from .types import RotorParams
from . import impl_a, impl_b

def max_rel_err(a: np.ndarray, b: np.ndarray, floor: float = 1e-20) -> float:
    """Return the maximum elementwise relative error."""
    denom = np.maximum(np.abs(a), np.abs(b)) + floor
    diff = np.abs(a - b)
    return np.max(diff / denom)

def mean_rel_err(a: np.ndarray, b: np.ndarray, floor: float = 1e-20) -> float:
    """Return the mean elementwise relative error."""
    denom = np.maximum(np.abs(a), np.abs(b)) + floor
    diff = np.abs(a - b)
    return np.mean(diff / denom)

def compare_grid(p: RotorParams, nu_grid: np.ndarray) -> Dict[str, Any]:
    """Evaluate implementations A and B on a shared frequency grid."""
    res_a = {
        'P_parallel': impl_a.jnu_P_parallel(nu_grid, p),
        'P_perp': impl_a.jnu_P_perp(nu_grid, p),
        'Q_perp': impl_a.jnu_Q_perp(nu_grid, p),
        'total': impl_a.jnu_total(nu_grid, p)
    }
    
    res_b = {
        'P_parallel': impl_b.jnu_P_parallel(nu_grid, p),
        'P_perp': impl_b.jnu_P_perp(nu_grid, p),
        'Q_perp': impl_b.jnu_Q_perp(nu_grid, p),
        'total': impl_b.jnu_total(nu_grid, p)
    }
    
    metrics = {}
    floor = 1e-30
    
    for key in res_a:
        va = res_a[key]
        vb = res_b[key]
        
        metrics[key] = {
            'max_rel_err': max_rel_err(va, vb, floor=floor),
            'mean_rel_err': mean_rel_err(va, vb, floor=floor),
            'max_abs_err': np.max(np.abs(va - vb))
        }
        
    return {
        'a': res_a,
        'b': res_b,
        'metrics': metrics
    }
