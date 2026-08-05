"""Comparison helpers for discrete and continuum emissivity spectra."""

import numpy as np
from typing import Dict

def rel_err(a: np.ndarray, b: np.ndarray, floor: float = 1e-40) -> np.ndarray:
    """Return the elementwise relative error against a reference array."""
    denom = np.maximum(np.abs(b), floor)
    return np.abs(a - b) / denom

def max_rel_err(a: np.ndarray, b: np.ndarray, floor: float = 1e-40) -> float:
    """Return the maximum relative error against a reference array."""
    return np.max(rel_err(a, b, floor))

def compare_spectra(d_spec: Dict[str, np.ndarray], 
                    c_spec: Dict[str, np.ndarray], 
                    param_desc: str = "") -> Dict[str, float]:
    """Compare a discrete spectrum against a continuum spectrum branch by branch."""
    metrics = {}
    for branch in ['total', 'P_parallel', 'P_perp', 'Q_perp']:
        if branch in d_spec and branch in c_spec:
            v_d = d_spec[branch]
            v_c = c_spec[branch]
            
            # Floor relevant to peak
            peak = np.max(v_d)
            if peak == 0:
                err = 0.0
            else:
                floor = peak * 1e-6 # Ignore deep tails
                err = max_rel_err(v_d, v_c, floor=floor)
            metrics[branch] = err
            
    return metrics
