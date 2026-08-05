"""Partition functions for the two-temperature rotor population model.

These helpers normalize the level populations from Sect. 2.2, especially
Eq. (4), for the continuum and discrete emissivity calculations that appear in
Sect. 5.1 of `the companion VSG/AME manuscript`.
"""

import numpy as np
from .types import RotorParams
from .constants import h_cgs, k_B_cgs

def _get_beta_gamma(p: RotorParams):
    """Return the dimensionless population parameters of Eq. (4)."""
    beta = h_cgs * p.B / (k_B_cgs * p.T_rot)
    delta = p.B - p.A
    gamma = h_cgs * delta / (k_B_cgs * p.T_int)
    return beta, gamma

def Z_analytic(p: RotorParams) -> float:
    """Evaluate the high-``J`` continuum normalization for Eq. (4).

    This is the analytic partition function used when the Appendix E continuum
    approximation is valid and ``beta > gamma``.
    """
    beta, gamma = _get_beta_gamma(p)
    
    if beta <= gamma:
        return float('inf')
        
    term = beta - gamma
    if term <= 0:
        return float('inf')

    return np.sqrt(np.pi) / (beta * np.sqrt(term))

def Z_numeric(p: RotorParams, Jmax: int = None) -> float:
    """Compute the discrete partition sum corresponding to Eq. (4)."""
    if Jmax is None:
        Jmax = p.Jmax_Z_numeric
        
    beta, gamma = _get_beta_gamma(p)
    J = np.arange(0, Jmax + 1)
    term_J = (2 * J + 1) * np.exp(-beta * J * (J + 1))

    K_all = np.arange(0, Jmax + 1)
    exp_gamma_K2 = np.exp(gamma * K_all**2)
    S = np.cumsum(exp_gamma_K2)
    inner_sums = 2 * S - 1

    return float(np.sum(term_J * inner_sums))

def Z(p: RotorParams) -> float:
    """Dispatch to the requested partition-function approximation."""
    mode = p.Z_mode
    if mode == "auto":
        beta, gamma = _get_beta_gamma(p)
        if beta > gamma:
            return Z_analytic(p)
        else:
            return Z_numeric(p)
    elif mode == "analytic":
        return Z_analytic(p)
    elif mode == "numeric":
        return Z_numeric(p)
    else:
        raise ValueError(f"Unknown Z_mode: {mode}")
