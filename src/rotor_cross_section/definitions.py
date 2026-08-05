"""Population definitions used by the continuum cross-section integrals.

The key quantity is the two-temperature level population from Eq. (4) of
`the companion VSG/AME manuscript`, reused here for the
absorption-side calculations connected to Sect. 6.
"""

import numpy as np
from rotor_emissivity.types import RotorParams
from rotor_emissivity.constants import h_cgs, k_B_cgs
from rotor_emissivity.partition import Z

def _get_beta_gamma(p: RotorParams):
    """Return the dimensionless population parameters of Eq. (4)."""
    beta = h_cgs * p.B / (k_B_cgs * p.T_rot)
    delta = p.B - p.A
    gamma = h_cgs * delta / (k_B_cgs * p.T_int)
    return beta, gamma

def P_JK(J, K, p: RotorParams) -> float:
    """Evaluate the Eq. (4) rotor population at possibly continuous ``J`` and ``K``.

    This helper lets the absorption-side continuum integrals reuse the same
    level population model as the emissivity calculations.
    """
    z_val = Z(p)
    if np.isinf(z_val) or z_val == 0:
        return 0.0
        
    beta, gamma = _get_beta_gamma(p)
    
    if beta <= gamma and p.Z_mode != "analytic":
        if np.isscalar(J):
            if J > p.Jmax_Z_numeric:
                return 0.0

    term1 = 2 * J + 1
    exponent = -beta * J * (J + 1) + gamma * K**2
    log_z = np.log(z_val)
    final_exponent = exponent - log_z
    exp_combined = np.exp(final_exponent)
    result = term1 * exp_combined

    if beta <= gamma and p.Z_mode != "analytic" and not np.isscalar(J):
        result = np.where(J > p.Jmax_Z_numeric, 0.0, result)
        
    return result
