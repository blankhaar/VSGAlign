"""Bridge from `DustGrain` objects to polarized rotor parameters.

These helpers connect the grain model used in Sect. 2 and Sect. 4 to the
polarized emissivity routines of Sect. 5.
"""

from typing import Callable, Union
from dust_properties.dust_alignment import DustGrain
from dust_emissivity.adapter import rotor_params_from_dust
from .types import RotorPolParams

def params_from_dust(grain: DustGrain, 
                     chi: float = 0.0,
                     sigma: Union[float, Callable[[float, float], float]] = 0.0,
                     n_limit: float = None) -> RotorPolParams:
    """Build `RotorPolParams` from a `DustGrain` alignment model.

    The returned parameter set is suitable for the polarized emissivity
    formulas of Sect. 5, using the alignment moments supplied or inferred from
    the Sect. 4 grain model.
    """
    base_p = rotor_params_from_dust(grain)
    
    sigma_func = sigma
    
    if sigma == 0.0 or sigma is None:
        from .sigma_utils import SigmaComputer
        from .constants import h_cgs, k_B_cgs
        
        w_val = 1.0
        x_rat = grain.x_ratio()
        y_par = grain.y_param(w=w_val)
        sigma_comp = SigmaComputer(
            x_ratio=x_rat,
            y_param=y_par,
            B=base_p.B,
            A=base_p.A,
            T_int=base_p.T_int
        )
        
        val_under = k_B_cgs * base_p.T_rot / (h_cgs * base_p.B)
        J_th = int(val_under**0.5)
        J_max_precompute = max(2000, 4 * J_th)
        
        sigma_comp.precompute_moments(J_max_precompute)
        sigma_func = sigma_comp

    return RotorPolParams(
        n=n_limit if n_limit is not None else base_p.n,
        A=base_p.A,
        B=base_p.B,
        mu_par=base_p.mu_par,
        mu_perp=base_p.mu_perp,
        T_rot=base_p.T_rot,
        T_int=base_p.T_int,
        sigma=sigma_func,
        chi=chi,
        Z_mode=base_p.Z_mode,
        Jmax_Z_numeric=base_p.Jmax_Z_numeric
    )
