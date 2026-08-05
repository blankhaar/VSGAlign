"""Convenience wrappers for running emissivity calculations from `DustGrain` objects."""

import numpy as np
import copy
from typing import Dict, Any, List
from dust_properties.dust_alignment import DustGrain
from rotor_emissivity import impl_a, impl_b
from .adapter import rotor_params_from_dust, validate_mapping

def jnu_dust(grain: DustGrain, nu: np.ndarray, *, 
             impl: str = "a", 
             branch: str = "total", 
             n_dust: float = 1.0) -> np.ndarray:
    """Compute Sect. 5 emissivity for a single `DustGrain` model."""
    p = rotor_params_from_dust(grain, n_dust=n_dust)
    validate_mapping(p)
    
    if impl == "a":
        module = impl_a
    elif impl == "b":
        module = impl_b
    else:
        raise ValueError(f"Unknown implementation: {impl}")
        
    func_name = f"jnu_{branch}"
    if not hasattr(module, func_name):
        raise ValueError(f"Unknown branch: {branch}")
        
    func = getattr(module, func_name)
    return func(nu, p)

def simulate_over_sizes(a_list_cm: List[float], 
                        base_kwargs: Dict[str, Any], 
                        nu_grid: np.ndarray, 
                        impl: str = "a") -> Dict[str, np.ndarray]:
    """Evaluate the total emissivity over a list of effective grain sizes."""
    results = {}
    
    for a in a_list_cm:
        kwargs = copy.deepcopy(base_kwargs)
        kwargs['a_eff'] = a
        
        g = DustGrain(**kwargs)
        jnu = jnu_dust(g, nu_grid, impl=impl, branch="total", n_dust=1.0)
        results[a] = jnu
        
    return results
