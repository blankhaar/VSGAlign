"""Frequency-grid binning for the discrete rotational line lists."""

import numpy as np
from typing import Dict, Any

def spectrum_binned(line_data: np.ndarray, nu_grid: np.ndarray, method: str = "hist", sigma_nu: float = None) -> Dict[str, np.ndarray]:
    """Bin the discrete line list onto a frequency grid."""
    if len(nu_grid) < 2:
        raise ValueError("nu_grid must have at least 2 points.")
        
    edges = np.zeros(len(nu_grid) + 1)
    edges[0] = nu_grid[0] - (nu_grid[1] - nu_grid[0]) / 2
    edges[-1] = nu_grid[-1] + (nu_grid[-1] - nu_grid[-2]) / 2
    edges[1:-1] = (nu_grid[1:] + nu_grid[:-1]) / 2
    dnu = np.diff(edges)

    results = {
        'total': np.zeros_like(nu_grid),
        'P_parallel': np.zeros_like(nu_grid),
        'P_perp': np.zeros_like(nu_grid),
        'Q_perp': np.zeros_like(nu_grid)
    }

    unique_branches = np.unique(line_data['branch'])
    
    for branch in unique_branches:
        mask = line_data['branch'] == branch
        subset = line_data[mask]
        
        nu_vals = subset['nu']
        weights = subset['weight']
        
        hist, _ = np.histogram(nu_vals, bins=edges, weights=weights)
        j_nu_branch = hist / dnu
        
        if branch in results:
            results[branch] += j_nu_branch
        
        results['total'] += j_nu_branch
        
    return results
