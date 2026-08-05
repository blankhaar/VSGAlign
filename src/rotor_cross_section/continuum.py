"""Continuum absorption cross sections for the aligned symmetric-top model.

These functions are the absorption-side analogues of the rotational-emissivity
calculations. They are most directly related to the absorption discussion in
Sect. 6.1 of `the companion VSG/AME manuscript`,
especially Eqs. (26a)-(27b).
"""

import numpy as np
from scipy.integrate import quad

from rotor_emissivity.constants import c_cgs, ensure_cgs_dipole, h_cgs
from rotor_emissivity.types import RotorParams

from .definitions import P_JK

def sigma_nu_P_parallel_continuum_general(nu, p: RotorParams):
    """Compute the continuum parallel P-branch absorption cross section.

    This helper is related to the Sect. 6 absorption formalism and evaluates
    the continuum analogue of the parallel ``Delta K = 0`` branch.
    """
    nu = np.asarray(nu)
    scalar_input = nu.ndim == 0
    if scalar_input:
        nu = nu[None]
        
    res = np.zeros_like(nu, dtype=float)
    pre_const = (4 * np.pi**3) / (3 * h_cgs * c_cgs)
    mu_sq = ensure_cgs_dipole(p.mu_par, unit="CGS")**2
    inv_2B = 1.0 / (2 * p.B)
    
    for i, freq in enumerate(nu):
        if freq <= 0:
            continue
            
        J_nu = freq * inv_2B
        prefactor = pre_const * mu_sq * J_nu**2
        
        def integrand(x):
            if J_nu < 1.0:
                return 0.0
            
            term1_fac = (2 * J_nu + 1) / (2 * J_nu - 1)
            p_minus = P_JK(J_nu - 1, J_nu * x, p)
            p_curr = P_JK(J_nu, J_nu * x, p)
            
            bracket = term1_fac * p_minus - p_curr
            return (1 - x**2) * bracket

        if J_nu < 1.0:
            res[i] = 0.0
            continue
            
        val, _ = quad(integrand, -1.0, 1.0, epsabs=p.quad_epsabs, epsrel=p.quad_epsrel)
        res[i] = prefactor * val
        
    if scalar_input:
        return res[0]
    return res

def sigma_nu_P_perp_continuum_general(nu, p: RotorParams):
    """Compute the continuum perpendicular P-branch absorption cross section."""
    nu = np.asarray(nu)
    scalar_input = nu.ndim == 0
    if scalar_input:
        nu = nu[None]
        
    res = np.zeros_like(nu, dtype=float)
    pre_const = (np.pi**3) / (6 * p.B * h_cgs * c_cgs)
    mu_sq = ensure_cgs_dipole(p.mu_perp, unit="CGS")**2
    delta_dimless = (p.B - p.A) / p.B
    
    if abs(delta_dimless) < 1e-6:
        return np.zeros_like(nu, dtype=float)

    for i, freq in enumerate(nu):
        if freq <= 0:
            continue
            
        prefactor = pre_const * mu_sq * freq / delta_dimless
        J_lower = freq / (2 * p.B * (1 + delta_dimless))
        J_upper = freq / (2 * p.B * (1 - delta_dimless))
        
        if J_upper < J_lower:
            J_lower, J_upper = J_upper, J_lower
        
        if J_upper <= 0:
            res[i] = 0.0
            continue
        
        actual_lower = max(1.0, J_lower)
        if actual_lower >= J_upper:
            res[i] = 0.0
            continue
            
        def x_nu(J):
            return (freq / (2 * p.B * J) - 1.0) / delta_dimless
            
        def integrand(J):
            x = x_nu(J)
            if abs(x) > 1.0001 or J < 1.0:
                return 0.0
            
            term1_fac = (2 * J + 1) / (2 * J - 1)
            t1 = term1_fac * P_JK(J - 1, J * x + 1, p) - P_JK(J, J * x, p)
            t2 = term1_fac * P_JK(J - 1, -J * x - 1, p) - P_JK(J, -J * x, p)
            
            return (1 - x) ** 2 * (t1 + t2)
        
        val, _ = quad(integrand, actual_lower, J_upper, epsabs=p.quad_epsabs, epsrel=p.quad_epsrel)
        res[i] = prefactor * val
        
    if scalar_input:
        return res[0]
    return res

def sigma_nu_Q_perp_continuum_general(nu, p: RotorParams):
    """Compute the continuum perpendicular Q-branch absorption cross section."""
    nu = np.asarray(nu)
    scalar_input = nu.ndim == 0
    if scalar_input:
        nu = nu[None]
        
    res = np.zeros_like(nu, dtype=float)
    pre_const = (np.pi**3) / (3 * p.B * h_cgs * c_cgs)
    mu_sq = ensure_cgs_dipole(p.mu_perp, unit="CGS")**2
    delta_dimless = (p.B - p.A) / p.B
    if abs(delta_dimless) < 1e-6:
        return np.zeros_like(nu, dtype=float)
    
    for i, freq in enumerate(nu):
        if freq <= 0:
            continue
        
        K_nu = freq / (2 * (p.B - p.A))
        if K_nu < 0:
            K_nu = abs(K_nu)
        
        prefactor = pre_const * mu_sq * freq / delta_dimless
        
        def integrand(J):
            if J < K_nu or J == 0:
                return 0.0
            
            fac = 1.0 - (K_nu / J)**2
            if fac < 0:
                return 0.0
            
            t1 = P_JK(J, K_nu + 1, p) - P_JK(J, K_nu, p)
            t2 = P_JK(J, -K_nu - 1, p) - P_JK(J, -K_nu, p)
            
            return fac * (t1 + t2)
            
        limit_lower = max(1.0, K_nu)
        
        val, _ = quad(integrand, limit_lower, np.inf, epsabs=p.quad_epsabs, epsrel=p.quad_epsrel)
        res[i] = prefactor * val
        
    if scalar_input:
        return res[0]
    return res

def sigma_nu_total(nu, p: RotorParams):
    """Return the total continuum absorption cross section."""
    return (sigma_nu_P_parallel_continuum_general(nu, p) +
            sigma_nu_P_perp_continuum_general(nu, p) +
            sigma_nu_Q_perp_continuum_general(nu, p))
            
