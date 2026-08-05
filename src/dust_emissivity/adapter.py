"""Adapters from `DustGrain` objects to the public VSGAlign rotor parameters."""

from rotor_emissivity.types import RotorParams
from dust_properties.dust_alignment import DustGrain

def rotor_params_from_dust(grain: DustGrain, *, n_dust: float = 1.0, unit_mode: str = "auto") -> RotorParams:
    """Map a `DustGrain` model onto the Sect. 5 rotor parameter set.

    The mapping pulls the rotational constants and dipole components implied by
    the Sect. 2 grain model, together with the freeze-out temperature of Eq. (3).
    """
    A = grain.Arot
    B = grain.Brot
    T_int = grain.critical_temperature()
    T_rot = grain.T_rot
    mu_par = grain.mu_par
    mu_perp = grain.mu_perp
    
    return RotorParams(
        n=float(n_dust),
        A=float(A),
        B=float(B),
        mu_par=float(mu_par),
        mu_perp=float(mu_perp),
        T_rot=float(T_rot),
        T_int=float(T_int),
        Z_mode="auto",
        Jmax_Z_numeric=max(2000, int(5.0 * (1.38e-16 * T_rot / (6.626e-27 * B))**0.5))
    )

def validate_mapping(p: RotorParams) -> None:
    """Validate that the mapped rotor parameters are physically usable."""
    if p.n < 0:
        raise ValueError("Dust density n must be non-negative.")
    if p.A <= 0 or p.B <= 0:
        raise ValueError("Rotational constants A and B must be positive.")
    if p.T_rot <= 0:
        raise ValueError("Rotational temperature must be positive.")
    if p.T_int <= 0:
        raise ValueError("Internal temperature (T_int) must be positive.")
    if p.mu_par < 0 or p.mu_perp < 0:
        raise ValueError("Dipole moments must be non-negative.")
    
    if p.B <= p.A:
        raise ValueError(f"Implementation requires B > A (oblate). Got A={p.A:.2e}, B={p.B:.2e}.")
