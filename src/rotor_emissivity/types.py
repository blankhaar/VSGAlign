"""Parameter containers for continuum rotor emissivity calculations.

`RotorParams` stores the oblate symmetric-top inputs used throughout the
continuum implementation of Sect. 5.1 and Appendix E of
`the companion VSG/AME manuscript`.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class RotorParams:
    """Inputs for the symmetric-top emissivity model.

    The parameters encode the rotational constants from Sect. 2.1, the
    two-temperature population model of Eq. (4), and the dipole geometry used
    in the rotational-emissivity formulas of Sect. 5.1.
    """

    n: float            # grain number density [cm^-3]
    A: float            # rotational constant parallel to the symmetry axis [Hz]
    B: float            # rotational constant perpendicular to the symmetry axis [Hz]
    mu_par: float       # dipole component parallel to the symmetry axis [esu cm]
    mu_perp: float      # dipole component perpendicular to the symmetry axis [esu cm]
    T_rot: float        # rotational temperature [K]
    T_int: float        # internal freeze-out temperature [K]

    use_highJ_J2: bool = True
    Z_mode: str = "analytic"      # "analytic"|"numeric"|"auto"
    Jmax_Z_numeric: int = 2000
    quad_epsabs: float = 1e-10
    quad_epsrel: float = 1e-8
