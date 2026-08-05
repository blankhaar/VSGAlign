"""Parameter containers for polarized emissivity calculations.

These types extend the unpolarized rotor parameters with the alignment moment
``sigma(J, K)`` and the viewing geometry needed for Sect. 4 and Sect. 5 of
`the companion VSG/AME manuscript`.
"""

from dataclasses import dataclass, field
from typing import Callable, Union, Optional
import numpy as np
from rotor_emissivity.types import RotorParams

SigmaFunc = Callable[[float, float], float]

@dataclass(frozen=True)
class RotorPolParams(RotorParams):
    """Inputs for the polarized emissivity formulas of Sect. 5.

    The additional fields represent the rank-2 alignment information from
    Sect. 4 and the observer geometry used in Eq. (17b) and Eqs. (22a)-(22c).
    """

    sigma: Union[float, SigmaFunc] = 0.0
    chi: float = 0.0
    _sigma_const: Optional[float] = field(init=False, default=None)
    
    def __post_init__(self):
        if isinstance(self.sigma, (int, float)):
            val = float(self.sigma)
            object.__setattr__(self, "_sigma_const", val)
            object.__setattr__(self, "sigma", lambda J, K: val)

            
    def get_sigma(self, J, K):
        """Evaluate the alignment moment ``sigma(J, K)``."""
        return self.sigma(J, K)
