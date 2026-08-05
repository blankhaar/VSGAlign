"""Constants for polarized spinning-dust emissivity calculations.

The geometry factors in this module are the large-``J`` polarization weights
used in Sect. 5.1 and Appendix E of
`the companion VSG/AME manuscript`.
"""

import numpy as np
from rotor_emissivity.constants import h_cgs, k_B_cgs, c_cgs

__all__ = ['h_cgs', 'k_B_cgs', 'c_cgs', 'hbar_cgs', 
           'w_P', 'w_Q', 'calc_P_chi']

hbar_cgs = h_cgs / (2 * np.pi)

w_P = np.sqrt(1.0 / 10.0)
w_Q = -np.sqrt(2.0 / 5.0)

def calc_P_chi(chi: float) -> float:
    """Return the ``sin^2 chi`` viewing factor from Eq. (E.3)."""
    return 3.0 * np.sin(chi)**2 / (2.0 * np.sqrt(2.0))
