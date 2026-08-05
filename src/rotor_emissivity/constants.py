"""
Physical constants and unit conversion helpers.
All internal calculations use CGS units.
"""

# Physical constants in CGS
h_cgs = 6.62607015e-27  # erg s
k_B_cgs = 1.380649e-16  # erg K^-1
c_cgs = 2.99792458e10   # cm s^-1

# Conversion factors
DEBYE_TO_ESU_CM = 1e-18
CM_TO_ESU_CM = 2.99792458e9 * 10  # 1 C = 3e9 esu, 1 m = 100 cm. So 1 C m = 3e11 esu cm. Wait.
# 1 C = 2.99792458e9 esu.
# 1 m = 100 cm.
# 1 C m = 2.99792458e9 * 100 esu cm = 2.99792458e11 esu cm.
SI_DIPOLE_TO_CGS = 2.99792458e11 

# Helper to ensure we have CGS dipole moment
def ensure_cgs_dipole(val: float, unit: str = "SI") -> float:
    """
    Convert dipole moment to esu cm.
    
    Args:
        val: Value of dipole moment
        unit: 'SI' (C m) or 'Debye' or 'CGS' (esu cm)
    
    Returns:
        Value in esu cm
    """
    if unit == "SI":
        return val * SI_DIPOLE_TO_CGS
    elif unit == "Debye":
        return val * DEBYE_TO_ESU_CM
    elif unit == "CGS":
        return val
    else:
        raise ValueError(f"Unknown unit: {unit}")
