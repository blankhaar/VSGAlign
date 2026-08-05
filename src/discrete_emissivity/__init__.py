"""Reference discrete-line emissivity tools used for validation.

These routines keep the explicit line-by-line version of the Sect. 5.1
rotational emissivity from
`the companion VSG/AME manuscript` and provide the
baseline against which the continuum approximation is checked.
"""

from .compare import compare_spectra, max_rel_err, rel_err
from .rotor_lines import estimate_sufficient_Jmax, line_list
from .spectrum import spectrum_binned

__all__ = [
    "compare_spectra",
    "estimate_sufficient_Jmax",
    "line_list",
    "max_rel_err",
    "rel_err",
    "spectrum_binned",
]
