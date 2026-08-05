"""Single-temperature spinning-dust rotational-state utilities.

This package implements the DL98/AHD09 rotational-temperature machinery used
to supply `T_rot` for the grain model in
`dust_properties.dust_alignment`.
That grain model is then bridged into the emissivity and absorption solvers
used by the paper workflow.

In terms of `the companion VSG/AME manuscript`, this
package provides the external rotational-temperature input referenced in
Sect. 2.2 below Eq. (4), where the manuscript states that `T_rot` is computed
from the DL98 formalism with AHD09 updates.
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
