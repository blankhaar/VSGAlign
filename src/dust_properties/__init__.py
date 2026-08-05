"""Dust-grain model interfaces used by the emissivity packages.

The main `DustGrain` class encapsulates the oblate symmetric-top grain model
and temperature scaling introduced in Sect. 2 of
`the companion VSG/AME manuscript`.
"""

from .dust_alignment import DustGrain

__all__ = ["DustGrain"]
