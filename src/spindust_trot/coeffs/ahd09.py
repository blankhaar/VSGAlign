"""AHD09 coefficient backend (minimal single-T adaptation).

Ali-Haïmoud, Hirata & Dickinson (2009; arXiv:0812.2904) show that the
infrared-emission angular-momentum loss in DL98b is underestimated by a
factor of 2 (see AHD09 eq. (151)), leading to a factor-of-2 change in the
normalized IR damping coefficient definition (AHD09 eq. (153)).

In this backend we implement a *minimal* improved-coefficient option:

- keep DL98b analytic IR scaling forms (DL98b eqs. (30)–(33)) for the IR spectrum,
- but multiply DL98b's dimensionless IR damping factor F_IR by 2,
  consistent with the corrected angular-momentum loss rate.

All other DL98b coefficients are retained.

Repository role
---------------
This backend is the practical default for `DustGrain.T_rot` in the paper
workflow. It supplies the rotational temperatures that later enter the
Sect. 2.2 population model and, through that, the Sect. 5 emissivity and
Sect. 6 cross-section figures.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..env import Environment
from ..grain import GrainModel
from . import dl98

__all__ = [
    "compute_FG",
]


def compute_FG(a_cm: ArrayLike, env: Environment, grain: GrainModel) -> dl98.DL98FG:
    """Compute F/G terms with the AHD09 IR damping correction applied.

    References
    ----------
    - AHD09 eq. (151): corrected IR angular-momentum loss is 2× DL98b.
    - AHD09 eq. (153): normalized IR damping coefficient FIR carries this factor of 2.

    The output is consumed by `DustGrain.compute_rotational_temperature`, not
    by the paper scripts directly.
    """

    fg = dl98.compute_FG(a_cm=a_cm, env=env, grain=grain)

    # Apply the factor-of-2 correction to IR damping (AHD09 eq. (153)).
    F_IR = 2.0 * fg.F_IR
    F_total = fg.F_n + fg.F_i + fg.F_p + F_IR  # DL98b eq. (18) structure preserved.

    # Systematic-torque term depends on the total damping F through ω_s (DL98b eq. (50)).
    geom = grain.geometry(fg.a_cm)
    tauH = dl98.tau_H(geom, env, grain)
    G_s = dl98._G_s_H2_systematic(  # type: ignore[attr-defined]
        fg.a_cm, geom, env, F_total, tauH, fg.mu2_cgs
    )
    G_total = fg.G_n + fg.G_i + fg.G_p + fg.G_IR + G_s  # DL98b eq. (56)

    return dl98.DL98FG(
        a_cm=fg.a_cm,
        mean_Z2=fg.mean_Z2,
        mu2_cgs=fg.mu2_cgs,
        F_n=fg.F_n,
        F_i=fg.F_i,
        F_p=fg.F_p,
        F_IR=F_IR,
        F_total=F_total,
        G_n=fg.G_n,
        G_i_in=fg.G_i_in,
        G_i_ev=fg.G_i_ev,
        G_i=fg.G_i,
        G_p=fg.G_p,
        G_IR=fg.G_IR,
        G_pe=fg.G_pe,
        G_H2=fg.G_H2,
        G_s=G_s,
        G_total=G_total,
    )
