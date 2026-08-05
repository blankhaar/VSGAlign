"""Regression checks for the documented public API example."""

import warnings

import numpy as np
from scipy.integrate import IntegrationWarning

from vsgalign import DustGrain, jnuq_total, polarized_params_from_dust


def test_polarized_quickstart_has_no_integration_warnings():
    """The README-sized polarized calculation is finite and warning-free."""
    grain = DustGrain(a_eff=5e-8, T_gas=100.0)
    params = polarized_params_from_dust(grain, chi=np.pi / 2)
    frequencies = np.linspace(1e9, 80e9, 200)

    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        polarized = jnuq_total(frequencies, params)

    assert polarized.shape == frequencies.shape
    assert np.isfinite(polarized).all()
