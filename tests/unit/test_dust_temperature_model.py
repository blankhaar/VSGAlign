from __future__ import annotations

import numpy as np

from dust_properties.dust_alignment import DustGrain, _excitation_equivalent_radius_oblate
from spindust_trot.charge import ChargeDistribution


def _numeric_ax_oblate(a: float, c: float, n_theta: int = 50001) -> float:
    theta = np.linspace(0.0, np.pi, n_theta, dtype=np.float64)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    r2 = a * a * sin_t * sin_t + c * c * cos_t * cos_t
    ds = a * sin_t * np.sqrt(c * c * sin_t * sin_t + a * a * cos_t * cos_t)
    integral = 2.0 * np.pi * np.trapezoid(r2 * ds, theta)
    return float((integral / (4.0 * np.pi)) ** 0.25)


def test_default_backend_is_ahd09_and_used_on_init() -> None:
    g = DustGrain(a_eff=9.0e-8, T_gas=100.0, n_H=30.0, U=1.0)
    assert g.coeff_version == "ahd09"
    t_from_default = g.compute_rotational_temperature(coeff_version="ahd09", update=False)
    np.testing.assert_allclose(g.T_rot, t_from_default, rtol=0, atol=0)


def test_ahd09_temperature_not_above_dl98_for_default_case() -> None:
    g = DustGrain(recompute_t_rot=False)
    t_dl98 = g.compute_rotational_temperature(coeff_version="dl98", update=False)
    t_ahd09 = g.compute_rotational_temperature(coeff_version="ahd09", update=False)
    assert t_ahd09 <= t_dl98


def test_ax_closed_form_matches_numeric_reference() -> None:
    a = 1.7e-7
    for q in [1.0, 0.85, 0.6, 0.4, 0.2]:
        c = q * a
        ax_closed = _excitation_equivalent_radius_oblate(a, c)
        ax_numeric = _numeric_ax_oblate(a, c)
        np.testing.assert_allclose(ax_closed, ax_numeric, rtol=5e-8, atol=0.0)


def test_recompute_toggle_preserves_legacy_initialization() -> None:
    g = DustGrain(T_gas=123.0, recompute_t_rot=False)
    assert g.T_rot == 123.0


def test_constructor_accepts_new_temperature_inputs() -> None:
    g = DustGrain(
        a_eff=7.5e-8,
        rho=3.2,
        s=0.55,
        T_gas=90.0,
        n_H=45.0,
        U=1.7,
        beta_debye=0.35,
        T_dust=25.0,
        x_H=0.002,
        x_M=0.0005,
        y=0.03,
        he_fraction=0.11,
        cos2_psi=0.4,
        charge_distribution=ChargeDistribution.delta(1),
        charge_centroid_fraction=0.05,
        coeff_version="ahd09",
        recompute_t_rot=True,
    )
    assert np.isfinite(g.T_rot)
    assert g.T_rot > 0.0
