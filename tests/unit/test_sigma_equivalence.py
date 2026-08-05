import numpy as np

from dust_properties.dust_alignment import DustGrain
from rotor_polarized_emissivity.dust_bridge import params_from_dust


def test_sigma_computer_equivalence():
    """The cached integer-J values match the direct discrete calculation."""
    grain = DustGrain(a_eff=1.0e-7, T_gas=100.0)
    p = params_from_dust(grain)

    assert callable(p.sigma), "sigma should be a callable SigmaComputer"
    assert not isinstance(p.sigma, float), "sigma should not be a float 0.0"

    for J in [10, 100, 500, 1000]:
        Ks = np.linspace(0, J, 10, dtype=int)
        for K in Ks:
            val_ref = grain.sigma_JK(J, K, w=1.0)
            val_fast = p.get_sigma(J, K)

            np.testing.assert_allclose(
                val_fast,
                val_ref,
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"Mismatch at J={J}, K={K}",
            )


def test_sigma_computer_is_smooth_between_integer_J_values():
    """Continuum calls interpolate without changing either integer endpoint."""
    grain = DustGrain(a_eff=5.0e-8, T_gas=100.0)
    p = params_from_dust(grain)

    J_low = 100
    J_high = 101
    K_fraction = 0.4
    low = p.get_sigma(J_low, K_fraction * J_low)
    high = p.get_sigma(J_high, K_fraction * J_high)
    midpoint = p.get_sigma(100.5, K_fraction * 100.5)

    np.testing.assert_allclose(low, grain.sigma_JK(J_low, K_fraction * J_low))
    np.testing.assert_allclose(high, grain.sigma_JK(J_high, K_fraction * J_high))
    assert min(low, high) <= midpoint <= max(low, high)


def test_sigma_uses_revised_positive_formalism_sign():
    """Lock in sigma = (y P2 - <y P2>_K) / (1 + x)."""
    grain = DustGrain(a_eff=5.0e-8, T_gas=100.0)
    p = params_from_dust(grain)

    J = 100
    K = J
    gamma = (grain.Brot - grain.Arot) * 6.62607015e-27 / (
        1.380649e-16 * grain.critical_temperature()
    )
    k_values = np.arange(J + 1, dtype=float)
    rho = np.exp(gamma * k_values**2 - gamma * J**2)
    weights = np.full(J + 1, 2.0)
    weights[0] = 1.0
    p2_values = 0.5 * (3.0 * (k_values / J) ** 2 - 1.0)
    y_p2_average = np.sum(grain.y_param() * p2_values * rho * weights) / np.sum(
        rho * weights
    )
    expected = (grain.y_param() - y_p2_average) / (1.0 + grain.x_ratio())

    np.testing.assert_allclose(grain.sigma_JK(J, K), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(p.get_sigma(J, K), expected, rtol=1e-12, atol=1e-12)
