from vsgalign import (
    DustGrain,
    jnu_total,
    jnuq_total,
    polarized_params_from_dust,
    rotor_params_from_dust,
    sigma_nu_total,
)


def test_vsgalign_namespace_exports():
    grain = DustGrain(a_eff=5e-8, T_gas=100.0)
    params = rotor_params_from_dust(grain)
    pol_params = polarized_params_from_dust(grain)

    assert params.A > 0
    assert pol_params.A > 0
    assert callable(jnu_total)
    assert callable(jnuq_total)
    assert callable(sigma_nu_total)
