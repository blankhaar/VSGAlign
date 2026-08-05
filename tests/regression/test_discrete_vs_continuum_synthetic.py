
import pytest
import numpy as np
from rotor_emissivity.types import RotorParams
from rotor_emissivity import impl_a
from discrete_emissivity import rotor_lines, spectrum, compare

def test_synthetic_agreement():
    """
    Test that discrete sum agrees with continuum for a synthetic "heavy" rotor
    where density of states is high (semi-classical limit).
    """
    # Heavy rotor: B is small, T is large-ish.
    # B ~ 1e8 Hz. T ~ 100 K.
    # h B / k T ~ 6e-27 * 1e8 / (1e-16 * 100) ~ 6e-19 / 1e-14 ~ 6e-5 (Very small beta).
    # Good for continuum.
    
    p = RotorParams(
        n=1.0,
        A=1e8,
        B=2e8, # B>A, oblate
        mu_par=1e-18,
        mu_perp=1e-18,
        T_rot=100.0,
        T_int=100.0 # B>A, so Tint=20 was unstable (beta < gamma). T=100 ensures beta > gamma.
    )
    
    # Frequency range covering the peak
    # Peak J ~ sqrt(k T / h B) ~ sqrt(1/6e-5) ~ 130.
    # nu ~ 2 B J ~ 2 * 2e8 * 130 ~ 5e10 Hz (50 GHz).
    nu_grid = np.linspace(20e9, 80e9, 50)
    
    # Generate discrete
    # We need Jmax >> peak J.
    lines = rotor_lines.line_list(p, Jmax=400)
    
    # Bin
    d_spec = spectrum.spectrum_binned(lines, nu_grid)
    
    # Continuum
    c_spec = {}
    c_spec['total'] = impl_a.jnu_total(nu_grid, p)
    
    # Compare
    # Expect rough agreement. Discrete spectrum is jagged if bins are small.
    # But total power should be close.
    # Let's check mean relative error?
    # Or smoothen?
    # Rel error elementwise might be high if a bin is empty between lines.
    # With 50 GHz / 50 bins = 1 GHz bin width.
    # Line spacing ~ 2 B ~ 4e8 Hz = 0.4 GHz.
    # So we have ~2.5 lines per bin. Should be somewhat smooth.
    
    err = compare.rel_err(d_spec['total'], c_spec['total'])
    mean_err = np.mean(err)
    
    print(f"Mean rel err: {mean_err}")
    assert mean_err < 0.2 # 20% tolerance for binning noise
