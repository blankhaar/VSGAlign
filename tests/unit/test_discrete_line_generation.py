
import pytest
import numpy as np
from rotor_emissivity.types import RotorParams
from discrete_emissivity import rotor_lines

def test_lines_basic():
    p = RotorParams(n=1, A=1e9, B=2e9, mu_par=1e-18, mu_perp=1e-18, T_rot=10, T_int=10)
    lines = rotor_lines.line_list(p, Jmax=5)
    
    assert len(lines) > 0
    assert np.all(lines['nu'] > 0)
    assert np.all(lines['weight'] >= 0)
    
    # Check branches
    branches = np.unique(lines['branch'])
    assert 'P_parallel' in branches
    assert 'P_perp' in branches
    # Q might be missing if low J?
    # Q condition: K+1 <= J.
    # J=1, K=0 -> 1 <= 1 (Yes).
    # So Q should exist.
    assert 'Q_perp' in branches

def test_lines_energy_order():
    # Check that nu matches manual calc for simple transition.
    # P_parallel: J=1, K=0 -> J=0, K=0.
    # E(1,0) - E(0,0) = B*1*2 - 0 = 2B.
    # nu = 2B.
    p = RotorParams(n=1, A=1e9, B=2e9, mu_par=1e-18, mu_perp=0, T_rot=10, T_int=10)
    lines = rotor_lines.line_list(p, Jmax=1)
    
    # K=0, J=1 -> K=0, J=0.
    # Should be one line at nu = 2*2e9 = 4e9.
    
    # Search for P_parallel
    mask = lines['branch'] == 'P_parallel'
    subset = lines[mask]
    
    assert len(subset) == 1
    assert np.isclose(subset[0]['nu'], 4e9)
