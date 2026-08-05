import unittest
import numpy as np
from rotor_emissivity.types import RotorParams
from rotor_cross_section.definitions import P_JK, Z

class TestDefinitions(unittest.TestCase):
    def setUp(self):
        # Setup standard oblate grain parameters
        # B > A
        self.p = RotorParams(
            n=1.0,
            A=1e9,
            B=2e9,
            mu_par=0.0,
            mu_perp=0.0,
            T_rot=20.0,
            T_int=15.0,
            Z_mode="auto"
        )

    def test_Z_positive_finite(self):
        z_val = Z(self.p)
        self.assertGreater(z_val, 0.0)
        self.assertFalse(np.isinf(z_val))
        
    def test_P_JK_normalization(self):
        """
        Sum P(J, K) over J and K should approach 1.
        """
        limit_J = 200 # Sufficient for these params?
        # beta approx 6e-27 * 2e9 / (1e-16 * 20) ~ 6e-18 / 2e-15 ~ 0.003
        # J(J+1) ~ 300 * 0.003 ~ 1. exp(-1) is fine. 
        # Actually beta ~ 3e-3. J ~ 100 -> J^2 ~ 1e4 -> beta J^2 ~ 30. exp(-30) is small.
        # So J=200 is plenty.
        
        sum_P = 0.0
        for J in range(limit_J):
            for K in range(-J, J+1):
                sum_P += P_JK(J, K, self.p)
                
        self.assertAlmostEqual(sum_P, 1.0, delta=1e-3)

    def test_P_JK_vectorization(self):
        J = np.array([10, 20])
        K = np.array([5, 10])
        val = P_JK(J, K, self.p)
        self.assertEqual(val.shape, (2,))
        
        val0 = P_JK(10, 5, self.p)
        self.assertAlmostEqual(val[0], val0)

if __name__ == '__main__':
    unittest.main()
