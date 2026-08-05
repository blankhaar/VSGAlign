# Continuum Cross Sections

This module (`src/rotor_cross_section`) calculates the frequency-dependent absorption cross section $\sigma_\nu$ for spinning dust grains in the continuum limit.

## Usage example

```python
import numpy as np
from dust_properties.dust_alignment import DustGrain
from rotor_polarized_emissivity.dust_bridge import params_from_dust
from rotor_cross_section.continuum import sigma_nu_total

# 1. Setup grain parameters
grain = DustGrain(a_eff=10e-8, T_gas=100.0)
p = params_from_dust(grain)

# 2. Define frequency grid
nu = np.linspace(1e9, 100e9, 200)

# 3. Calculate cross section [cm^2]
sigma = sigma_nu_total(nu, p)
```

## Equations

The total cross section is the sum of three components:

$$
\sigma_{\nu}(a) = \sigma_{\nu}^{(P,\parallel)}(a) + \sigma_{\nu}^{(P,\perp)}(a) + \sigma_{\nu}^{(Q,\perp)}(a)
$$

### P-parallel
$$
\sigma_{\nu}^{(P,\parallel)}(a) \simeq \frac{4\pi^3}{3 h c} |\mu_{\parallel}|^2 \nu \frac{J_\nu}{2B} \int_{-1}^{1} dx (1-x^2) \left[ \frac{2J_\nu+1}{2J_\nu-1} P(J_\nu-1, J_\nu x) - P(J_\nu, J_\nu x) \right]
$$
with $J_\nu = \nu / 2B$.

### P-perpendicular
$$
\sigma_{\nu}^{(P,\perp)}(a) \simeq \frac{\pi^3}{6 B h c} |\mu_{\perp}|^2 \frac{\nu}{\delta} \int_{J_{min}}^{J_{max}} dJ [1-x_\nu(J)]^2 \{ ... \}
$$
where $x_\nu(J) = (\frac{\nu}{2B J} - 1)/\delta$.

### Q-perpendicular
$$
\sigma_{\nu}^{(Q,\perp)}(a) \simeq \frac{\pi^3}{3 B h c} |\mu_{\perp}|^2 \frac{\nu}{\delta} \int_{K_\nu}^{\infty} dJ [1-(K_\nu/J)^2] [ P(J, K_\nu+1) - P(J, K_\nu) + ... ]
$$

### Population Function
$$
P(J,K;a) = \frac{1}{Z(a)} (2J+1) \exp\left[-\frac{h B J(J+1)}{k T_{rot}}\right] \exp\left[-\frac{h (A-B) K^2}{k T_{int}}\right]
$$

## Implementation Notes

-   **Numerical Stability**: The implementation fuses the Boltzmann exponents before computing the exponential to avoid overflow in the term $\exp(+\gamma K^2)$ when $\exp(-\beta J^2)$ compensates for it.
-   **Integration**: Uses `scipy.integrate.quad`.
-   **Units**: All inputs/outputs in CGS. $\sigma_\nu$ is in cm$^2$.
