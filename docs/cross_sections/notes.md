# Notes on Cross Section Implementation

## Repository Structure Reconnaissance
-   **Existing Emissivity Code**: located in `src/rotor_emissivity/`.
-   **Key Types**: `RotorParams` (in `types.py`) carries all grain state ($A, B, n, T_{rot}, T_{int}, \mu$).
-   **Constants**: `constants.py` has $h, k_B, c$ in CGS.
-   **Partition Function**: `partition.py` has `Z(p)` which dispatches to analytic/numeric.
-   **Emissivity Implementation**: `impl_b.py` contains the reference continuum emissivity implementation using `scipy.integrate.quad`.

## Plan Implications
1.  **Module Location**: New code will be in `src/rotor_cross_section/`.
2.  **Dependencies**:
    -   Import `RotorParams` from `..rotor_emissivity.types`.
    -   Import constants from `..rotor_emissivity.constants`.
    -   Reuse `Z` from `..rotor_emissivity.partition`?
        -   AGENT.md Eq 122 matches `partition.py` implementation logic (discrete sums).
        -   However, AGENT.md Eq 115 defines $P(J,K;a)$ explicitly. I should implement `P_JK` helper.
3.  **API Design**:
    -   Functions will take `(nu, p: RotorParams)` similar to `impl_b.py`.
    -   Vectorization: `nu` vectorization is standard. `p` is scalar per call usually, but scripts handle size distributions.

## Implementation Details
-   **P(J,K)**: Needs to be robust. `exp(-beta J(J+1) - gamma K^2)`.
    -   Beta and gamma are already calculated in `partition.py` helpers. I should probably expose `_get_beta_gamma` or re-implement it to avoid private import.
    -   Accessing `_get_beta_gamma` from `rotor_emissivity.partition` is possible but "private".
    -   Better to copy the small helper or make it public in `rotor_emissivity`.
    -   *Decision*: I will re-implement the helper or make it public if I modify `rotor_emissivity`. The instructions say "Do not refactor emissivity code unless it is required". So I will re-implement the lightweight helper `get_beta_gamma` in my new module to be safe and independent.
-   **Integration**:
    -   Use `scipy.integrate.quad`.
    -   `sigma_P_parallel`: Integral over $x$ from -1 to 1. Legendre quadrature is good.
    -   `sigma_P_perp`: Integral over $J$. Finite bounds.
    -   `sigma_Q_perp`: Integral over $J$. Infinite bound. Use `quad` with infinite limit or a substitution.

## Next Steps
-   Create `src/rotor_cross_section/`
-   Implement `definitions.py` (P, Z wrappers if needed)
-   Implement `continuum.py` (the cross sections)
