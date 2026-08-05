# Architecture

## Public Name

The public project and distribution name is `VSGAlign`.

For convenience, the repository now exposes a small top-level namespace package:

- `vsgalign`
  convenience facade for the most common public entry points

## Supported Layers

The repository is organized around one main scientific path and two validation layers.

### Grain and thermodynamics

- `dust_properties.DustGrain`
  grain geometry, dipole properties, environment parameters, and first-principles rotational temperature
- `spindust_trot`
  low-level support code for the single-temperature rotational closure

### Canonical continuum calculations

- `rotor_emissivity`
  unpolarized continuum emissivity
- `rotor_polarized_emissivity`
  polarized continuum emissivity and the dust-to-rotor alignment bridge
- `rotor_cross_section`
  continuum absorption cross sections

### Validation path

- `discrete_emissivity`
  discrete line-list and binned-spectrum machinery used to validate the continuum approximation
- `validation/`
  small repo-local workflows for non-paper inspection plots; generated artifacts belong in ignored `artifacts/`

## Canonical Entry Points

Use the package-level exports first.

- `vsgalign`
- `dust_properties.DustGrain`
- `dust_emissivity.rotor_params_from_dust`
- `rotor_emissivity.jnu_total`
- `rotor_polarized_emissivity.jnuq_total`
- `rotor_polarized_emissivity.params_from_dust`
- `rotor_cross_section.sigma_nu_total`

## Comparison Implementations

Some modules exist primarily as cross-checks rather than primary user APIs.

- `rotor_emissivity.impl_a`
- `rotor_emissivity.impl_b`
- `rotor_polarized_emissivity.continuum_impl_a`
- `rotor_polarized_emissivity.continuum_impl_b`

The package-level API currently exports the `impl_b` continuum path as the canonical public interface. The `impl_a` variants remain useful for tests and derivation parity checks.

## Paper Workflow

The supported paper workflow lives in `paper/`.

- Each figure script exposes a `main()` function.
- `paper/reproduce_all.py` runs the two VSGAlign figure families used by the
  final 2026-08-04 manuscript submission.
- `--include-diagnostics` additionally runs cross-section plots that are not in
  that final manuscript.
- These scripts are meant to be run from a repository checkout after editable install.

The public API is the `vsgalign` facade and its documented command-line entry
points. Descriptive top-level scientific packages remain importable for
advanced and validation use, but their internal helpers do not carry the same
stability promise.
