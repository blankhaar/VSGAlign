# Testing

## Run Everything

```bash
pytest -q
```

## Run By Group

```bash
pytest -q tests/unit
pytest -q tests/regression
pytest -q tests/workflows
```

## What The Tests Are Checking

- package import and adapter sanity
- continuum implementation parity
- cross-section definitions and normalization
- dust-temperature model behavior
- sigma-bridge equivalence
- smooth continuum evaluation of the cached alignment moment
- discrete-vs-continuum agreement
- warning-free execution of the documented polarized quickstart
- command-line unit and parameter validation
- paper script smoke coverage

## Test Layout

- `tests/unit/`
  small-scope component behavior and local invariants
- `tests/regression/`
  numerical parity, continuum/discrete agreement, and release-facing regression checks
- `tests/workflows/`
  smoke tests for paper and validation entry points

## Most Important Release-Facing Tests

- `tests/regression/test_publication_continuum_validation.py`
  validates the continuum approximation against the discrete polarized spectrum for a representative 5 Angstrom grain
- `tests/unit/test_sigma_equivalence.py`
  checks exact integer-J agreement with `DustGrain.sigma_JK` and smooth
  interpolation between discrete levels
- `tests/regression/test_public_quickstart.py`
  requires the documented polarized API example to finish without SciPy
  integration warnings
- `tests/regression/test_continuum_A_vs_B.py`
  checks that the two polarized continuum implementations stay numerically aligned
- `tests/workflows/test_paper_scripts_smoke.py`
  verifies that the single-grain paper plotting workflow runs end to end
- `tests/workflows/test_validation_visuals_smoke.py`
  verifies that the non-paper continuum-vs-discrete visual report can be generated into a temporary output directory

## Suggested Pre-Release Checklist

```bash
pytest -q
python -m paper.reproduce_all --output-dir reproduced-figures
python -m build
python -m twine check dist/*
```
