# Test Layout

- `tests/unit/`
  focused component tests with small scope
- `tests/regression/`
  numerical parity and continuum-vs-discrete regression checks
- `tests/workflows/`
  end-to-end workflow smoke tests for paper and validation entry points

Useful commands:

```bash
pytest -q
pytest -q tests/unit
pytest -q tests/regression
pytest -q tests/workflows
```

The complete suite is bounded for ordinary laptops and CI runners. In
particular, sign invariants use an explicit discrete-state cutoff rather than
constructing the adaptive 10 Angstrom line list.
