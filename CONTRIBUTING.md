# Contributing

Contributions are welcome through focused issues and pull requests.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
```

Use Python 3.10 or newer. Keep public behavior behind the `vsgalign` facade,
state physical units in public interfaces, and add a regression test for every
numerical correction. Validation implementations may remain separate when
their independence is scientifically useful.

## Scientific changes

A numerical change should document:

- the equation, approximation, or physical convention affected;
- the expected numerical impact and relevant parameter range;
- the comparison or invariant used to validate it; and
- whether checked-in reference figures should change.

Do not silence integration or floating-point warnings without first showing
that the result and its error control remain valid.

## Before opening a pull request

```bash
pytest -q
python -m compileall src paper validation
python -m build
python -m twine check dist/*
```

Generated wheels, source distributions, caches, and temporary figures are not
tracked. See [docs/releasing.md](docs/releasing.md) for the release checklist.
