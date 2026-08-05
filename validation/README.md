# Validation Visuals

This directory is for non-paper visual inspection workflows.

The intent is:

- keep paper outputs in `paper/figures/`
- keep non-paper inspection plots out of the repository root
- write all temporary validation artifacts into ignored `artifacts/`

## Continuum vs discrete

```bash
PYTHONPATH=src python -m validation.continuum_vs_discrete --output-dir artifacts/validation
```

This generates:

- a four-panel comparison plot
- a JSON file with branch-by-branch relative-error metrics

