# Paper figures

## Final-manuscript authority

The figure authority is the final A&A revision-3 source submission dated
2026-08-04. It is newer than arXiv v1. The accepted copies and SHA-256 records
for VSGAlign's two submitted outputs are described in
[`paper/REFERENCE_FIGURES.md`](../paper/REFERENCE_FIGURES.md).

## Full reproduction

From the repository root:

```bash
python -m pip install -r requirements/reproduction.txt -e .
python -m paper.reproduce_all --output-dir reproduced-figures
```

The pinned environment uses CPython 3.12.10. An ordinary editable development
install also supports the workflow, but exact PDF bytes can vary with plotting
and PDF-library versions.

## Final-manuscript scripts

- `paper/plot_single_grain_overview.py`: single-grain `eta_I`, `eta_Q`, and
  polarization fraction. Its combined PDF appears in the final manuscript.
- `paper/plot_size_distribution.py`: size-distribution-integrated emissivity
  and polarization fraction. Its combined PDF appears in the final manuscript.

## Diagnostic scripts

The following scripts are scientifically useful but their figures are not in
the final revision-3 manuscript:

- `paper/plot_cross_sections.py`: spinning-dust and vibrational cross sections
  for representative grain sizes;
- `paper/plot_cnm_cross_sections.py`: CNM per-H cross-section comparison.

Run all maintained diagnostics with:

```bash
python -m paper.reproduce_all --include-diagnostics
```

## Outside this repository

The final manuscript also contains a grain-angle schematic and a
ground-state-alignment PNG. The schematic generator is manuscript-specific and
is preserved with the manuscript project. No generator for the alignment PNG
was included in the owner-identified final implementation, so VSGAlign does not
claim to regenerate it.

## Lightweight workflow test

```bash
pytest -q tests/workflows/test_paper_scripts_smoke.py
```

This checks that the single-grain workflow runs and writes all expected PDFs.
