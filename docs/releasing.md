# Release checklist

1. Run the complete test suite on each supported Python version.
2. Reproduce the two VSGAlign figures used by the final manuscript with the
   pinned environment in `requirements/reproduction.txt`.
3. Compare the rendered figures with the accepted references described in
   `paper/REFERENCE_FIGURES.md`.
4. Set the release version and actual release date in `pyproject.toml` and
   `CITATION.cff`.
5. Add the public repository URL to project and citation metadata once it
   exists.
6. Build from a clean checkout and run `python -m twine check dist/*`.
7. Install the wheel into a fresh environment and run the documented API and
   command-line examples.
8. Attach newly built artifacts to the release; never reuse historical files
   from an earlier source state.

The repository does not track `dist/`. Release artifacts should be generated
from the tagged commit and published through the hosting service.
