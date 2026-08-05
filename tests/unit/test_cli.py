"""Command-line parsing and unit-contract checks."""

import math

import pytest

from rotor_emissivity.cli import main as emissivity_main
from rotor_polarized_emissivity.cli import main as polarized_main


def test_emissivity_cli_uses_cgs_dipole_defaults(capsys):
    emissivity_main(["eval", "--nu", "1e10"])
    assert float(capsys.readouterr().out) >= 0.0


def test_polarized_cli_uses_cgs_dipole_defaults(capsys):
    polarized_main(["eval", "--nu", "1e10"])
    assert math.isfinite(float(capsys.readouterr().out))


@pytest.mark.parametrize("entrypoint", [emissivity_main, polarized_main])
def test_cli_rejects_non_oblate_rotational_constants(entrypoint):
    with pytest.raises(SystemExit):
        entrypoint(["eval", "--nu", "1e10", "--A", "2e9", "--B", "1e9"])
