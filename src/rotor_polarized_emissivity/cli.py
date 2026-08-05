"""Command-line entry point for polarized rotor emissivity calculations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import numpy as np

from . import compare, continuum_impl_a, continuum_impl_b, discrete
from .types import RotorPolParams


def _positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result


def _add_rotor_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n", type=_nonnegative_float, default=1.0, help="rotor density [cm^-3]")
    parser.add_argument(
        "--A", type=_positive_float, default=1e9, help="parallel rotational constant [Hz]"
    )
    parser.add_argument(
        "--B",
        type=_positive_float,
        default=2e9,
        help="perpendicular rotational constant [Hz]; must exceed A",
    )
    parser.add_argument(
        "--T-rot", type=_positive_float, default=100.0, help="rotational temperature [K]"
    )
    parser.add_argument(
        "--T-int", type=_positive_float, default=100.0, help="internal temperature [K]"
    )
    parser.add_argument(
        "--mu-par",
        type=_nonnegative_float,
        default=1e-18,
        help="parallel dipole component [esu cm]",
    )
    parser.add_argument(
        "--mu-perp",
        type=_nonnegative_float,
        default=1e-18,
        help="perpendicular dipole component [esu cm]",
    )
    parser.add_argument("--sigma", type=float, default=0.1, help="constant rank-2 alignment moment")
    parser.add_argument("--chi", type=float, default=np.pi / 2, help="viewing angle [radians]")


def build_parser() -> argparse.ArgumentParser:
    """Construct the public command-line parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate VSGAlign polarized emissivity in CGS units."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("eval", help="evaluate one polarized emissivity value")
    eval_parser.add_argument("--mode", choices=("continuum", "discrete"), default="continuum")
    eval_parser.add_argument(
        "--impl", choices=("a", "b"), default="b", help="continuum implementation"
    )
    eval_parser.add_argument(
        "--branch",
        choices=("P_parallel", "P_perp", "Q_perp", "total"),
        default="total",
        help="rotational branch to evaluate",
    )
    eval_parser.add_argument("--nu", type=_positive_float, required=True, help="frequency [Hz]")
    _add_rotor_arguments(eval_parser)

    compare_parser = subparsers.add_parser(
        "compare",
        help="compare polarized continuum implementations A and B",
    )
    compare_parser.add_argument(
        "--nu-min", type=_positive_float, default=1e9, help="minimum frequency [Hz]"
    )
    compare_parser.add_argument(
        "--nu-max", type=_positive_float, default=100e9, help="maximum frequency [Hz]"
    )
    compare_parser.add_argument(
        "--n-nu", type=int, default=100, help="number of logarithmic frequency samples"
    )
    _add_rotor_arguments(compare_parser)
    return parser


def _params_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> RotorPolParams:
    if args.B <= args.A:
        parser.error(f"the oblate model requires B > A; received A={args.A:g}, B={args.B:g}")
    if getattr(args, "n_nu", 2) < 2:
        parser.error("--n-nu must be at least 2")
    if getattr(args, "nu_max", np.inf) <= getattr(args, "nu_min", 0.0):
        parser.error("--nu-max must exceed --nu-min")

    return RotorPolParams(
        n=args.n,
        A=args.A,
        B=args.B,
        mu_par=args.mu_par,
        mu_perp=args.mu_perp,
        T_rot=args.T_rot,
        T_int=args.T_int,
        sigma=args.sigma,
        chi=args.chi,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run a polarized-emissivity evaluation or implementation comparison."""
    parser = build_parser()
    args = parser.parse_args(argv)
    params = _params_from_args(args, parser)

    if args.command == "compare":
        frequencies = np.geomspace(args.nu_min, args.nu_max, args.n_nu)
        print(json.dumps(compare.compare_continuum_impls(params, frequencies), indent=2))
        return

    if args.mode == "continuum":
        module = continuum_impl_a if args.impl == "a" else continuum_impl_b
        function = getattr(module, f"jnuq_{args.branch}")
        value = float(function(args.nu, params))
    else:
        # Bin a narrow, three-point interval around the requested frequency.
        # This path is a reference approximation; it is not line-profile fitting.
        frequencies = np.linspace(args.nu * 0.99, args.nu * 1.01, 3)
        lines = discrete.polarized_line_list(params)
        value = float(discrete.spectrum_binned_pol(lines, frequencies)[args.branch][1])

    print(f"{value:.6e}")


if __name__ == "__main__":
    main()
