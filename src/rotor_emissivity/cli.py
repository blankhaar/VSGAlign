"""Command-line entry point for continuum rotor emissivity calculations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from . import compare, impl_a, impl_b
from .types import RotorParams


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
        "--Trot", type=_positive_float, default=20.0, help="rotational temperature [K]"
    )
    parser.add_argument(
        "--Tint", type=_positive_float, default=300.0, help="internal temperature [K]"
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


def build_parser() -> argparse.ArgumentParser:
    """Construct the public command-line parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate VSGAlign continuum emissivity in CGS units."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser(
        "compare",
        help="compare continuum implementations A and B",
    )
    compare_parser.add_argument(
        "--nu-min", type=_positive_float, default=1e9, help="minimum frequency [Hz]"
    )
    compare_parser.add_argument(
        "--nu-max", type=_positive_float, default=1e12, help="maximum frequency [Hz]"
    )
    compare_parser.add_argument(
        "--n-pts", type=int, default=100, help="number of logarithmic frequency samples"
    )
    _add_rotor_arguments(compare_parser)

    eval_parser = subparsers.add_parser("eval", help="evaluate one emissivity value")
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
    return parser


def _params_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> RotorParams:
    if args.B <= args.A:
        parser.error(f"the oblate model requires B > A; received A={args.A:g}, B={args.B:g}")
    if getattr(args, "n_pts", 2) < 2:
        parser.error("--n-pts must be at least 2")
    if getattr(args, "nu_max", np.inf) <= getattr(args, "nu_min", 0.0):
        parser.error("--nu-max must exceed --nu-min")

    return RotorParams(
        n=args.n,
        A=args.A,
        B=args.B,
        mu_par=args.mu_par,
        mu_perp=args.mu_perp,
        T_rot=args.Trot,
        T_int=args.Tint,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the comparison or single-frequency evaluation command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    params = _params_from_args(args, parser)

    if args.command == "compare":
        frequencies = np.geomspace(args.nu_min, args.nu_max, args.n_pts)
        result = compare.compare_grid(params, frequencies)

        print(f"Comparison summary ({args.n_pts} points):")
        for branch, metrics in result["metrics"].items():
            print(
                f"  {branch:12s}: "
                f"max relative error={metrics['max_rel_err']:.2e}, "
                f"mean relative error={metrics['mean_rel_err']:.2e}"
            )
        return

    module = impl_a if args.impl == "a" else impl_b
    function = getattr(module, f"jnu_{args.branch}")
    value = function(args.nu, params)
    print(f"{value:.6e}")


if __name__ == "__main__":
    main()
