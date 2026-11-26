"""Command-line entry point."""

from __future__ import annotations

import argparse

from qtb import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qtb", description=__doc__)
    parser.add_argument("--version", action="version", version=f"qtb {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0
