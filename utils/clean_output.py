#!/usr/bin/env python3
"""Clean the output directory."""
from __future__ import annotations

import argparse
from pathlib import Path

from utils.cache_utils import clear_dir


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Clear output folder contents.")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Path to the output directory.",
    )
    args = parser.parse_args()

    clear_dir(Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
