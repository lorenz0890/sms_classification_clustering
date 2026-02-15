#!/usr/bin/env python3
"""Clean classic_ml and genai_llm cache directories."""
from __future__ import annotations

import argparse
from pathlib import Path

from utils.cache_utils import clear_dir


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Clear classic/genai cache folders.")
    parser.add_argument(
        "--classic-cache",
        default="classic_ml/cache",
        help="Path to the classic_ml cache directory.",
    )
    parser.add_argument(
        "--genai-cache",
        default="genai_llm/cache",
        help="Path to the genai_llm cache directory.",
    )
    args = parser.parse_args()

    clear_dir(Path(args.classic_cache))
    clear_dir(Path(args.genai_cache))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
