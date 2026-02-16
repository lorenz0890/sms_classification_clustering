#!/usr/bin/env python3
"""Generate code metrics reports for the project."""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
from typing import Sequence


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "metrics"
PACKAGES = ("genai_llm", "classic_ml", "utils")


def _run_to_file(cmd: Sequence[str], output_path: pathlib.Path) -> bool:
    """Run a subprocess command and write stdout/stderr to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode == 0


def _ensure_tool(name: str, hint: str) -> bool:
    """Ensure a required CLI tool is available."""
    if shutil.which(name):
        return True
    print(f"Missing {name}. {hint}", file=sys.stderr)
    return False


def main() -> int:
    """Generate radon and pylint design reports."""
    ok = True

    if _ensure_tool("radon", "Install radon to generate complexity metrics."):
        ok = (
            _run_to_file(
                ["radon", "cc", "-s", "-a", *PACKAGES],
                OUTPUT_DIR / "radon_cc.txt",
            )
            and ok
        )
        ok = (
            _run_to_file(
                ["radon", "mi", "-s", "."],
                OUTPUT_DIR / "radon_mi.txt",
            )
            and ok
        )
        ok = (
            _run_to_file(
                ["radon", "raw", "-s", "."],
                OUTPUT_DIR / "radon_raw.txt",
            )
            and ok
        )
    else:
        ok = False

    if _ensure_tool("pylint", "Install pylint to generate design reports."):
        ok = (
            _run_to_file(
                ["pylint", "--reports=y", "--enable=design", *PACKAGES],
                OUTPUT_DIR / "pylint_design.txt",
            )
            and ok
        )
    else:
        ok = False

    if ok:
        print(f"wrote metrics to {OUTPUT_DIR}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
