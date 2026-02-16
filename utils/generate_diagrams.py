#!/usr/bin/env python3
"""Generate UML and dependency diagrams for the project."""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
from typing import Sequence


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "diagrams"
PACKAGES = ("genai_llm", "classic_ml", "utils")


def _run(cmd: Sequence[str]) -> bool:
    """Run a subprocess command from the project root."""
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    return result.returncode == 0


def _ensure_tool(name: str, hint: str) -> bool:
    """Ensure a required CLI tool is available."""
    if shutil.which(name):
        return True
    print(f"Missing {name}. {hint}", file=sys.stderr)
    return False


def main() -> int:
    """Generate UML and component diagrams."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ok = True

    if _ensure_tool("pyreverse", "Install pylint to get pyreverse."):
        cmd = [
            "pyreverse",
            "-o",
            "png",
            "-p",
            "sms_classification_clustering",
            "-d",
            str(OUTPUT_DIR),
            *PACKAGES,
        ]
        ok = _run(cmd) and ok
    else:
        ok = False

    if _ensure_tool("pydeps", "Install pydeps to generate dependency graphs."):
        if not _ensure_tool("dot", "Install Graphviz to render diagrams."):
            ok = False
        for package in PACKAGES:
            output_path = OUTPUT_DIR / f"deps_{package}.svg"
            cmd = [
                "pydeps",
                "--noshow",
                "--cluster",
                "--max-bacon",
                "2",
                "-T",
                "svg",
                "-o",
                str(output_path),
                package,
            ]
            ok = _run(cmd) and ok
    else:
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
