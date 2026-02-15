"""Entry point for running classic_ml or genai_llm pipelines from JSON config."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

from classic_ml import ClassicMLFacade
from classic_ml.config import ClassicMLConfig
from genai_llm import GenaiLLMFacade
from genai_llm.config import GenaiLLMConfig


def load_json_config(path: pathlib.Path) -> Dict[str, Any]:
    """Load a JSON configuration file from disk."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}")


def main(argv: Optional[List[str]] = None) -> int:
    """Run the selected pipeline using the provided JSON config."""
    parser = argparse.ArgumentParser(description="Run SMS analysis pipelines.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config (classic_ml or genai_llm).",
    )
    args = parser.parse_args(argv)

    try:
        config_data = load_json_config(pathlib.Path(args.config))
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    package = config_data.get("package")
    if package == "classic_ml":
        try:
            config = ClassicMLConfig.from_dict(config_data)
            facade = ClassicMLFacade(config)
        except ValueError as exc:
            print(f"Config error: {exc}", file=sys.stderr)
            return 1
        return facade.run()
    if package == "genai_llm":
        try:
            config = GenaiLLMConfig.from_dict(config_data)
            facade = GenaiLLMFacade(config)
        except ValueError as exc:
            print(f"Config error: {exc}", file=sys.stderr)
            return 1
        return facade.run()

    print("Config must include package: classic_ml or genai_llm", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
