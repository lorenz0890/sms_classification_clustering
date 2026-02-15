"""Path resolution helpers for genai_llm outputs and cache."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass


@dataclass
class GenaiPathResolver:
    """Resolve cache/output paths relative to configured base directories."""

    cache_dir: pathlib.Path
    output_dir: pathlib.Path

    @classmethod
    def from_config(cls, cache_dir: str, output_dir: str) -> "GenaiPathResolver":
        """Build a resolver from config values."""
        return cls(pathlib.Path(cache_dir), pathlib.Path(output_dir))

    def ensure_dirs(self) -> None:
        """Ensure the cache/output base directories exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def resolve_cache_path(self, path: str) -> str:
        """Resolve a cache path relative to the cache directory."""
        path_obj = pathlib.Path(path)
        if path_obj.is_absolute():
            return str(path_obj)
        return str(self.cache_dir / path_obj)

    def resolve_output_path(self, path: str) -> str:
        """Resolve an output path relative to the output directory."""
        path_obj = pathlib.Path(path)
        if path_obj.is_absolute():
            return str(path_obj)
        return str(self.output_dir / path_obj)
