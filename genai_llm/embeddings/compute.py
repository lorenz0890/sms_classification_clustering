"""Embedding computation workflow for SMS data."""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Optional

import numpy as np

from utils import iter_labeled_messages

from .providers import EmbeddingProvider, build_embedding_provider
from ..config import EmbeddingsConfig


class EmbeddingComputer:
    """Compute and persist embeddings for the SMS dataset."""

    def __init__(
        self, config: EmbeddingsConfig, provider: Optional[EmbeddingProvider] = None
    ) -> None:
        """Initialize the computation with configuration and provider."""
        self._config = config
        self._provider = provider

    def _build_provider(self) -> Optional[EmbeddingProvider]:
        """Build the default provider from environment configuration."""
        try:
            return build_embedding_provider(self._config.provider, self._config.params)
        except (ValueError, RuntimeError) as exc:
            print(str(exc), file=sys.stderr)
            return None

    def _load_cached_metadata(
        self, data: np.lib.npyio.NpzFile
    ) -> Optional[dict[str, object]]:
        """Load cached embedding metadata if available."""
        if "metadata" not in data:
            return None
        try:
            return json.loads(str(data["metadata"].item()))
        except (ValueError, TypeError):
            return None

    def _cache_matches(self, metadata: dict[str, object]) -> bool:
        """Check if cached metadata matches the current configuration."""
        cached_provider = metadata.get("provider") or "openai"
        cached_model = metadata.get("model")
        cached_params = metadata.get("provider_params", {})
        cached_limit = metadata.get("limit")
        return (
            cached_provider == self._config.provider
            and cached_model == self._config.model
            and cached_params == self._config.params
            and cached_limit == self._config.limit
        )

    def _try_use_cache(self, output_path: pathlib.Path) -> Optional[int]:
        """Return 0 if cache can be used, 1 on error, None to recompute."""
        if not output_path.exists():
            return None
        try:
            with np.load(output_path, allow_pickle=True) as data:
                metadata = self._load_cached_metadata(data)
                if isinstance(metadata, dict):
                    if self._cache_matches(metadata):
                        messages = metadata.get("num_messages")
                        print(f"messages={messages}")
                        print(f"cached {output_path}")
                        return 0
                    print(
                        "Cached embeddings do not match provider/model/params; "
                        "recomputing.",
                        file=sys.stderr,
                    )
                    return None
                messages = len(data["texts"]) if "texts" in data else "unknown"
                print(f"messages={messages}")
                print(f"cached {output_path}")
                return 0
        except (OSError, ValueError):
            print(f"Failed to read cached embeddings: {output_path}", file=sys.stderr)
            return 1

    def _load_labeled_data(self, path: pathlib.Path) -> tuple[list[str], list[str]]:
        """Load labels/texts from the dataset with optional limiting."""
        data = list(iter_labeled_messages(path))
        if self._config.limit and self._config.limit > 0:
            data = data[: self._config.limit]
        texts = [text for _, text in data]
        labels = [label for label, _ in data]
        return labels, texts

    def _build_metadata(
        self, provider: EmbeddingProvider, path: pathlib.Path, texts: list[str]
    ) -> dict[str, object]:
        """Build metadata for stored embeddings."""
        return {
            "provider": provider.provider_id,
            "provider_name": provider.name,
            "provider_params": self._config.params,
            "model": self._config.model,
            "batch_size": self._config.batch_size,
            "limit": self._config.limit,
            "source_path": str(path),
            "num_messages": len(texts),
        }

    def _write_embeddings(
        self,
        output_path: pathlib.Path,
        embeddings: np.ndarray,
        labels: list[str],
        texts: list[str],
        metadata: dict[str, object],
    ) -> None:
        """Persist embeddings and metadata to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            embeddings=embeddings.astype(np.float32),
            labels=np.array(labels, dtype=object),
            texts=np.array(texts, dtype=object),
            metadata=np.array(json.dumps(metadata), dtype=object),
        )

    def run(self) -> int:
        """Run embedding computation and save results to disk."""
        self._config.validate()
        output_path = pathlib.Path(self._config.output_npz)
        cache_status = self._try_use_cache(output_path)
        if cache_status is not None:
            return cache_status

        provider = self._provider or self._build_provider()
        if provider is None:
            return 1

        path = pathlib.Path(self._config.path)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            return 1

        labels, texts = self._load_labeled_data(path)

        embeddings = provider.embed_texts(
            texts, self._config.model, self._config.batch_size
        )

        metadata = self._build_metadata(provider, path, texts)
        self._write_embeddings(output_path, embeddings, labels, texts, metadata)

        print(f"messages={len(texts)}")
        print(f"wrote {output_path}")
        return 0
