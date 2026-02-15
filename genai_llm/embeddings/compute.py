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

    def run(self) -> int:
        """Run embedding computation and save results to disk."""
        self._config.validate()
        output_path = pathlib.Path(self._config.output_npz)
        if output_path.exists():
            try:
                with np.load(output_path, allow_pickle=True) as data:
                    metadata = None
                    if "metadata" in data:
                        try:
                            metadata = json.loads(str(data["metadata"].item()))
                        except (ValueError, TypeError):
                            metadata = None
                    if isinstance(metadata, dict):
                        cached_provider = metadata.get("provider") or "openai"
                        cached_model = metadata.get("model")
                        cached_params = metadata.get("provider_params", {})
                        cached_limit = metadata.get("limit")
                        if (
                            cached_provider != self._config.provider
                            or cached_model != self._config.model
                            or cached_params != self._config.params
                            or cached_limit != self._config.limit
                        ):
                            print(
                                "Cached embeddings do not match provider/model/params; "
                                "recomputing.",
                                file=sys.stderr,
                            )
                        else:
                            messages = (
                                metadata.get("num_messages")
                                if isinstance(metadata, dict)
                                else len(data["texts"]) if "texts" in data else "unknown"
                            )
                            print(f"messages={messages}")
                            print(f"cached {output_path}")
                            return 0
                    else:
                        messages = len(data["texts"]) if "texts" in data else "unknown"
                        print(f"messages={messages}")
                        print(f"cached {output_path}")
                        return 0
            except (OSError, ValueError):
                print(
                    f"Failed to read cached embeddings: {output_path}", file=sys.stderr
                )
                return 1

        provider = self._provider or self._build_provider()
        if provider is None:
            return 1

        path = pathlib.Path(self._config.path)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            return 1

        data = list(iter_labeled_messages(path))
        if self._config.limit and self._config.limit > 0:
            data = data[: self._config.limit]

        texts = [text for _, text in data]
        labels = [label for label, _ in data]

        embeddings = provider.embed_texts(
            texts, self._config.model, self._config.batch_size
        )

        metadata = {
            "provider": provider.provider_id,
            "provider_name": provider.name,
            "provider_params": self._config.params,
            "model": self._config.model,
            "batch_size": self._config.batch_size,
            "limit": self._config.limit,
            "source_path": str(path),
            "num_messages": len(texts),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            embeddings=embeddings.astype(np.float32),
            labels=np.array(labels, dtype=object),
            texts=np.array(texts, dtype=object),
            metadata=np.array(json.dumps(metadata), dtype=object),
        )

        print(f"messages={len(texts)}")
        print(f"wrote {output_path}")
        return 0
