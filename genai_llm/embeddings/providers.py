"""Embedding provider strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
import re
import sys
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np
from openai import OpenAI


class EmbeddingProvider(ABC):
    """Abstract embedding provider interface."""

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Return the provider identifier."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider display name."""

    @abstractmethod
    def embed_texts(
        self, texts: Sequence[str], model: str, batch_size: int
    ) -> np.ndarray:
        """Embed texts using the configured provider."""


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by the OpenAI API."""

    provider_id = "openai"
    name = "OpenAI"

    def __init__(self, api_key: str, params: Dict[str, Any]) -> None:
        """Initialize the provider with an API key and parameters."""
        self._client = OpenAI(api_key=api_key)
        self._params = {
            key: value for key, value in params.items() if key != "api_key_env"
        }

    def embed_texts(
        self, texts: Sequence[str], model: str, batch_size: int
    ) -> np.ndarray:
        """Embed texts in batches using the OpenAI API."""
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._client.embeddings.create(
                model=model, input=batch, **self._params
            )
            embeddings.extend([item.embedding for item in resp.data])
        return np.array(embeddings, dtype=np.float32)


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by the Gemini API."""

    provider_id = "gemini"
    name = "Google Gemini"

    def __init__(self, api_key: str, params: Dict[str, Any]) -> None:
        """Initialize the provider with an API key and parameters."""
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError(
                "Missing google-generativeai. Install it to use the Gemini provider."
            ) from exc
        genai.configure(api_key=api_key)
        self._genai = genai
        cleaned_params = {
            key: value for key, value in params.items() if key != "api_key_env"
        }
        self._max_retries = int(cleaned_params.pop("max_retries", 5))
        self._retry_base_seconds = float(cleaned_params.pop("retry_base_seconds", 10.0))
        self._retry_max_seconds = float(cleaned_params.pop("retry_max_seconds", 60.0))
        self._params = cleaned_params

    def _extract_embeddings(self, response: Any) -> list[list[float]]:
        """Extract embedding vectors from a Gemini response."""
        embedding = None
        if isinstance(response, dict):
            embedding = response.get("embedding")
        elif hasattr(response, "embedding"):
            embedding = response.embedding
        if isinstance(embedding, list):
            if not embedding:
                return []
            first = embedding[0]
            if isinstance(first, list):
                return embedding
            if isinstance(first, (int, float)):
                return [embedding]
        raise RuntimeError("Unexpected Gemini embedding response format.")

    def _is_quota_error(self, exc: Exception) -> bool:
        """Return True if the exception looks like a quota/rate-limit error."""
        name = exc.__class__.__name__.lower()
        message = str(exc).lower()
        if "resourceexhausted" in name:
            return True
        if "quota" in message or "rate limit" in message or "429" in message:
            return True
        return False

    def _retry_sleep(self, exc: Exception, attempt: int) -> float:
        """Compute a sleep duration before retrying."""
        message = str(exc)
        match = re.search(r"retry in ([0-9.]+)s", message, re.IGNORECASE)
        if match:
            try:
                return min(self._retry_max_seconds, float(match.group(1)))
            except ValueError:
                pass
        delay = self._retry_base_seconds * (2**attempt)
        return min(self._retry_max_seconds, delay)

    def embed_texts(
        self, texts: Sequence[str], model: str, batch_size: int
    ) -> np.ndarray:
        """Embed texts in batches using the Gemini API."""
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            kwargs = {"model": model, "content": list(batch), **self._params}
            try:
                attempt = 0
                while True:
                    try:
                        resp = self._genai.embed_content(**kwargs)
                        embeddings.extend(self._extract_embeddings(resp))
                        break
                    except Exception as exc:
                        if self._is_quota_error(exc) and attempt < self._max_retries:
                            sleep_seconds = self._retry_sleep(exc, attempt)
                            print(
                                f"Gemini quota hit; retrying in {sleep_seconds:.1f}s.",
                                file=sys.stderr,
                            )
                            time.sleep(sleep_seconds)
                            attempt += 1
                            continue
                        raise
            except Exception as exc:
                print(
                    "Gemini batch embedding failed; falling back to single requests.",
                    file=sys.stderr,
                )
                print(f"Batch error: {exc}", file=sys.stderr)
                for text in batch:
                    attempt = 0
                    while True:
                        try:
                            resp = self._genai.embed_content(
                                model=model, content=text, **self._params
                            )
                            embeddings.extend(self._extract_embeddings(resp))
                            break
                        except Exception as inner_exc:
                            if (
                                self._is_quota_error(inner_exc)
                                and attempt < self._max_retries
                            ):
                                sleep_seconds = self._retry_sleep(inner_exc, attempt)
                                print(
                                    f"Gemini quota hit; retrying in {sleep_seconds:.1f}s.",
                                    file=sys.stderr,
                                )
                                time.sleep(sleep_seconds)
                                attempt += 1
                                continue
                            raise
        return np.array(embeddings, dtype=np.float32)


def _resolve_api_key(params: Dict[str, Any], env_keys: Sequence[str]) -> Optional[str]:
    """Resolve an API key from params or environment."""
    override_env = params.get("api_key_env")
    if isinstance(override_env, str) and override_env:
        env_keys = [override_env]
    for key in env_keys:
        value = os.environ.get(key)
        if value:
            return value
    return None


def build_embedding_provider(
    provider: str, params: Dict[str, Any]
) -> EmbeddingProvider:
    """Factory for embedding providers based on the provider id."""
    provider_id = provider.lower()
    if provider_id == "openai":
        api_key = _resolve_api_key(params, ["OPENAI_API_KEY"])
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
        return OpenAIEmbeddingProvider(api_key=api_key, params=params)
    if provider_id in {"gemini", "google"}:
        api_key = _resolve_api_key(
            params, ["GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GEMINI_API_KEY"]
        )
        if not api_key:
            raise RuntimeError(
                "Missing GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )
        return GeminiEmbeddingProvider(api_key=api_key, params=params)
    raise ValueError(f"Unsupported embeddings.provider: {provider}")
