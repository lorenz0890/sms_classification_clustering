"""Formatting and naming helpers for cluster outputs."""

from __future__ import annotations

import pathlib
from typing import Any, Mapping, Optional


def normalize_reduction_token(algorithm_id: Optional[str]) -> Optional[str]:
    """Normalize reduction tokens for filenames."""
    if not algorithm_id:
        return None
    token = str(algorithm_id).lower()
    if token == "kernel_pca":
        return "kernelpca"
    return token


def format_reduction_name(algorithm_id: Optional[str]) -> str:
    """Format a display name for reduction algorithms."""
    if not algorithm_id:
        return "PCA"
    token = str(algorithm_id).lower()
    if token in {"kernel_pca", "kernelpca"}:
        return "KernelPCA"
    if token == "svd":
        return "SVD"
    return token.upper()


def resolve_reduction_id(
    coords_source: str,
    reduce_algorithm: str,
    metadata: Optional[Mapping[str, Any]],
) -> Optional[str]:
    """Resolve the reduction algorithm used for plotting."""
    if coords_source == "reduce":
        return reduce_algorithm
    if isinstance(metadata, Mapping):
        reduction_id = metadata.get("reduce_algorithm")
        if reduction_id:
            return str(reduction_id)
    return reduce_algorithm


def resolve_provider_id(metadata: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Resolve the embedding provider id from metadata."""
    if isinstance(metadata, Mapping):
        embedding_meta = metadata.get("embedding_metadata")
        if isinstance(embedding_meta, Mapping):
            return embedding_meta.get("provider") or embedding_meta.get("provider_name")
    return None


def normalize_provider_token(provider_id: Optional[str]) -> Optional[str]:
    """Normalize provider tokens for filenames."""
    if not provider_id:
        return None
    token = str(provider_id).lower()
    if "openai" in token:
        return "openai"
    if "gemini" in token or "google" in token:
        return "google"
    return token


def format_embedding_label(metadata: Optional[Mapping[str, Any]]) -> str:
    """Format the embedding provider label for plot titles."""
    provider_name: Optional[str] = None
    if isinstance(metadata, Mapping):
        embedding_meta = metadata.get("embedding_metadata")
        if isinstance(embedding_meta, Mapping):
            provider_name = embedding_meta.get("provider_name") or embedding_meta.get(
                "provider"
            )
    if not provider_name:
        return "Embeddings"
    normalized = str(provider_name)
    lower_name = normalized.lower()
    if "openai" in lower_name:
        normalized = "OpenAI"
    elif "gemini" in lower_name or "google" in lower_name:
        normalized = "Google Gemini"
    return f"{normalized} Embeddings"


def build_output_path(
    output: str,
    algorithm_id: Optional[str],
    reduction_id: Optional[str],
    provider_id: Optional[str],
) -> str:
    """Append algorithm names to the output filename if missing."""
    cluster_token = str(algorithm_id).lower() if algorithm_id else None
    reduction_token = normalize_reduction_token(reduction_id)
    provider_token = normalize_provider_token(provider_id)
    tokens = [tok for tok in (cluster_token, reduction_token, provider_token) if tok]
    if not tokens:
        return output
    output_path = pathlib.Path(output)
    suffix = output_path.suffix
    stem_tokens = output_path.stem.split("_")
    known_tokens = {
        "hdbscan",
        "dbscan",
        "kmeans",
        "pca",
        "kernelpca",
        "svd",
        "openai",
        "google",
    }
    while stem_tokens and stem_tokens[-1].lower() in known_tokens:
        stem_tokens.pop()
    base_stem = "_".join(stem_tokens) if stem_tokens else output_path.stem
    for token in tokens:
        base_stem = f"{base_stem}_{token}"
    return str(output_path.with_name(f"{base_stem}{suffix}"))
