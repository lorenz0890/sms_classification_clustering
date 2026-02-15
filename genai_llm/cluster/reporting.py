"""Metrics reporting helpers for cluster runs."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Mapping, Optional

from .formatting import build_output_path
from .metrics import ClusterMetrics


def _normalize_meta_dict(meta: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Coerce optional metadata mapping to a dictionary."""
    if not meta:
        return {}
    return dict(meta)


def write_cluster_report(
    cache_dir: pathlib.Path,
    metrics: ClusterMetrics,
    metadata: Optional[Mapping[str, Any]],
    coords_source: str,
    plot_reduce: Mapping[str, Any],
    algorithm_id: Optional[str],
    reduction_id: Optional[str],
    provider_id: Optional[str],
) -> str:
    """Write a JSON metrics report to the cache directory."""
    meta = _normalize_meta_dict(metadata)
    embedding_meta = meta.get("embedding_metadata")
    embedding_meta = embedding_meta if isinstance(embedding_meta, Mapping) else {}

    cluster_config = {
        "algorithm": meta.get("cluster_algorithm"),
        "algorithm_id": meta.get("cluster_algorithm_id"),
        "params": meta.get("cluster_params"),
        "seed": meta.get("seed"),
        "reduce": {
            "algorithm": meta.get("reduce_algorithm"),
            "params": meta.get("reduce_params"),
            "dims": meta.get("reduce_dims"),
        },
    }
    plot_config = {
        "coords_source": coords_source,
        "reduce": dict(plot_reduce),
    }
    embeddings_config = {
        "provider": embedding_meta.get("provider"),
        "provider_name": embedding_meta.get("provider_name"),
        "model": embedding_meta.get("model"),
        "provider_params": embedding_meta.get("provider_params"),
        "batch_size": embedding_meta.get("batch_size"),
        "limit": embedding_meta.get("limit"),
        "source_path": embedding_meta.get("source_path"),
        "num_messages": embedding_meta.get("num_messages"),
    }

    report = {
        "config": {
            "cluster": cluster_config,
            "plot": plot_config,
            "embeddings": embeddings_config,
        },
        "metrics": metrics.to_dict(),
        "metrics_text": metrics.text(),
    }

    metrics_base = cache_dir / "cluster_metrics.json"
    output_path = build_output_path(
        str(metrics_base), algorithm_id, reduction_id, provider_id
    )
    output_path_obj = pathlib.Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    output_path_obj.write_text(json.dumps(report, indent=2) + "\n")
    return str(output_path_obj)
