"""Utilities for loading clustering datasets."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ..algorithms.reduction import reduce_embeddings


@dataclass
class ClusterDataset:
    """Loaded data for clustering visualization/analysis."""

    reduced: np.ndarray
    coords: np.ndarray
    cluster_ids: np.ndarray
    labels: List[str]
    texts: List[str]
    metadata: Optional[Dict[str, Any]]


def load_cluster_dataset(
    input_path: str,
    coords_source: str,
    reduce_algorithm: str,
    reduce_params: Dict[str, Any],
) -> ClusterDataset:
    """Load cluster data from disk and compute coordinates as needed."""
    path = pathlib.Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        reduced: np.ndarray = np.asarray(data["reduced"])
        coords_2d = np.asarray(data["coords_2d"]) if "coords_2d" in data else None
        cluster_ids: np.ndarray = np.asarray(data["cluster_ids"])
        labels = np.asarray(data["labels"]).tolist() if "labels" in data else []
        texts = np.asarray(data["texts"]).tolist() if "texts" in data else []

        metadata: Optional[Dict[str, Any]] = None
        if "metadata" in data:
            try:
                metadata_value = np.asarray(data["metadata"]).item()
                metadata = json.loads(str(metadata_value))
            except (ValueError, TypeError):
                metadata = None

    coords = coords_2d
    if coords_source == "reduce" or coords is None:
        seed_value = metadata.get("seed") if isinstance(metadata, dict) else None
        random_state = seed_value if isinstance(seed_value, int) else None
        coords = (
            reduced
            if reduced.shape[1] == 2
            else reduce_embeddings(
                reduced,
                reduce_algorithm,
                2,
                params=reduce_params,
                random_state=random_state,
            )
        )

    return ClusterDataset(
        reduced=reduced,
        coords=coords,
        cluster_ids=cluster_ids,
        labels=labels,
        texts=texts,
        metadata=metadata,
    )
