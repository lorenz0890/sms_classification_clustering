"""Cluster computation workflow for SMS embeddings."""
from __future__ import annotations

import csv
import json
import pathlib
import sys
from typing import Dict, Optional

import numpy as np

from ..algorithms.clustering import build_clustering_strategy
from ..algorithms.reduction import reduce_embeddings
from ..config import ClusterConfig


def write_tsv(
    path: pathlib.Path,
    labels: list[str],
    cluster_ids: np.ndarray,
    texts: list[str],
    coords_2d: np.ndarray,
) -> None:
    """Write cluster assignments to a TSV file."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["label", "cluster", "x", "y", "text"])
        for label, cluster_id, text, (x, y) in zip(labels, cluster_ids, texts, coords_2d):
            writer.writerow([label, int(cluster_id), f"{x:.6f}", f"{y:.6f}", text])


class ClusterComputer:
    """Compute clusters from stored embeddings."""

    def __init__(self, config: ClusterConfig) -> None:
        """Initialize the computer with configuration."""
        self._config = config

    def _load_embedding_metadata(self, data: np.lib.npyio.NpzFile) -> Optional[Dict[str, object]]:
        """Load metadata embedded in the embeddings file if available."""
        if "metadata" not in data:
            return None
        try:
            return json.loads(str(data["metadata"].item()))
        except (ValueError, TypeError):
            return None

    def run(self) -> int:
        """Run clustering and save outputs to disk."""
        self._config.validate()
        embeddings_path = pathlib.Path(self._config.input_embeddings)
        if not embeddings_path.exists():
            print(f"File not found: {embeddings_path}", file=sys.stderr)
            return 1

        try:
            data = np.load(embeddings_path, allow_pickle=True)
        except ValueError:
            print(f"Failed to read embeddings file: {embeddings_path}", file=sys.stderr)
            return 1

        if "embeddings" not in data:
            print(f"Missing embeddings in {embeddings_path}", file=sys.stderr)
            return 1

        embeddings = data["embeddings"]
        labels = data["labels"].tolist() if "labels" in data else []
        texts = data["texts"].tolist() if "texts" in data else []
        if not labels or not texts:
            print(f"Missing labels/texts in {embeddings_path}", file=sys.stderr)
            return 1

        reduced = reduce_embeddings(
            embeddings,
            self._config.reduce.algorithm,
            self._config.reduce.dims,
            params=self._config.reduce.params,
            random_state=self._config.seed,
        )
        strategy = build_clustering_strategy(self._config.algorithm, self._config.params)
        cluster_ids = strategy.fit_predict(reduced)

        coords_2d = (
            reduced
            if reduced.shape[1] == 2
            else reduce_embeddings(
                reduced,
                self._config.reduce.algorithm,
                2,
                params=self._config.reduce.params,
                random_state=self._config.seed,
            )
        )

        metadata = {
            "cluster_algorithm": strategy.name,
            "cluster_algorithm_id": self._config.algorithm,
            "cluster_params": self._config.params,
            "seed": self._config.seed,
            "reduce_algorithm": self._config.reduce.algorithm,
            "reduce_params": self._config.reduce.params,
            "reduce_dims": self._config.reduce.dims,
            "embeddings_path": str(embeddings_path),
            "embedding_metadata": self._load_embedding_metadata(data),
            "num_messages": len(texts),
        }

        output_npz_path = pathlib.Path(self._config.output_npz)
        output_tsv_path = pathlib.Path(self._config.output_tsv)
        output_npz_path.parent.mkdir(parents=True, exist_ok=True)
        output_tsv_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            output_npz_path,
            reduced=reduced.astype(np.float32),
            coords_2d=coords_2d.astype(np.float32),
            cluster_ids=cluster_ids.astype(np.int32),
            labels=np.array(labels, dtype=object),
            texts=np.array(texts, dtype=object),
            metadata=np.array(json.dumps(metadata), dtype=object),
        )
        write_tsv(output_tsv_path, labels, cluster_ids, texts, coords_2d)

        num_noise = int(np.sum(cluster_ids == -1))
        unique_clusters = len(set(cluster_ids.tolist()))
        print(f"messages={len(texts)} clusters={unique_clusters} noise={num_noise}")
        print(f"wrote {output_npz_path}")
        print(f"wrote {output_tsv_path}")
        return 0
