"""Cluster visualization workflow."""
from __future__ import annotations

import sys
from typing import Optional

from ..config import VisualizeConfig
from .data_loader import ClusterDataset, load_cluster_dataset
from .formatting import (
    build_output_path,
    format_embedding_label,
    format_reduction_name,
    resolve_provider_id,
    resolve_reduction_id,
)
from .metrics import compute_cluster_metrics
from .plotting import ClusterPlotData, ClusterPlotter


class ClusterVisualizer:
    """Generate scatter plots for clustered SMS embeddings."""

    def __init__(self, config: VisualizeConfig) -> None:
        """Initialize the visualizer with configuration."""
        self._config = config

    def _load_dataset(self) -> Optional[ClusterDataset]:
        """Load the clustering dataset from disk."""
        try:
            return load_cluster_dataset(
                self._config.input,
                self._config.coords_source,
                self._config.reduce.algorithm,
                self._config.reduce.params,
            )
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return None

    def run(self) -> int:
        """Render the cluster visualization and save it to disk."""
        self._config.validate()
        dataset = self._load_dataset()
        if dataset is None:
            return 1
        if not dataset.labels:
            print(f"Missing labels in {self._config.input}", file=sys.stderr)
            return 1

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Missing matplotlib. Install it in your venv to generate plots.", file=sys.stderr)
            return 1

        plt.figure(figsize=(9.72, 7.776))
        plot_data = ClusterPlotData(
            coords=dataset.coords, cluster_ids=dataset.cluster_ids, labels=dataset.labels
        )
        plotter = ClusterPlotter(plt, plot_data)
        plotter.draw_scatter()

        algorithm = "HDBSCAN"
        algorithm_id: Optional[str] = None
        if isinstance(dataset.metadata, dict):
            algorithm = dataset.metadata.get("cluster_algorithm") or algorithm
            algorithm_id = (
                dataset.metadata.get("cluster_algorithm_id")
                or dataset.metadata.get("cluster_algorithm")
            )
        reduction_id = resolve_reduction_id(
            self._config.coords_source, self._config.reduce.algorithm, dataset.metadata
        )
        provider_id = resolve_provider_id(dataset.metadata)
        reduction_name = format_reduction_name(reduction_id)
        embedding_label = format_embedding_label(dataset.metadata)
        plt.title(
            f"SMS Clusters ({algorithm} + {reduction_name} on {embedding_label})"
        )
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")

        metrics = compute_cluster_metrics(dataset.coords, dataset.cluster_ids, dataset.labels)
        plotter.add_metrics_box(metrics.text())
        legends = plotter.add_legends()
        output_path = build_output_path(
            self._config.output, algorithm_id, reduction_id, provider_id
        )
        plotter.save(output_path, legends)

        messages = (
            dataset.metadata.get("num_messages")
            if isinstance(dataset.metadata, dict)
            else len(dataset.labels)
        )
        num_noise = int((dataset.cluster_ids == -1).sum())
        print(f"messages={messages} clusters={len(plot_data.unique_clusters())} noise={num_noise}")
        print(f"wrote {output_path}")
        return 0
