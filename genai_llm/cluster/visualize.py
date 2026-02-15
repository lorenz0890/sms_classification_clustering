"""Cluster visualization workflow."""
from __future__ import annotations

import pathlib
import sys
from typing import Optional

from ..config import VisualizeConfig
from .data_loader import ClusterDataset, load_cluster_dataset
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

    def _normalize_reduction_token(self, algorithm_id: Optional[str]) -> Optional[str]:
        """Normalize reduction tokens for filenames."""
        if not algorithm_id:
            return None
        token = str(algorithm_id).lower()
        if token == "kernel_pca":
            return "kernelpca"
        return token

    def _format_reduction_name(self, algorithm_id: Optional[str]) -> str:
        """Format a display name for reduction algorithms."""
        if not algorithm_id:
            return "PCA"
        token = str(algorithm_id).lower()
        if token in {"kernel_pca", "kernelpca"}:
            return "KernelPCA"
        if token == "svd":
            return "SVD"
        return token.upper()

    def _resolve_reduction_id(self, dataset: ClusterDataset) -> Optional[str]:
        """Resolve the reduction algorithm used for plotting."""
        if self._config.coords_source == "reduce":
            return self._config.reduce.algorithm
        if isinstance(dataset.metadata, dict):
            reduction_id = dataset.metadata.get("reduce_algorithm")
            if reduction_id:
                return reduction_id
        return self._config.reduce.algorithm

    def _resolve_provider_id(self, dataset: ClusterDataset) -> Optional[str]:
        """Resolve the embedding provider id from metadata."""
        if isinstance(dataset.metadata, dict):
            embedding_meta = dataset.metadata.get("embedding_metadata")
            if isinstance(embedding_meta, dict):
                return embedding_meta.get("provider") or embedding_meta.get(
                    "provider_name"
                )
        return None

    def _normalize_provider_token(self, provider_id: Optional[str]) -> Optional[str]:
        """Normalize provider tokens for filenames."""
        if not provider_id:
            return None
        token = str(provider_id).lower()
        if "openai" in token:
            return "openai"
        if "gemini" in token or "google" in token:
            return "google"
        return token

    def _format_embedding_label(self, dataset: ClusterDataset) -> str:
        """Format the embedding provider label for plot titles."""
        provider_name: Optional[str] = None
        if isinstance(dataset.metadata, dict):
            embedding_meta = dataset.metadata.get("embedding_metadata")
            if isinstance(embedding_meta, dict):
                provider_name = (
                    embedding_meta.get("provider_name")
                    or embedding_meta.get("provider")
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

    def _build_output_path(
        self,
        algorithm_id: Optional[str],
        reduction_id: Optional[str],
        provider_id: Optional[str],
    ) -> str:
        """Append algorithm names to the output filename if missing."""
        cluster_token = str(algorithm_id).lower() if algorithm_id else None
        reduction_token = self._normalize_reduction_token(reduction_id)
        provider_token = self._normalize_provider_token(provider_id)
        tokens = [tok for tok in (cluster_token, reduction_token, provider_token) if tok]
        if not tokens:
            return self._config.output
        output_path = pathlib.Path(self._config.output)
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
        reduction_id = self._resolve_reduction_id(dataset)
        provider_id = self._resolve_provider_id(dataset)
        reduction_name = self._format_reduction_name(reduction_id)
        embedding_label = self._format_embedding_label(dataset)
        plt.title(
            f"SMS Clusters ({algorithm} + {reduction_name} on {embedding_label})"
        )
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")

        metrics = compute_cluster_metrics(dataset.coords, dataset.cluster_ids, dataset.labels)
        plotter.add_metrics_box(metrics.text())
        legends = plotter.add_legends()
        output_path = self._build_output_path(algorithm_id, reduction_id, provider_id)
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
