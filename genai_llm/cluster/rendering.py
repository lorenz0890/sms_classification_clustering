"""Shared helpers for cluster plotting workflows."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any, Dict, Optional, Tuple

from .data_loader import ClusterDataset, load_cluster_dataset
from .formatting import (
    format_embedding_label,
    format_reduction_name,
    resolve_provider_id,
    resolve_reduction_id,
)
from .plotting import ClusterPlotData, ClusterPlotter


@dataclass
class PlotContext:
    """Resolved metadata for cluster plots."""

    algorithm: str
    algorithm_id: Optional[str]
    reduction_id: Optional[str]
    provider_id: Optional[str]
    reduction_name: str
    embedding_label: str


def load_dataset_or_warn(
    input_path: str,
    coords_source: str,
    reduce_algorithm: str,
    reduce_params: Dict[str, Any],
) -> Optional[ClusterDataset]:
    """Load the clustering dataset from disk and warn on failure."""
    try:
        return load_cluster_dataset(
            input_path,
            coords_source,
            reduce_algorithm,
            reduce_params,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return None


def import_matplotlib() -> Optional[object]:
    """Import matplotlib and return the pyplot module."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Missing matplotlib. Install it in your venv to generate plots.",
            file=sys.stderr,
        )
        return None
    return plt


def build_plot_context(
    coords_source: str,
    reduce_algorithm: str,
    metadata: Optional[Dict[str, Any]],
) -> PlotContext:
    """Resolve metadata needed for plot labels and filenames."""
    algorithm = "HDBSCAN"
    algorithm_id: Optional[str] = None
    if isinstance(metadata, dict):
        algorithm = metadata.get("cluster_algorithm") or algorithm
        algorithm_id = metadata.get("cluster_algorithm_id") or metadata.get(
            "cluster_algorithm"
        )
    reduction_id = resolve_reduction_id(coords_source, reduce_algorithm, metadata)
    provider_id = resolve_provider_id(metadata)
    reduction_name = format_reduction_name(reduction_id)
    embedding_label = format_embedding_label(metadata)
    return PlotContext(
        algorithm=algorithm,
        algorithm_id=algorithm_id,
        reduction_id=reduction_id,
        provider_id=provider_id,
        reduction_name=reduction_name,
        embedding_label=embedding_label,
    )


def apply_plot_labels(plt_module: object, context: PlotContext) -> None:
    """Apply title/axis labels to a matplotlib plot."""
    plt_module.title(
        f"SMS Clusters ({context.algorithm} + {context.reduction_name} on "
        f"{context.embedding_label})"
    )
    plt_module.xlabel("Component 1")
    plt_module.ylabel("Component 2")


def build_plotter(
    plt_module: object, dataset: ClusterDataset
) -> Tuple[ClusterPlotter, ClusterPlotData]:
    """Create plot data and plotter for the dataset."""
    plot_data = ClusterPlotData(
        coords=dataset.coords,
        cluster_ids=dataset.cluster_ids,
        labels=dataset.labels,
    )
    plotter = ClusterPlotter(plt_module, plot_data)
    return plotter, plot_data


def print_plot_summary(
    dataset: ClusterDataset,
    plot_data: ClusterPlotData,
    output_path: str,
    report_path: str,
) -> None:
    """Print summary lines after writing plots and reports."""
    messages = (
        dataset.metadata.get("num_messages")
        if isinstance(dataset.metadata, dict)
        else len(dataset.labels)
    )
    clusters_count = len(plot_data.unique_clusters())
    num_noise = int((dataset.cluster_ids == -1).sum())
    print(f"messages={messages} clusters={clusters_count} noise={num_noise}")
    print(f"wrote {output_path}")
    print(f"wrote {report_path}")
