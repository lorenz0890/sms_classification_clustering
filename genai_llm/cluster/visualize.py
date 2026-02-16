"""Cluster visualization workflow."""

from __future__ import annotations

import pathlib
import sys

from ..config import VisualizeConfig
from .formatting import build_output_path
from .metrics import compute_cluster_metrics
from .reporting import write_cluster_report
from .rendering import (
    apply_plot_labels,
    build_plot_context,
    build_plotter,
    import_matplotlib,
    load_dataset_or_warn,
    print_plot_summary,
)


class ClusterVisualizer:
    """Generate scatter plots for clustered SMS embeddings."""

    def __init__(self, config: VisualizeConfig) -> None:
        """Initialize the visualizer with configuration."""
        self._config = config

    def run(self) -> int:
        """Render the cluster visualization and save it to disk."""
        self._config.validate()
        dataset = load_dataset_or_warn(
            self._config.input,
            self._config.coords_source,
            self._config.reduce.algorithm,
            self._config.reduce.params,
        )
        if dataset is None:
            return 1
        if not dataset.labels:
            print(f"Missing labels in {self._config.input}", file=sys.stderr)
            return 1
        plt = import_matplotlib()
        if plt is None:
            return 1

        plt.figure(figsize=(9.72, 7.776))
        plotter, plot_data = build_plotter(plt, dataset)
        plotter.draw_scatter()
        context = build_plot_context(
            self._config.coords_source,
            self._config.reduce.algorithm,
            dataset.metadata,
        )
        apply_plot_labels(plt, context)

        metrics = compute_cluster_metrics(
            dataset.coords, dataset.cluster_ids, dataset.labels
        )
        plotter.add_metrics_box(metrics.text())
        legends = plotter.add_legends()
        output_path = build_output_path(
            self._config.output,
            context.algorithm_id,
            context.reduction_id,
            context.provider_id,
        )
        plotter.save(output_path, legends)
        report_path = write_cluster_report(
            cache_dir=pathlib.Path(self._config.input).parent,
            metrics=metrics,
            metadata=dataset.metadata,
            coords_source=self._config.coords_source,
            plot_reduce={
                "algorithm": self._config.reduce.algorithm,
                "params": self._config.reduce.params,
                "dims": self._config.reduce.dims,
            },
            algorithm_id=context.algorithm_id,
            reduction_id=context.reduction_id,
            provider_id=context.provider_id,
        )

        print_plot_summary(dataset, plot_data, output_path, report_path)
        return 0
