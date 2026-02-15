"""Plotting helpers for cluster visualizations."""

from __future__ import annotations

from dataclasses import dataclass
import pathlib
from typing import Any, List, Sequence, Tuple

import numpy as np


@dataclass
class ClusterPlotData:
    """Container for plotting cluster scatter data."""

    coords: np.ndarray
    cluster_ids: np.ndarray
    labels: Sequence[str]

    def unique_clusters(self) -> List[int]:
        """Return sorted unique cluster identifiers."""
        return sorted(set(int(cluster_id) for cluster_id in self.cluster_ids))


class ClusterPlotter:
    """Plot scatter points, legends, and metric annotations for clusters."""

    def __init__(self, plt_module: Any, data: ClusterPlotData) -> None:
        """Initialize the plotter with matplotlib and plot data."""
        self._plt = plt_module
        self._data = data

    def draw_scatter(self) -> None:
        """Draw scatter points for clusters with label markers."""
        unique_clusters = self._data.unique_clusters()
        colors = self._plt.cm.tab20(np.linspace(0, 1, max(len(unique_clusters), 1)))

        marker_map = {"ham": "o", "spam": "x"}
        label_array = np.array(self._data.labels)
        for idx, cluster_id in enumerate(unique_clusters):
            cluster_mask = self._data.cluster_ids == cluster_id
            color = "#999999" if cluster_id == -1 else colors[idx % len(colors)]
            cluster_label = "noise" if cluster_id == -1 else f"cluster {cluster_id}"

            for label_value, marker in marker_map.items():
                label_mask = np.array([lbl == label_value for lbl in label_array])
                mask = cluster_mask & label_mask
                if not np.any(mask):
                    continue
                self._plt.scatter(
                    self._data.coords[mask, 0],
                    self._data.coords[mask, 1],
                    s=14,
                    c=[color],
                    marker=marker,
                    label=cluster_label,
                    alpha=0.8,
                    linewidths=0.6,
                )

    def add_metrics_box(self, metrics_text: str) -> None:
        """Add a metrics annotation box to the current axes."""
        self._plt.text(
            0.02,
            0.98,
            metrics_text,
            transform=self._plt.gca().transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                alpha=0.8,
                edgecolor="none",
            ),
        )

    def add_legends(self) -> Tuple[Any, Any]:
        """Add cluster and label legends to the plot."""
        cluster_handles, cluster_labels = self._plt.gca().get_legend_handles_labels()
        cluster_legend = self._plt.legend(
            cluster_handles,
            cluster_labels,
            markerscale=2,
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
            title="Clusters",
        )
        self._plt.gca().add_artist(cluster_legend)

        shape_handles = [
            self._plt.Line2D(
                [0], [0], marker="o", color="k", linestyle="None", label="ham"
            ),
            self._plt.Line2D(
                [0], [0], marker="x", color="k", linestyle="None", label="spam"
            ),
        ]
        label_legend = self._plt.legend(
            handles=shape_handles,
            bbox_to_anchor=(1.04, 0),
            loc="lower left",
            title="Labels",
        )
        return cluster_legend, label_legend

    def save(self, output: str, legends: Sequence[Any]) -> None:
        """Save the figure with legends included in the bounding box."""
        output_path = pathlib.Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._plt.tight_layout()
        self._plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
            bbox_extra_artists=list(legends),
        )
