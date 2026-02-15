"""Cluster evaluation metrics helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)


@dataclass
class ClusterMetrics:
    """Container for cluster metric output strings."""

    silhouette_text: str
    top_cluster_lines: List[str]
    silhouette: Optional[float] = None
    top_clusters: List["ClusterSummary"] = field(default_factory=list)

    def text(self) -> str:
        """Combine all metric lines into a single block of text."""
        return "\n".join([self.silhouette_text] + self.top_cluster_lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics into a JSON-serializable dictionary."""
        return {
            "silhouette": self.silhouette,
            "top_clusters": [cluster.as_dict() for cluster in self.top_clusters],
        }


@dataclass
class ClusterSummary:
    """Structured metrics for a single cluster."""

    cluster_id: int
    majority_label: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    label_in_cluster: int
    label_total: int
    cluster_size: int

    def as_dict(self) -> Dict[str, Any]:
        """Convert the summary to a JSON-serializable dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "majority_label": self.majority_label,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "label_in_cluster": self.label_in_cluster,
            "label_total": self.label_total,
            "cluster_size": self.cluster_size,
        }


def compute_cluster_metrics(
    coords: np.ndarray, cluster_ids: np.ndarray, labels: Sequence[str]
) -> ClusterMetrics:
    """Compute silhouette score and top-cluster label alignment metrics."""
    silhouette_text = "Silhouette: n/a"
    silhouette_value: Optional[float] = None
    non_noise_mask = cluster_ids != -1
    if np.any(non_noise_mask):
        non_noise_labels = cluster_ids[non_noise_mask]
        if len(set(non_noise_labels.tolist())) > 1 and np.sum(non_noise_mask) > 1:
            try:
                score = silhouette_score(coords[non_noise_mask], non_noise_labels)
                silhouette_text = f"Silhouette: {score:.3f}"
                silhouette_value = float(score)
            except ValueError:
                pass

    cluster_metrics_lines: List[str] = ["Top clusters: N/A"]
    cluster_summaries: List[ClusterSummary] = []
    non_noise_ids = cluster_ids[non_noise_mask]
    if non_noise_ids.size:
        cluster_ids_unique, cluster_counts = np.unique(
            non_noise_ids, return_counts=True
        )
        if len(cluster_ids_unique) >= 2:
            top_indices = np.argsort(cluster_counts)[::-1][:2]
            top_clusters = cluster_ids_unique[top_indices]
            label_array = np.array(labels)
            cluster_metrics_lines = ["Top clusters:"]
            for cluster_id in top_clusters:
                cluster_mask = cluster_ids == cluster_id
                cluster_labels = label_array[cluster_mask]
                if cluster_labels.size == 0:
                    continue
                values, value_counts = np.unique(cluster_labels, return_counts=True)
                majority_label = values[np.argmax(value_counts)]
                label_mask = label_array == majority_label
                label_total = int(np.sum(label_mask))
                label_in_cluster = int(np.sum(cluster_mask & label_mask))
                y_true = label_mask
                y_pred = cluster_mask
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                cluster_size = int(np.sum(cluster_mask))
                cluster_summaries.append(
                    ClusterSummary(
                        cluster_id=int(cluster_id),
                        majority_label=str(majority_label),
                        accuracy=float(acc),
                        precision=float(prec),
                        recall=float(rec),
                        f1=float(f1),
                        label_in_cluster=label_in_cluster,
                        label_total=label_total,
                        cluster_size=cluster_size,
                    )
                )
                cluster_metrics_lines.append(f"cluster {cluster_id} ({majority_label})")
                cluster_metrics_lines.append(
                    f"  acc={acc:.3f} prec={prec:.3f} rec={rec:.3f}"
                )
                cluster_metrics_lines.append(
                    f"  f1={f1:.3f} label_in_cluster={label_in_cluster} "
                    f"label_total={label_total}"
                )
    return ClusterMetrics(
        silhouette_text=silhouette_text,
        top_cluster_lines=cluster_metrics_lines,
        silhouette=silhouette_value,
        top_clusters=cluster_summaries,
    )
