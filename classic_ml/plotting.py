"""Plotting helpers for classic_ml visualizations."""
from __future__ import annotations

import pathlib
from typing import Mapping, Sequence, Tuple

import numpy as np


def plot_classifier_performance(
    output: str,
    results: Sequence[object],
) -> bool:
    """Plot classifier performance metrics and save to disk."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    if not results:
        return False

    metrics = ("accuracy", "precision", "recall", "f1")
    names = [getattr(result, "classifier_name", "classifier") for result in results]
    values = {
        metric: [float(getattr(result, metric, 0.0)) for result in results]
        for metric in metrics
    }

    x = np.arange(len(names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 6))
    offsets = np.linspace(-(len(metrics) - 1) / 2, (len(metrics) - 1) / 2, len(metrics))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for idx, metric in enumerate(metrics):
        ax.bar(
            x + offsets[idx] * width,
            values[metric],
            width,
            label=metric.capitalize(),
            color=colors[idx % len(colors)],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Classifier Performance")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_label_bars(
    output: str,
    label_items: Mapping[str, Sequence[Tuple[str, int]]],
    title_suffix: str,
) -> bool:
    """Plot horizontal bar charts for each label and save to disk."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    labels = list(label_items.keys())
    max_items = max((len(items) for items in label_items.values()), default=0)
    per_row_height = max(3.0, max_items * 0.25)
    fig_height = per_row_height * max(len(labels), 1)

    fig, axes = plt.subplots(len(labels), 1, figsize=(12, fig_height), squeeze=False)
    for idx, label in enumerate(labels):
        ax = axes[idx][0]
        items = label_items[label]
        if not items:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue
        names = [name for name, _ in items]
        counts = [count for _, count in items]
        y_pos = list(range(len(names)))
        ax.barh(y_pos, counts, color="#4C72B0")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        title = f"{label.upper()} {title_suffix}".strip()
        ax.set_title(title)

    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True
