#!/usr/bin/env python3
"""Aggregate clustering metrics across genai_llm configs."""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from typing import Any, Dict, List, Sequence


def _load_reports(cache_dir: pathlib.Path) -> List[Dict[str, Any]]:
    """Load all metrics reports from the cache directory."""
    reports: List[Dict[str, Any]] = []
    for path in sorted(cache_dir.glob("cluster_metrics*.json")):
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        report["_path"] = str(path)
        reports.append(report)
    return reports


def _normalize_reduction_token(token: Any) -> str:
    """Normalize reduction tokens for display."""
    if not token:
        return "reduce"
    value = str(token).lower()
    if value == "kernel_pca":
        return "kernelpca"
    return value


def _normalize_provider_token(token: Any) -> str:
    """Normalize provider tokens for display."""
    if not token:
        return "provider"
    value = str(token).lower()
    if "openai" in value:
        return "openai"
    if "gemini" in value or "google" in value:
        return "gemini"
    return value


def _canonical_provider(token: Any) -> str:
    """Normalize provider for filtering."""
    return _normalize_provider_token(token)


def _build_config_id(report: Dict[str, Any]) -> str:
    """Build a short identifier for a config based on stored config data."""
    config = report.get("config", {}) if isinstance(report, dict) else {}
    cluster_cfg = config.get("cluster", {}) if isinstance(config, dict) else {}
    plot_cfg = config.get("plot", {}) if isinstance(config, dict) else {}
    embeddings_cfg = config.get("embeddings", {}) if isinstance(config, dict) else {}

    cluster_algo = (
        cluster_cfg.get("algorithm_id") or cluster_cfg.get("algorithm") or "cluster"
    )
    coords_source = plot_cfg.get("coords_source")
    plot_reduce = plot_cfg.get("reduce", {}) if isinstance(plot_cfg, dict) else {}
    cluster_reduce = (
        cluster_cfg.get("reduce", {}) if isinstance(cluster_cfg, dict) else {}
    )

    if coords_source == "reduce":
        reduction_token = _normalize_reduction_token(plot_reduce.get("algorithm"))
    else:
        reduction_token = _normalize_reduction_token(cluster_reduce.get("algorithm"))

    provider_token = _normalize_provider_token(
        embeddings_cfg.get("provider") or embeddings_cfg.get("provider_name")
    )
    return f"{str(cluster_algo).lower()}_{reduction_token}_{provider_token}"


def _extract_primary_cluster(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the primary cluster metrics from a report."""
    top_clusters = metrics.get("top_clusters", []) if isinstance(metrics, dict) else []
    if not top_clusters:
        return {}
    return top_clusters[0] if isinstance(top_clusters[0], dict) else {}


def _build_rows(
    reports: Sequence[Dict[str, Any]], provider_filter: str
) -> List[Dict[str, Any]]:
    """Build flat rows for CSV and plotting."""
    rows: List[Dict[str, Any]] = []
    for report in reports:
        metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
        primary = _extract_primary_cluster(metrics)
        if not primary:
            continue
        config = report.get("config", {}) if isinstance(report, dict) else {}
        cluster_cfg = config.get("cluster", {}) if isinstance(config, dict) else {}
        plot_cfg = config.get("plot", {}) if isinstance(config, dict) else {}
        embeddings_cfg = (
            config.get("embeddings", {}) if isinstance(config, dict) else {}
        )
        plot_reduce = plot_cfg.get("reduce", {}) if isinstance(plot_cfg, dict) else {}
        cluster_reduce = (
            cluster_cfg.get("reduce", {}) if isinstance(cluster_cfg, dict) else {}
        )

        provider_value = embeddings_cfg.get("provider") or embeddings_cfg.get(
            "provider_name"
        )
        if provider_filter != "all":
            if _canonical_provider(provider_value) != provider_filter:
                continue

        row = {
            "config_id": _build_config_id(report),
            "cluster_algorithm": cluster_cfg.get("algorithm_id")
            or cluster_cfg.get("algorithm"),
            "cluster_reduce": cluster_reduce.get("algorithm"),
            "plot_reduce": plot_reduce.get("algorithm"),
            "embedding_provider": provider_value,
            "embedding_model": embeddings_cfg.get("model"),
            "coords_source": plot_cfg.get("coords_source"),
            "seed": cluster_cfg.get("seed"),
            "silhouette": metrics.get("silhouette"),
            "cluster_id": primary.get("cluster_id"),
            "majority_label": primary.get("majority_label"),
            "accuracy": primary.get("accuracy"),
            "precision": primary.get("precision"),
            "recall": primary.get("recall"),
            "f1": primary.get("f1"),
            "coverage": primary.get("coverage"),
            "label_in_cluster": primary.get("label_in_cluster"),
            "label_total": primary.get("label_total"),
            "cluster_size": primary.get("cluster_size"),
            "report_path": report.get("_path"),
        }
        rows.append(row)
    return rows


def _write_csv(rows: Sequence[Dict[str, Any]], output_path: pathlib.Path) -> None:
    """Write aggregated results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config_id",
        "cluster_algorithm",
        "cluster_reduce",
        "plot_reduce",
        "embedding_provider",
        "embedding_model",
        "coords_source",
        "seed",
        "silhouette",
        "cluster_id",
        "majority_label",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "coverage",
        "label_in_cluster",
        "label_total",
        "cluster_size",
        "report_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_metric(
    rows: Sequence[Dict[str, Any]],
    metric: str,
    output_path: pathlib.Path,
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> bool:
    """Plot a bar chart for a metric."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    names = [row["config_id"] for row in rows]
    values = [row.get(metric) or 0.0 for row in rows]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(names, values, color="#4C72B0")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"GenAI Clustering {metric.upper()} by Configuration")
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> int:
    """Entry point for report aggregation."""
    parser = argparse.ArgumentParser(description="Aggregate GenAI clustering metrics.")
    parser.add_argument(
        "--cache-dir",
        default="genai_llm/cache",
        help="Directory containing cluster_metrics*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to write CSV and charts.",
    )
    parser.add_argument(
        "--provider",
        choices=("all", "openai", "gemini"),
        default="all",
        help="Filter reports by embedding provider.",
    )
    args = parser.parse_args()

    cache_dir = pathlib.Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}", file=sys.stderr)
        return 0

    reports = _load_reports(cache_dir)
    if not reports:
        print("No cluster metrics reports found.", file=sys.stderr)
        return 0

    rows = _build_rows(reports, args.provider)
    if not rows:
        print("No cluster metrics with top clusters found.", file=sys.stderr)
        return 0

    output_dir = pathlib.Path(args.output_dir)
    csv_path = output_dir / "genai_cluster_comparison.csv"
    _write_csv(rows, csv_path)
    print(f"wrote {csv_path}")

    accuracy_path = output_dir / "genai_cluster_accuracy.png"
    f1_path = output_dir / "genai_cluster_f1.png"
    silhouette_path = output_dir / "genai_cluster_silhouette.png"
    coverage_path = output_dir / "genai_cluster_coverage.png"

    if not _plot_metric(rows, "accuracy", accuracy_path):
        print(
            "Missing matplotlib. Install it in your venv to generate charts.",
            file=sys.stderr,
        )
        return 0
    _plot_metric(rows, "f1", f1_path)
    _plot_metric(rows, "silhouette", silhouette_path, y_min=-1.0, y_max=1.0)
    _plot_metric(rows, "coverage", coverage_path)
    print(f"wrote {accuracy_path}")
    print(f"wrote {f1_path}")
    print(f"wrote {silhouette_path}")
    print(f"wrote {coverage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
