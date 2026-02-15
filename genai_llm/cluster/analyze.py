"""Cluster n-gram analysis and annotated visualization."""
from __future__ import annotations

import collections
import pathlib
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import nltk
import numpy as np

from utils import ensure_nltk_data, load_stopwords, tokenize

from ..config import AnalyzeConfig
from .data_loader import ClusterDataset, load_cluster_dataset
from .metrics import compute_cluster_metrics
from .plotting import ClusterPlotData, ClusterPlotter


NgramCounts = Dict[int, List[Tuple[Tuple[str, ...], int]]]


class ClusterNgramAnalyzer:
    """Analyze top n-grams for clusters and annotate a scatter plot."""

    def __init__(self, config: AnalyzeConfig) -> None:
        """Initialize the analyzer with configuration."""
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

    def _collect_cluster_ngrams(
        self,
        texts: Sequence[str],
        cluster_ids: np.ndarray,
        n: int,
        top_k: int,
        stopwords_set: Optional[set[str]],
    ) -> NgramCounts:
        """Collect top-k n-grams for each non-noise cluster."""
        text_array = np.array(texts, dtype=object)
        ngram_by_cluster: NgramCounts = {}
        for cluster_id in sorted(set(cluster_ids.tolist())):
            if cluster_id == -1:
                continue
            cluster_mask = cluster_ids == cluster_id
            cluster_texts = text_array[cluster_mask]
            counter: collections.Counter[Tuple[str, ...]] = collections.Counter()
            for text in cluster_texts:
                tokens = tokenize(text, stopwords_set=stopwords_set)
                for gram in nltk.ngrams(tokens, n):
                    counter[gram] += 1
            ngram_by_cluster[int(cluster_id)] = counter.most_common(top_k)
        return ngram_by_cluster

    def _format_gram(self, gram: Sequence[str]) -> str:
        """Format an n-gram tuple for display."""
        return " ".join(gram)

    def _annotate_ngrams(
        self, plt_module: object, coords: np.ndarray, cluster_ids: np.ndarray, ngrams: NgramCounts
    ) -> None:
        """Annotate cluster centers with top n-grams."""
        y_span = float(np.ptp(coords[:, 1])) if coords.size else 1.0
        line_offset = y_span * 0.02 if y_span > 0 else 0.1
        all_counts = [count for grams in ngrams.values() for _, count in grams if count > 0]
        min_count = min(all_counts) if all_counts else 1
        max_count = max(all_counts) if all_counts else 1
        min_font = 8.0
        max_font = 14.0

        def scale_font(count: int) -> float:
            """Scale font size based on count."""
            if max_count == min_count:
                return (min_font + max_font) / 2
            return min_font + (count - min_count) * (max_font - min_font) / (max_count - min_count)

        for cluster_id, grams in ngrams.items():
            cluster_mask = cluster_ids == cluster_id
            if not np.any(cluster_mask) or not grams:
                continue
            center = coords[cluster_mask].mean(axis=0)
            mid_index = (len(grams) - 1) / 2.0
            for idx, (gram, count) in enumerate(grams):
                y = center[1] + (mid_index - idx) * line_offset
                plt_module.text(
                    center[0],
                    y,
                    self._format_gram(gram),
                    ha="center",
                    va="center",
                    fontsize=scale_font(int(count)),
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        alpha=0.65,
                        edgecolor="none",
                    ),
                )

    def run(self) -> int:
        """Run the cluster n-gram analysis and save the plot."""
        self._config.validate()
        if self._config.top_k < 1:
            print("top_k must be at least 1.", file=sys.stderr)
            return 1
        if self._config.n < 1:
            print("n must be at least 1.", file=sys.stderr)
            return 1

        dataset = self._load_dataset()
        if dataset is None:
            return 1
        if not dataset.labels or not dataset.texts:
            print(f"Missing labels/texts in {self._config.input}", file=sys.stderr)
            return 1

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Missing matplotlib. Install it in your venv to generate plots.", file=sys.stderr)
            return 1

        remove_stopwords = self._config.remove_stopwords
        ensure_nltk_data(include_stopwords=remove_stopwords)
        stopwords_set = load_stopwords() if remove_stopwords else None

        ngrams_by_cluster = self._collect_cluster_ngrams(
            dataset.texts,
            dataset.cluster_ids,
            self._config.n,
            self._config.top_k,
            stopwords_set,
        )

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
        self._annotate_ngrams(plt, dataset.coords, dataset.cluster_ids, ngrams_by_cluster)

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
