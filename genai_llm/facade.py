"""Facade for orchestrating genai_llm workflows."""

from __future__ import annotations

import pathlib

from .cluster.analyze import ClusterNgramAnalyzer
from .cluster.compute import ClusterComputer
from .cluster.visualize import ClusterVisualizer
from .config import GenaiLLMConfig
from .embeddings.compute import EmbeddingComputer
from .paths import GenaiPathResolver


class GenaiLLMFacade:
    """Facade that exposes high-level GenAI/LLM workflows."""

    def __init__(self, config: GenaiLLMConfig) -> None:
        """Initialize the facade with validated configuration."""
        self._config = config
        self._config.validate()
        self._paths = GenaiPathResolver.from_config(
            self._config.cache_dir, self._config.output_dir
        )
        self._paths.ensure_dirs()
        self._resolve_paths()

    def _resolve_paths(self) -> None:
        """Resolve cache/output paths relative to configured directories."""
        provider_token = self._embedding_provider_token()
        original_embeddings_output = self._config.embeddings.output_npz
        self._config.embeddings.output_npz = self._append_token_to_filename(
            original_embeddings_output,
            provider_token,
        )
        if self._config.cluster.input_embeddings == original_embeddings_output:
            self._config.cluster.input_embeddings = self._config.embeddings.output_npz

        self._config.embeddings.output_npz = self._paths.resolve_cache_path(
            self._config.embeddings.output_npz
        )
        self._config.cluster.input_embeddings = self._paths.resolve_cache_path(
            self._config.cluster.input_embeddings
        )
        self._config.cluster.output_npz = self._paths.resolve_cache_path(
            self._config.cluster.output_npz
        )
        self._config.cluster.output_tsv = self._paths.resolve_cache_path(
            self._config.cluster.output_tsv
        )
        self._config.visualize.input = self._paths.resolve_cache_path(
            self._config.visualize.input
        )
        self._config.visualize.output = self._paths.resolve_output_path(
            self._prepend_output_subdir(
                self._config.visualize.output, "clusters_visualized"
            )
        )
        self._config.analyze.input = self._paths.resolve_cache_path(
            self._config.analyze.input
        )
        self._config.analyze.output = self._paths.resolve_output_path(
            self._prepend_output_subdir(
                self._config.analyze.output, "clusters_visualized_annotated"
            )
        )

    def _embedding_provider_token(self) -> str:
        """Normalize embedding provider for filenames."""
        provider = self._config.embeddings.provider.lower()
        if provider in {"gemini", "google"}:
            return "google"
        return provider

    def _append_token_to_filename(self, path: str, token: str) -> str:
        """Append a token to a filename if missing."""
        path_obj = pathlib.Path(path)
        suffix = path_obj.suffix
        stem = path_obj.stem
        if stem.lower().endswith(f"_{token}"):
            return path
        return str(path_obj.with_name(f"{stem}_{token}{suffix}"))

    def _prepend_output_subdir(self, path: str, subdir: str) -> str:
        """Place relative output paths inside a configured subdirectory."""
        path_obj = pathlib.Path(path)
        if path_obj.is_absolute():
            return path
        parts = path_obj.parts
        if parts and parts[0] == subdir:
            return path
        return str(pathlib.Path(subdir) / path_obj)

    def _run_embeddings(self) -> int:
        """Run the embeddings computation step."""
        computer = EmbeddingComputer(self._config.embeddings)
        return computer.run()

    def _run_cluster(self) -> int:
        """Run the clustering computation step."""
        computer = ClusterComputer(self._config.cluster)
        return computer.run()

    def _run_visualize(self) -> int:
        """Run the cluster visualization step."""
        visualizer = ClusterVisualizer(self._config.visualize)
        return visualizer.run()

    def _run_analyze(self) -> int:
        """Run the cluster n-gram analysis step."""
        analyzer = ClusterNgramAnalyzer(self._config.analyze)
        return analyzer.run()

    def run(self) -> int:
        """Run the configured pipeline in order."""
        step_handlers = {
            "embeddings": self._run_embeddings,
            "cluster": self._run_cluster,
            "visualize": self._run_visualize,
            "analyze": self._run_analyze,
        }
        for step in self._config.pipeline:
            handler = step_handlers.get(step)
            if handler is None:
                raise ValueError(f"Unsupported pipeline step: {step}")
            exit_code = handler()
            if exit_code != 0:
                return exit_code
        return 0
