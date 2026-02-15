"""Facade for orchestrating classic_ml workflows."""
from __future__ import annotations

import pathlib

from .classifier import SmsClassifierRunner
from .config import ClassicMLConfig
from .ngrams import NgramFrequencyAnalyzer
from .words import WordFrequencyAnalyzer


class ClassicMLFacade:
    """Facade that exposes high-level classic ML workflows."""

    def __init__(self, config: ClassicMLConfig) -> None:
        """Initialize the facade with validated configuration."""
        self._config = config
        self._config.validate()
        self._resolve_paths()

    def _resolve_output_path(self, path: str) -> str:
        """Resolve output paths relative to the configured output directory."""
        path_obj = pathlib.Path(path)
        if path_obj.is_absolute():
            return str(path_obj)
        return str(pathlib.Path(self._config.output_dir) / path_obj)

    def _resolve_cache_path(self, path: str) -> str:
        """Resolve cache paths relative to the configured cache directory."""
        path_obj = pathlib.Path(path)
        if path_obj.is_absolute():
            return str(path_obj)
        return str(pathlib.Path(self._config.cache_dir) / path_obj)

    def _resolve_paths(self) -> None:
        """Ensure output directory exists and resolve output paths."""
        output_dir = pathlib.Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = pathlib.Path(self._config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._config.words.output = self._resolve_output_path(self._config.words.output)
        self._config.ngrams.output = self._resolve_output_path(self._config.ngrams.output)
        self._config.classifier.output = self._resolve_output_path(
            self._config.classifier.output
        )
        self._config.classifier.results_output = self._resolve_cache_path(
            self._config.classifier.results_output
        )

    def _run_words(self) -> int:
        """Run the word analysis step."""
        analyzer = WordFrequencyAnalyzer(self._config.words)
        return analyzer.run()

    def _run_ngrams(self) -> int:
        """Run the n-gram analysis step."""
        analyzer = NgramFrequencyAnalyzer(self._config.ngrams)
        return analyzer.run()

    def _run_classifier(self) -> int:
        """Run the classifier training/evaluation step."""
        classifier = SmsClassifierRunner(self._config.classifier)
        return classifier.run()

    def run(self) -> int:
        """Run the configured pipeline in order."""
        for step in self._config.pipeline:
            if step == "words":
                exit_code = self._run_words()
            elif step == "ngrams":
                exit_code = self._run_ngrams()
            elif step == "classifier":
                exit_code = self._run_classifier()
            else:
                raise ValueError(f"Unsupported pipeline step: {step}")
            if exit_code != 0:
                return exit_code
        return 0
