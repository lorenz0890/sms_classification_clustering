"""Configuration models for the classic_ml package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from utils import (
    require_bool,
    require_non_negative_int,
    require_positive_int,
    require_ratio,
    require_str,
)
from .classifiers.registry import list_supported_classifiers, normalize_classifier_id


_ALLOWED_PIPELINE = ("words", "ngrams", "classifier")
_DEFAULT_PIPELINE = ("words", "ngrams", "classifier")
_DEFAULT_CLASSIFIERS = ("naive_bayes",)


@dataclass
class WordsConfig:
    """Configuration for word frequency analysis."""

    path: str = "data/SMSSpamCollection"
    top: int = 25
    remove_stopwords: bool = False
    output: str = "classic_words.png"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WordsConfig":
        """Create a config from a dictionary."""
        return cls(
            path=data.get("path", cls.path),
            top=data.get("top", cls.top),
            remove_stopwords=data.get("remove_stopwords", cls.remove_stopwords),
            output=data.get("output", cls.output),
        )

    def validate(self) -> None:
        """Validate configuration values."""
        self.path = require_str(self.path, "words.path")
        self.top = require_positive_int(self.top, "words.top")
        self.remove_stopwords = require_bool(
            self.remove_stopwords, "words.remove_stopwords"
        )
        self.output = require_str(self.output, "words.output")


@dataclass
class NgramsConfig:
    """Configuration for n-gram frequency analysis."""

    path: str = "data/SMSSpamCollection"
    top: int = 25
    n: int = 2
    remove_stopwords: bool = False
    output: str = "classic_ngrams.png"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NgramsConfig":
        """Create a config from a dictionary."""
        return cls(
            path=data.get("path", cls.path),
            top=data.get("top", cls.top),
            n=data.get("n", cls.n),
            remove_stopwords=data.get("remove_stopwords", cls.remove_stopwords),
            output=data.get("output", cls.output),
        )

    def validate(self) -> None:
        """Validate configuration values."""
        self.path = require_str(self.path, "ngrams.path")
        self.top = require_positive_int(self.top, "ngrams.top")
        self.n = require_positive_int(self.n, "ngrams.n")
        self.remove_stopwords = require_bool(
            self.remove_stopwords, "ngrams.remove_stopwords"
        )
        self.output = require_str(self.output, "ngrams.output")


@dataclass
class ClassifierConfig:
    """Configuration for classic ML classifiers."""

    path: str = "data/SMSSpamCollection"
    test_ratio: float = 0.2
    seed: int = 42
    top_words: int = 2000
    show_informative: int = 0
    remove_stopwords: bool = False
    classifiers: Sequence[str] = field(default_factory=lambda: _DEFAULT_CLASSIFIERS)
    output: str = "classic_classifiers.png"
    results_output: str = "classic_classifier_results.json"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClassifierConfig":
        """Create a config from a dictionary."""
        classifiers = data.get("classifiers", _DEFAULT_CLASSIFIERS)
        if isinstance(classifiers, str):
            classifiers = [classifiers]
        if classifiers is None:
            classifiers = _DEFAULT_CLASSIFIERS
        if not isinstance(classifiers, (list, tuple)):
            raise ValueError("classifier.classifiers must be a list of strings.")
        normalized = []
        for classifier_id in classifiers:
            if not isinstance(classifier_id, str):
                raise ValueError("classifier.classifiers must be a list of strings.")
            normalized.append(normalize_classifier_id(classifier_id))
        return cls(
            path=data.get("path", cls.path),
            test_ratio=data.get("test_ratio", cls.test_ratio),
            seed=data.get("seed", cls.seed),
            top_words=data.get("top_words", cls.top_words),
            show_informative=data.get("show_informative", cls.show_informative),
            remove_stopwords=data.get("remove_stopwords", cls.remove_stopwords),
            classifiers=tuple(normalized),
            output=data.get("output", cls.output),
            results_output=data.get("results_output", cls.results_output),
        )

    def validate(self) -> None:
        """Validate configuration values."""
        self.path = require_str(self.path, "classifier.path")
        self.test_ratio = require_ratio(self.test_ratio, "classifier.test_ratio")
        self.seed = require_non_negative_int(self.seed, "classifier.seed")
        self.top_words = require_positive_int(self.top_words, "classifier.top_words")
        self.show_informative = require_non_negative_int(
            self.show_informative, "classifier.show_informative"
        )
        self.remove_stopwords = require_bool(
            self.remove_stopwords, "classifier.remove_stopwords"
        )
        if not isinstance(self.classifiers, (list, tuple)) or not self.classifiers:
            raise ValueError("classifier.classifiers must be a non-empty list.")
        allowed = list_supported_classifiers()
        normalized = []
        for classifier_id in self.classifiers:
            if not isinstance(classifier_id, str):
                raise ValueError("classifier.classifiers must be a list of strings.")
            token = normalize_classifier_id(classifier_id)
            if token not in allowed:
                raise ValueError(f"classifier.classifiers must be one of {allowed}.")
            normalized.append(token)
        self.classifiers = tuple(normalized)
        self.output = require_str(self.output, "classifier.output")
        self.results_output = require_str(
            self.results_output, "classifier.results_output"
        )


@dataclass
class ClassicMLConfig:
    """Root configuration for the classic_ml package."""

    pipeline: Sequence[str] = field(default_factory=lambda: _DEFAULT_PIPELINE)
    output_dir: str = "output"
    cache_dir: str = "classic_ml/cache"
    words: WordsConfig = field(default_factory=WordsConfig)
    ngrams: NgramsConfig = field(default_factory=NgramsConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClassicMLConfig":
        """Create a config from a dictionary."""
        return cls(
            pipeline=tuple(data.get("pipeline", _DEFAULT_PIPELINE)),
            output_dir=data.get("output_dir", cls.output_dir),
            cache_dir=data.get("cache_dir", cls.cache_dir),
            words=WordsConfig.from_dict(data.get("words", {})),
            ngrams=NgramsConfig.from_dict(data.get("ngrams", {})),
            classifier=ClassifierConfig.from_dict(data.get("classifier", {})),
        )

    def validate(self) -> None:
        """Validate configuration values."""
        if not isinstance(self.pipeline, (list, tuple)) or not self.pipeline:
            raise ValueError("pipeline must be a non-empty list of steps.")
        for step in self.pipeline:
            if step not in _ALLOWED_PIPELINE:
                raise ValueError(f"Unsupported pipeline step: {step}")
        self.output_dir = require_str(self.output_dir, "output_dir")
        self.cache_dir = require_str(self.cache_dir, "cache_dir")
        self.words.validate()
        self.ngrams.validate()
        self.classifier.validate()
