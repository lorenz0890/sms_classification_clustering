"""SMS classifier evaluation for classic ML workflows."""
from __future__ import annotations

import json
import pathlib
import random
import sys
from typing import List

from sklearn.feature_extraction import DictVectorizer

from utils import ensure_nltk_data, iter_labeled_messages, load_stopwords

from .classifiers import build_classifier_strategy
from .classifiers.base import ClassifierResult, FeatureBuilder, compute_metrics
from .config import ClassifierConfig
from .plotting import plot_classifier_performance


class SmsClassifierRunner:
    """Train and evaluate multiple classifiers for SMS spam detection."""

    def __init__(self, config: ClassifierConfig) -> None:
        """Initialize the runner with configuration."""
        self._config = config

    def _write_results(self, results: List[ClassifierResult]) -> None:
        """Persist classifier results to disk."""
        payload = {
            "train_size": results[0].train_size if results else 0,
            "test_size": results[0].test_size if results else 0,
            "top_words": self._config.top_words,
            "remove_stopwords": self._config.remove_stopwords,
            "seed": self._config.seed,
            "results": [result.as_dict() for result in results],
        }
        output_path = pathlib.Path(self._config.results_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {output_path}")

    def run(self) -> int:
        """Run training/evaluation and print metrics for each classifier."""
        self._config.validate()
        ensure_nltk_data(include_stopwords=self._config.remove_stopwords)
        path = pathlib.Path(self._config.path)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            return 1

        data = list(iter_labeled_messages(path))
        random.Random(self._config.seed).shuffle(data)
        split = int(len(data) * (1.0 - self._config.test_ratio))
        train_data = data[:split]
        test_data = data[split:]

        stopwords_set = load_stopwords() if self._config.remove_stopwords else None
        feature_builder = FeatureBuilder(self._config.top_words, stopwords_set)
        vocab = feature_builder.build_vocab(train_data)
        feature_fn = feature_builder.make_features(vocab)

        train_features = [feature_fn(text) for label, text in train_data]
        test_features = [feature_fn(text) for label, text in test_data]
        train_labels = [label for label, _ in train_data]
        test_labels = [label for label, _ in test_data]

        vectorizer = DictVectorizer(sparse=True)
        train_matrix = vectorizer.fit_transform(train_features)
        test_matrix = vectorizer.transform(test_features)

        results: List[ClassifierResult] = []
        for classifier_id in self._config.classifiers:
            strategy = build_classifier_strategy(classifier_id)
            print(f"[{strategy.name}]")
            if strategy.input_type == "dict":
                model = strategy.train(train_features, train_labels, self._config.seed)
                predictions = strategy.predict(model, test_features)
            else:
                model = strategy.train(train_matrix, train_labels, self._config.seed)
                predictions = strategy.predict(model, test_matrix)

            accuracy, precision, recall, f1, correct, total = compute_metrics(
                test_labels, predictions
            )
            result = ClassifierResult(
                classifier_id=strategy.classifier_id,
                classifier_name=strategy.name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                correct=correct,
                total=total,
                train_size=len(train_labels),
                test_size=len(test_labels),
            )
            results.append(result)

            print(f"train={len(train_labels)} test={len(test_labels)}")
            print(f"accuracy={accuracy:.4f} ({correct}/{total})")
            print(f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
            if self._config.show_informative > 0:
                strategy.show_informative(
                    model,
                    self._config.show_informative,
                    vectorizer.get_feature_names_out(),
                )
            print()

        if results:
            self._write_results(results)

        if not plot_classifier_performance(self._config.output, results):
            print(
                "Missing matplotlib. Install it in your venv to generate plots.",
                file=sys.stderr,
            )
            return 0
        print(f"wrote {self._config.output}")
        return 0
