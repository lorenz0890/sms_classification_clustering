"""Naive Bayes classifier strategy."""

from __future__ import annotations

from typing import Sequence

import nltk

from .base import ClassifierStrategy


class NaiveBayesStrategy(ClassifierStrategy):
    """Naive Bayes classifier using NLTK."""

    classifier_id = "naive_bayes"
    name = "Naive Bayes"
    input_type = "dict"

    def train(self, features: object, labels: Sequence[str], seed: int) -> object:
        """Train the Naive Bayes classifier."""
        _ = seed
        train_set = list(zip(features, labels))
        return nltk.NaiveBayesClassifier.train(train_set)

    def predict(self, model: object, features: object) -> Sequence[str]:
        """Predict labels for the feature set."""
        classifier = model
        return [classifier.classify(feature) for feature in features]  # type: ignore[arg-type]

    def show_informative(
        self, model: object, top_n: int, feature_names: Sequence[str] | None = None
    ) -> None:
        """Show most informative features for Naive Bayes."""
        _ = feature_names
        classifier = model
        if top_n > 0:
            classifier.show_most_informative_features(top_n)
