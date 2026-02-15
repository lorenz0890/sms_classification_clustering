"""Logistic regression classifier strategy."""
from __future__ import annotations

from typing import Sequence

from sklearn.linear_model import LogisticRegression

from .base import ClassifierStrategy


class LogisticRegressionStrategy(ClassifierStrategy):
    """Logistic regression classifier."""

    classifier_id = "logistic_regression"
    name = "Logistic Regression"
    input_type = "vector"

    def train(self, features: object, labels: Sequence[str], seed: int) -> object:
        """Train the logistic regression classifier."""
        model = LogisticRegression(max_iter=1000, random_state=seed)
        model.fit(features, labels)
        return model

    def predict(self, model: object, features: object) -> Sequence[str]:
        """Predict labels for the feature set."""
        return model.predict(features)
