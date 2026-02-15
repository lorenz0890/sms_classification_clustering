"""Support Vector Machine classifier strategy."""
from __future__ import annotations

from typing import Sequence

from sklearn.svm import LinearSVC

from .base import ClassifierStrategy


class SVMStrategy(ClassifierStrategy):
    """Linear SVM classifier."""

    classifier_id = "svm"
    name = "Linear SVM"
    input_type = "vector"

    def train(self, features: object, labels: Sequence[str], seed: int) -> object:
        """Train the SVM classifier."""
        model = LinearSVC(random_state=seed)
        model.fit(features, labels)
        return model

    def predict(self, model: object, features: object) -> Sequence[str]:
        """Predict labels for the feature set."""
        return model.predict(features)
