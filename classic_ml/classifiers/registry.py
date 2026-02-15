"""Classifier identifier registry and normalization helpers."""

from __future__ import annotations

from typing import Sequence


_ALLOWED_CLASSIFIERS = ("naive_bayes", "svm", "logistic_regression")
_CLASSIFIER_ALIASES = {
    "naivebayes": "naive_bayes",
    "naive-bayes": "naive_bayes",
    "nb": "naive_bayes",
    "logreg": "logistic_regression",
    "logistic": "logistic_regression",
    "logisticregression": "logistic_regression",
    "logistic-regression": "logistic_regression",
    "linear_svm": "svm",
    "linear-svm": "svm",
}


def normalize_classifier_id(classifier_id: str) -> str:
    """Normalize classifier identifiers."""
    token = classifier_id.strip().lower()
    token = token.replace(" ", "_")
    return _CLASSIFIER_ALIASES.get(token, token)


def list_supported_classifiers() -> Sequence[str]:
    """Return supported classifier identifiers."""
    return _ALLOWED_CLASSIFIERS


__all__ = ["list_supported_classifiers", "normalize_classifier_id"]
