"""Classifier strategy registry for classic_ml."""
from __future__ import annotations

from typing import Dict, Sequence, Type

from .base import ClassifierStrategy
from .logistic_regression import LogisticRegressionStrategy
from .naive_bayes import NaiveBayesStrategy
from .svm import SVMStrategy


_CLASSIFIER_REGISTRY: Dict[str, Type[ClassifierStrategy]] = {
    "naive_bayes": NaiveBayesStrategy,
    "svm": SVMStrategy,
    "logistic_regression": LogisticRegressionStrategy,
}

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


def build_classifier_strategy(classifier_id: str) -> ClassifierStrategy:
    """Build a classifier strategy by id."""
    normalized = normalize_classifier_id(classifier_id)
    strategy_cls = _CLASSIFIER_REGISTRY.get(normalized)
    if not strategy_cls:
        raise ValueError(f"Unsupported classifier: {classifier_id}")
    return strategy_cls()


def list_supported_classifiers() -> Sequence[str]:
    """Return supported classifier identifiers."""
    return tuple(_CLASSIFIER_REGISTRY.keys())


__all__ = [
    "ClassifierStrategy",
    "build_classifier_strategy",
    "list_supported_classifiers",
    "normalize_classifier_id",
]
