"""Classifier strategy registry for classic_ml."""

from __future__ import annotations

from typing import Dict, Type

from .base import ClassifierStrategy
from .logistic_regression import LogisticRegressionStrategy
from .naive_bayes import NaiveBayesStrategy
from .registry import list_supported_classifiers, normalize_classifier_id
from .svm import SVMStrategy


_CLASSIFIER_REGISTRY: Dict[str, Type[ClassifierStrategy]] = {
    "naive_bayes": NaiveBayesStrategy,
    "svm": SVMStrategy,
    "logistic_regression": LogisticRegressionStrategy,
}


def build_classifier_strategy(classifier_id: str) -> ClassifierStrategy:
    """Build a classifier strategy by id."""
    normalized = normalize_classifier_id(classifier_id)
    strategy_cls = _CLASSIFIER_REGISTRY.get(normalized)
    if not strategy_cls:
        raise ValueError(f"Unsupported classifier: {classifier_id}")
    return strategy_cls()


__all__ = [
    "ClassifierStrategy",
    "build_classifier_strategy",
    "list_supported_classifiers",
    "normalize_classifier_id",
]
