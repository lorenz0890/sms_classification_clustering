"""Shared classifier strategy interfaces and helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import nltk

from utils import tokenize


Features = Dict[str, bool]
LabeledText = Tuple[str, str]


@dataclass
class ClassifierResult:
    """Container for classifier evaluation metrics."""

    classifier_id: str
    classifier_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    correct: int
    total: int
    train_size: int
    test_size: int

    def as_dict(self) -> Dict[str, object]:
        """Convert the result to a JSON-serializable dictionary."""
        return {
            "classifier_id": self.classifier_id,
            "classifier_name": self.classifier_name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "correct": self.correct,
            "total": self.total,
            "train_size": self.train_size,
            "test_size": self.test_size,
        }


class FeatureBuilder:
    """Build vocabularies and boolean feature dictionaries."""

    def __init__(self, top_words: int, stopwords_set: Optional[set[str]]) -> None:
        """Initialize the feature builder."""
        self._top_words = top_words
        self._stopwords_set = stopwords_set

    def build_vocab(self, examples: Sequence[LabeledText]) -> List[str]:
        """Build a vocabulary from labeled examples."""
        freq = nltk.FreqDist()
        for _, text in examples:
            for word in tokenize(text, stopwords_set=self._stopwords_set):
                freq[word] += 1
        return [word for word, _ in freq.most_common(self._top_words)]

    def make_features(self, vocab: Iterable[str]) -> Callable[[str], Features]:
        """Create a feature extraction function from a vocabulary."""
        vocab_set = set(vocab)

        def features(text: str) -> Features:
            """Create a feature vector for a single text sample."""
            words = set(tokenize(text, stopwords_set=self._stopwords_set))
            return {f"has({word})": (word in words) for word in vocab_set}

        return features


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    positive_label: str = "spam",
) -> Tuple[float, float, float, float, int, int]:
    """Compute accuracy, precision, recall, f1, correct, total."""
    correct = 0
    total = 0
    tp = 0
    fp = 0
    fn = 0
    for true_label, pred_label in zip(y_true, y_pred):
        total += 1
        if pred_label == true_label:
            correct += 1
        if pred_label == positive_label and true_label == positive_label:
            tp += 1
        elif pred_label == positive_label and true_label != positive_label:
            fp += 1
        elif pred_label != positive_label and true_label == positive_label:
            fn += 1
    accuracy = correct / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )
    return accuracy, precision, recall, f1, correct, total


class ClassifierStrategy(ABC):
    """Abstract classifier strategy interface."""

    @property
    @abstractmethod
    def classifier_id(self) -> str:
        """Return the classifier identifier."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the classifier display name."""

    @property
    @abstractmethod
    def input_type(self) -> str:
        """Return the expected feature input type ('dict' or 'vector')."""

    @abstractmethod
    def train(self, features: object, labels: Sequence[str], seed: int) -> object:
        """Train the classifier and return a model."""

    @abstractmethod
    def predict(self, model: object, features: object) -> Sequence[str]:
        """Predict labels for the provided feature set."""

    def show_informative(
        self,
        model: object,
        top_n: int,
        feature_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Optionally display informative features for the classifier."""
        _ = model
        _ = top_n
        _ = feature_names
