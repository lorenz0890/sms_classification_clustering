"""Shared helpers for classic ML text statistics."""

from __future__ import annotations

import collections
import pathlib
from typing import Callable, Dict, Iterable, Optional, TypeVar

from utils import ensure_nltk_data, iter_labeled_messages, load_stopwords


Item = TypeVar("Item")


def load_dataset(
    path_str: str, remove_stopwords: bool
) -> tuple[pathlib.Path, Optional[set[str]]]:
    """Load the dataset path and stopwords configuration."""
    path = pathlib.Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ensure_nltk_data(include_stopwords=remove_stopwords)
    stopwords_set = load_stopwords() if remove_stopwords else None
    return path, stopwords_set


def count_items_by_label(
    path: pathlib.Path,
    stopwords_set: Optional[set[str]],
    item_fn: Callable[[str, Optional[set[str]]], Iterable[Item]],
) -> Dict[str, collections.Counter[Item]]:
    """Count items per label using a provided extraction function."""
    counters: Dict[str, collections.Counter[Item]] = {
        "ham": collections.Counter(),
        "spam": collections.Counter(),
    }
    for label, msg in iter_labeled_messages(path):
        if label not in counters:
            continue
        for item in item_fn(msg, stopwords_set):
            counters[label][item] += 1
    return counters
