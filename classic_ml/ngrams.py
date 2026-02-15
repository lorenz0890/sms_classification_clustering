"""N-gram frequency analysis for SMS data."""
from __future__ import annotations

import collections
import pathlib
import sys
from typing import Dict, Iterable, Optional, Tuple

import nltk

from utils import ensure_nltk_data, iter_labeled_messages, load_stopwords, tokenize

from .config import NgramsConfig
from .plotting import plot_label_bars


class NgramFrequencyAnalyzer:
    """Analyze n-gram frequencies for ham/spam SMS messages."""

    def __init__(self, config: NgramsConfig) -> None:
        """Initialize the analyzer with configuration."""
        self._config = config

    def _count_ngrams(
        self, path: pathlib.Path, stopwords_set: Optional[set[str]]
    ) -> Dict[str, collections.Counter[Tuple[str, ...]]]:
        """Count n-gram frequencies per label."""
        counters: Dict[str, collections.Counter[Tuple[str, ...]]] = {
            "ham": collections.Counter(),
            "spam": collections.Counter(),
        }
        for label, msg in iter_labeled_messages(path):
            if label not in counters:
                continue
            tokens = tokenize(msg, stopwords_set=stopwords_set)
            for gram in nltk.ngrams(tokens, self._config.n):
                counters[label][gram] += 1
        return counters

    def _format_gram(self, gram: Iterable[str]) -> str:
        """Format an n-gram tuple for display."""
        return " ".join(gram)

    def run(self) -> int:
        """Run the n-gram analysis and print results."""
        self._config.validate()
        path = pathlib.Path(self._config.path)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            return 1

        ensure_nltk_data(include_stopwords=self._config.remove_stopwords)
        stopwords_set = load_stopwords() if self._config.remove_stopwords else None
        counters = self._count_ngrams(path, stopwords_set=stopwords_set)
        label_items = {}
        for label in ("ham", "spam"):
            print(f"{label.upper()} {self._config.n}-GRAMS")
            items = []
            for gram, count in counters[label].most_common(self._config.top):
                formatted = self._format_gram(gram)
                items.append((formatted, count))
                print(f"{formatted}\t{count}")
            label_items[label] = items
            print()
        title_suffix = f"Top {self._config.n}-grams"
        if not plot_label_bars(self._config.output, label_items, title_suffix):
            print(
                "Missing matplotlib. Install it in your venv to generate plots.",
                file=sys.stderr,
            )
            return 0
        print(f"wrote {self._config.output}")
        return 0
