"""N-gram frequency analysis for SMS data."""

from __future__ import annotations

import sys
from typing import Iterable, Optional, Tuple

import nltk

from utils import tokenize

from .config import NgramsConfig
from .plotting import plot_label_bars
from .text_stats import count_items_by_label, load_dataset


class NgramFrequencyAnalyzer:
    """Analyze n-gram frequencies for ham/spam SMS messages."""

    def __init__(self, config: NgramsConfig) -> None:
        """Initialize the analyzer with configuration."""
        self._config = config

    def _ngram_items(
        self, text: str, stopwords_set: Optional[set[str]]
    ) -> Iterable[Tuple[str, ...]]:
        """Yield n-gram tuples for a message."""
        tokens = tokenize(text, stopwords_set=stopwords_set)
        return nltk.ngrams(tokens, self._config.n)

    def _format_gram(self, gram: Iterable[str]) -> str:
        """Format an n-gram tuple for display."""
        return " ".join(gram)

    def run(self) -> int:
        """Run the n-gram analysis and print results."""
        self._config.validate()
        try:
            path, stopwords_set = load_dataset(
                self._config.path, self._config.remove_stopwords
            )
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        counters = count_items_by_label(path, stopwords_set, self._ngram_items)
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
