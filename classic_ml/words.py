"""Word frequency analysis for SMS data."""

from __future__ import annotations

import sys
from typing import Iterable, Optional

from utils import tokenize

from .config import WordsConfig
from .plotting import plot_label_bars
from .text_stats import count_items_by_label, load_dataset


class WordFrequencyAnalyzer:
    """Analyze word frequencies for ham/spam SMS messages."""

    def __init__(self, config: WordsConfig) -> None:
        """Initialize the analyzer with configuration."""
        self._config = config

    def _word_items(
        self, text: str, stopwords_set: Optional[set[str]]
    ) -> Iterable[str]:
        """Yield tokenized words for a message."""
        return tokenize(text, stopwords_set=stopwords_set)

    def run(self) -> int:
        """Run the word frequency analysis and print results."""
        self._config.validate()
        try:
            path, stopwords_set = load_dataset(
                self._config.path, self._config.remove_stopwords
            )
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        counters = count_items_by_label(path, stopwords_set, self._word_items)
        label_items = {}
        for label in ("ham", "spam"):
            items = counters[label].most_common(self._config.top)
            label_items[label] = items
            print(label.upper())
            for word, count in items:
                print(f"{word}\t{count}")
            print()
        if not plot_label_bars(self._config.output, label_items, "Top Words"):
            print(
                "Missing matplotlib. Install it in your venv to generate plots.",
                file=sys.stderr,
            )
            return 0
        print(f"wrote {self._config.output}")
        return 0
