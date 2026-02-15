"""Word frequency analysis for SMS data."""

from __future__ import annotations

import collections
import pathlib
import sys
from typing import Dict, Optional

from utils import ensure_nltk_data, iter_labeled_messages, load_stopwords, tokenize

from .config import WordsConfig
from .plotting import plot_label_bars


class WordFrequencyAnalyzer:
    """Analyze word frequencies for ham/spam SMS messages."""

    def __init__(self, config: WordsConfig) -> None:
        """Initialize the analyzer with configuration."""
        self._config = config

    def _count_words(
        self, path: pathlib.Path, stopwords_set: Optional[set[str]]
    ) -> Dict[str, collections.Counter[str]]:
        """Count word frequencies per label."""
        counters: Dict[str, collections.Counter[str]] = {
            "ham": collections.Counter(),
            "spam": collections.Counter(),
        }
        for label, msg in iter_labeled_messages(path):
            if label not in counters:
                continue
            for word in tokenize(msg, stopwords_set=stopwords_set):
                counters[label][word] += 1
        return counters

    def run(self) -> int:
        """Run the word frequency analysis and print results."""
        self._config.validate()
        path = pathlib.Path(self._config.path)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            return 1

        ensure_nltk_data(include_stopwords=self._config.remove_stopwords)
        stopwords_set = load_stopwords() if self._config.remove_stopwords else None
        counters = self._count_words(path, stopwords_set=stopwords_set)
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
