from __future__ import annotations

import pathlib
from typing import Iterator, Optional

import nltk
from nltk.corpus import stopwords


def ensure_nltk_data(include_stopwords: bool = False) -> None:
    """Ensure required NLTK resources are available locally."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    if include_stopwords:
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)


def iter_labeled_messages(path: pathlib.Path) -> Iterator[tuple[str, str]]:
    """Yield (label, text) pairs from the SMS dataset file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            label, text = parts[0], parts[1]
            yield label, text


def load_stopwords() -> set[str]:
    """Load English stopwords from NLTK."""
    return set(stopwords.words("english"))


def tokenize(text: str, stopwords_set: Optional[set[str]] = None) -> list[str]:
    """Tokenize a string into alphanumeric words with optional stopword removal."""
    tokens = [tok for tok in nltk.word_tokenize(text.lower()) if tok.isalnum()]
    if stopwords_set:
        tokens = [tok for tok in tokens if tok not in stopwords_set]
    return tokens
