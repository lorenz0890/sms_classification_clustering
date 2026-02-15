"""Strategy pattern for clustering algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans


class ClusteringStrategy(ABC):
    """Abstract strategy for clustering algorithms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a display name for the algorithm."""

    @abstractmethod
    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        """Fit the model and return cluster assignments."""


class HDBSCANStrategy(ClusteringStrategy):
    """HDBSCAN clustering strategy."""

    def __init__(self, params: Dict[str, Any]) -> None:
        """Initialize the strategy with algorithm parameters."""
        cleaned = dict(params)
        cleaned.setdefault("copy", False)
        self._params = cleaned
        self._clusterer = HDBSCAN(**cleaned)

    @property
    def name(self) -> str:
        """Return a display name for HDBSCAN."""
        return "HDBSCAN"

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        """Fit HDBSCAN and return cluster assignments."""
        return self._clusterer.fit_predict(data)


class DBSCANStrategy(ClusteringStrategy):
    """DBSCAN clustering strategy."""

    def __init__(self, params: Dict[str, Any]) -> None:
        """Initialize the strategy with algorithm parameters."""
        self._params = params
        self._clusterer = DBSCAN(**params)

    @property
    def name(self) -> str:
        """Return a display name for DBSCAN."""
        return "DBSCAN"

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        """Fit DBSCAN and return cluster assignments."""
        return self._clusterer.fit_predict(data)


class KMeansStrategy(ClusteringStrategy):
    """KMeans clustering strategy."""

    def __init__(self, params: Dict[str, Any]) -> None:
        """Initialize the strategy with algorithm parameters."""
        self._params = params
        self._clusterer = KMeans(**params)

    @property
    def name(self) -> str:
        """Return a display name for KMeans."""
        return "KMeans"

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        """Fit KMeans and return cluster assignments."""
        return self._clusterer.fit_predict(data)


def build_clustering_strategy(
    algorithm: str, params: Dict[str, Any]
) -> ClusteringStrategy:
    """Factory for clustering strategies based on algorithm name."""
    algo = algorithm.lower()
    if algo == "hdbscan":
        return HDBSCANStrategy(params)
    if algo == "dbscan":
        return DBSCANStrategy(params)
    if algo == "kmeans":
        return KMeansStrategy(params)
    raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
