"""Dimensionality reduction helpers for embeddings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from sklearn.decomposition import KernelPCA, PCA, TruncatedSVD


class ReductionStrategy(ABC):
    """Abstract strategy for dimensionality reduction."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a display name for the reduction algorithm."""

    @abstractmethod
    def reduce(self, data: np.ndarray, n_components: int) -> np.ndarray:
        """Reduce embeddings to the requested number of dimensions."""


class _BaseReductionStrategy(ReductionStrategy):
    """Base class for reduction strategies that use n_components."""

    def __init__(self, params: Dict[str, Any], random_state: Optional[int]) -> None:
        """Initialize strategy with parameters and optional random state."""
        self._params = params
        self._random_state = random_state

    def _build_params(self, n_components: int) -> Dict[str, Any]:
        """Build parameters for the reduction estimator."""
        params = dict(self._params)
        params["n_components"] = n_components
        if self._random_state is not None:
            params.setdefault("random_state", self._random_state)
        return params


class PCAReductionStrategy(_BaseReductionStrategy):
    """PCA reduction strategy."""

    @property
    def name(self) -> str:
        """Return a display name for PCA."""
        return "PCA"

    def reduce(self, data: np.ndarray, n_components: int) -> np.ndarray:
        """Reduce embeddings using PCA."""
        params = self._build_params(n_components)
        return PCA(**params).fit_transform(data)


class KernelPCAReductionStrategy(_BaseReductionStrategy):
    """Kernel PCA reduction strategy."""

    @property
    def name(self) -> str:
        """Return a display name for Kernel PCA."""
        return "KernelPCA"

    def reduce(self, data: np.ndarray, n_components: int) -> np.ndarray:
        """Reduce embeddings using Kernel PCA."""
        params = self._build_params(n_components)
        return KernelPCA(**params).fit_transform(data)


class SVDReductionStrategy(_BaseReductionStrategy):
    """Truncated SVD reduction strategy."""

    @property
    def name(self) -> str:
        """Return a display name for Truncated SVD."""
        return "SVD"

    def reduce(self, data: np.ndarray, n_components: int) -> np.ndarray:
        """Reduce embeddings using Truncated SVD."""
        params = self._build_params(n_components)
        return TruncatedSVD(**params).fit_transform(data)


def build_reduction_strategy(
    algorithm: str, params: Dict[str, Any], random_state: Optional[int] = None
) -> ReductionStrategy:
    """Factory for reduction strategies based on algorithm name."""
    algo = algorithm.lower()
    if algo == "pca":
        return PCAReductionStrategy(params, random_state)
    if algo in {"kernel_pca", "kernelpca"}:
        return KernelPCAReductionStrategy(params, random_state)
    if algo == "svd":
        return SVDReductionStrategy(params, random_state)
    raise ValueError(f"Unsupported reduction algorithm: {algorithm}")


def reduce_embeddings(
    embeddings: np.ndarray,
    algorithm: str,
    n_components: int,
    params: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Reduce embeddings to the requested number of dimensions."""
    strategy = build_reduction_strategy(algorithm, params or {}, random_state)
    return strategy.reduce(embeddings, n_components)
