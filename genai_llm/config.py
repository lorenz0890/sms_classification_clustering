"""Configuration models for the genai_llm package."""

from __future__ import annotations

from dataclasses import dataclass, field
import pathlib
from typing import Any, Mapping, Sequence

from utils import (
    require_bool,
    require_non_negative_int,
    require_positive_int,
    require_str,
)


_ALLOWED_PIPELINE = ("embeddings", "cluster", "visualize", "analyze")
_DEFAULT_PIPELINE = ("embeddings", "cluster", "visualize", "analyze")
_ALLOWED_COORDS_SOURCE = ("stored", "reduce")
_ALLOWED_CLUSTER_ALGOS = ("hdbscan", "dbscan", "kmeans")
_ALLOWED_EMBEDDING_PROVIDERS = ("openai", "gemini")
_DEFAULT_EMBEDDING_MODELS = {
    "openai": "text-embedding-3-small",
    "gemini": "models/gemini-embedding-001",
}
_DEFAULT_EMBEDDING_PARAMS = {
    "openai": {},
    "gemini": {
        "task_type": "clustering",
        "output_dimensionality": 384,
    },
}
_ALLOWED_REDUCTION_ALGOS = ("pca", "kernel_pca", "svd")
_DEFAULT_REDUCTION_PARAMS = {
    "pca": {},
    "kernel_pca": {},
    "svd": {},
}
_DEFAULT_CLUSTER_REDUCE_DIMS = 50
_DEFAULT_PLOT_REDUCE_DIMS = 2
_DEFAULT_HDBSCAN_PARAMS = {
    "min_samples": 10,
    "min_cluster_size": 15,
    "metric": "euclidean",
}
_DEFAULT_DBSCAN_PARAMS = {"eps": 0.5, "min_samples": 5, "metric": "euclidean"}
_DEFAULT_KMEANS_PARAMS = {"n_clusters": 8, "n_init": 10}


def _default_cache_dir() -> str:
    """Return the default cache directory inside the package."""
    return str(pathlib.Path(__file__).resolve().parent / "cache")


@dataclass
class EmbeddingsConfig:
    """Configuration for embedding computation."""

    provider: str = "openai"
    path: str = "data/SMSSpamCollection"
    model: str = _DEFAULT_EMBEDDING_MODELS["openai"]
    batch_size: int = 128
    limit: int = 0
    output_npz: str = "sms_embeddings.npz"
    params: dict = field(
        default_factory=lambda: dict(_DEFAULT_EMBEDDING_PARAMS["openai"])
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EmbeddingsConfig":
        """Create a config from a dictionary."""
        provider = data.get("provider", cls.provider)
        if isinstance(provider, str) and provider.lower() == "google":
            provider = "gemini"
        model = data.get("model")
        if model is None and isinstance(provider, str):
            model = _DEFAULT_EMBEDDING_MODELS.get(provider.lower(), cls.model)
        params = data.get("params")
        if params is None and isinstance(provider, str):
            params = dict(_DEFAULT_EMBEDDING_PARAMS.get(provider.lower(), {}))
        return cls(
            provider=provider,
            path=data.get("path", cls.path),
            model=model,
            batch_size=data.get("batch_size", cls.batch_size),
            limit=data.get("limit", cls.limit),
            output_npz=data.get("output_npz", cls.output_npz),
            params=params,
        )

    def validate(self) -> None:
        """Validate configuration values."""
        self.provider = require_str(self.provider, "embeddings.provider")
        provider_lower = self.provider.lower()
        if provider_lower not in _ALLOWED_EMBEDDING_PROVIDERS:
            raise ValueError(
                f"embeddings.provider must be one of {_ALLOWED_EMBEDDING_PROVIDERS}."
            )
        self.path = require_str(self.path, "embeddings.path")
        self.model = require_str(self.model, "embeddings.model")
        self.batch_size = require_positive_int(self.batch_size, "embeddings.batch_size")
        self.limit = require_non_negative_int(self.limit, "embeddings.limit")
        self.output_npz = require_str(self.output_npz, "embeddings.output_npz")
        if not isinstance(self.params, dict):
            raise ValueError("embeddings.params must be a JSON object.")
        self.provider = provider_lower


@dataclass
class ReductionConfig:
    """Configuration for dimensionality reduction."""

    algorithm: str = "pca"
    params: dict = field(default_factory=lambda: dict(_DEFAULT_REDUCTION_PARAMS["pca"]))
    dims: int = _DEFAULT_CLUSTER_REDUCE_DIMS

    @classmethod
    def from_dict(cls, data: Any, default_dims: int) -> "ReductionConfig":
        """Create a config from a dictionary or string."""
        if data is None:
            data = {}
        if isinstance(data, str):
            algorithm = data
            params = None
            dims = default_dims
        elif isinstance(data, Mapping):
            algorithm = data.get("algorithm", cls.algorithm)
            params = data.get("params")
            dims = data.get("dims", default_dims)
        else:
            raise ValueError("reduce must be a string or JSON object.")
        if isinstance(algorithm, str) and algorithm.lower() == "kernelpca":
            algorithm = "kernel_pca"
        if params is None:
            algo_key = algorithm.lower() if isinstance(algorithm, str) else ""
            params = dict(_DEFAULT_REDUCTION_PARAMS.get(algo_key, {}))
        if isinstance(params, dict):
            params = dict(params)
        return cls(algorithm=algorithm, params=params, dims=dims)

    def validate(self, name: str) -> None:
        """Validate reduction configuration values."""
        self.algorithm = require_str(self.algorithm, f"{name}.algorithm")
        algo_lower = self.algorithm.lower()
        if algo_lower not in _ALLOWED_REDUCTION_ALGOS:
            raise ValueError(
                f"{name}.algorithm must be one of {_ALLOWED_REDUCTION_ALGOS}."
            )
        if not isinstance(self.params, dict):
            raise ValueError(f"{name}.params must be a JSON object.")
        self.dims = require_positive_int(self.dims, f"{name}.dims")


@dataclass
class ClusterConfig:
    """Configuration for clustering computation."""

    input_embeddings: str = "sms_embeddings.npz"
    algorithm: str = "hdbscan"
    seed: int = 42
    params: dict = field(default_factory=lambda: dict(_DEFAULT_HDBSCAN_PARAMS))
    reduce: ReductionConfig = field(default_factory=ReductionConfig)
    output_npz: str = "sms_clusters.npz"
    output_tsv: str = "sms_clusters.tsv"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClusterConfig":
        """Create a config from a dictionary."""
        algorithm = data.get("algorithm", cls.algorithm)
        seed = data.get("seed", cls.seed)
        params = data.get("params")
        if params is None:
            algo_lower = algorithm.lower() if isinstance(algorithm, str) else ""
            if algo_lower == "dbscan":
                params = dict(_DEFAULT_DBSCAN_PARAMS)
            elif algo_lower == "kmeans":
                params = dict(_DEFAULT_KMEANS_PARAMS)
            else:
                params = dict(_DEFAULT_HDBSCAN_PARAMS)
        if isinstance(params, dict):
            params = dict(params)
            if isinstance(algorithm, str) and algorithm.lower() == "kmeans":
                params.setdefault("random_state", seed)
        reduce_config = ReductionConfig.from_dict(
            data.get("reduce"), data.get("pca_dims", _DEFAULT_CLUSTER_REDUCE_DIMS)
        )
        return cls(
            input_embeddings=data.get("input_embeddings", cls.input_embeddings),
            algorithm=algorithm,
            seed=seed,
            params=params,
            reduce=reduce_config,
            output_npz=data.get("output_npz", cls.output_npz),
            output_tsv=data.get("output_tsv", cls.output_tsv),
        )

    def validate(self) -> None:
        """Validate configuration values."""
        self.input_embeddings = require_str(
            self.input_embeddings, "cluster.input_embeddings"
        )
        self.algorithm = require_str(self.algorithm, "cluster.algorithm")
        if self.algorithm.lower() not in _ALLOWED_CLUSTER_ALGOS:
            raise ValueError(
                f"cluster.algorithm must be one of {_ALLOWED_CLUSTER_ALGOS}."
            )
        if not isinstance(self.params, dict):
            raise ValueError("cluster.params must be a JSON object.")
        self.seed = require_non_negative_int(self.seed, "cluster.seed")
        if self.algorithm.lower() == "kmeans":
            self.params.setdefault("random_state", self.seed)
        self.reduce.validate("cluster.reduce")
        self.output_npz = require_str(self.output_npz, "cluster.output_npz")
        self.output_tsv = require_str(self.output_tsv, "cluster.output_tsv")


@dataclass
class VisualizeConfig:
    """Configuration for cluster visualization."""

    input: str = "sms_clusters.npz"
    reduce: ReductionConfig = field(
        default_factory=lambda: ReductionConfig(dims=_DEFAULT_PLOT_REDUCE_DIMS)
    )
    coords_source: str = "stored"
    output: str = "sms_clusters.png"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VisualizeConfig":
        """Create a config from a dictionary."""
        return cls(
            input=data.get("input", cls.input),
            reduce=ReductionConfig.from_dict(
                data.get("reduce"), _DEFAULT_PLOT_REDUCE_DIMS
            ),
            coords_source=data.get("coords_source", cls.coords_source),
            output=data.get("output", cls.output),
        )

    def validate(self) -> None:
        """Validate configuration values."""
        self.input = require_str(self.input, "visualize.input")
        self.reduce.validate("visualize.reduce")
        if self.reduce.dims != _DEFAULT_PLOT_REDUCE_DIMS:
            raise ValueError(
                f"visualize.reduce.dims must be {_DEFAULT_PLOT_REDUCE_DIMS}."
            )
        if self.coords_source not in _ALLOWED_COORDS_SOURCE:
            raise ValueError(
                f"visualize.coords_source must be one of {_ALLOWED_COORDS_SOURCE}."
            )
        self.output = require_str(self.output, "visualize.output")


@dataclass
class AnalyzeConfig:
    """Configuration for cluster n-gram analysis."""

    input: str = "sms_clusters.npz"
    reduce: ReductionConfig = field(
        default_factory=lambda: ReductionConfig(dims=_DEFAULT_PLOT_REDUCE_DIMS)
    )
    coords_source: str = "stored"
    output: str = "sms_clusters_ngrams.png"
    top_k: int = 3
    n: int = 3
    remove_stopwords: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalyzeConfig":
        """Create a config from a dictionary."""
        return cls(
            input=data.get("input", cls.input),
            reduce=ReductionConfig.from_dict(
                data.get("reduce"), _DEFAULT_PLOT_REDUCE_DIMS
            ),
            coords_source=data.get("coords_source", cls.coords_source),
            output=data.get("output", cls.output),
            top_k=data.get("top_k", cls.top_k),
            n=data.get("n", cls.n),
            remove_stopwords=data.get("remove_stopwords", cls.remove_stopwords),
        )

    def validate(self) -> None:
        """Validate configuration values."""
        self.input = require_str(self.input, "analyze.input")
        self.reduce.validate("analyze.reduce")
        if self.reduce.dims != _DEFAULT_PLOT_REDUCE_DIMS:
            raise ValueError(
                f"analyze.reduce.dims must be {_DEFAULT_PLOT_REDUCE_DIMS}."
            )
        if self.coords_source not in _ALLOWED_COORDS_SOURCE:
            raise ValueError(
                f"analyze.coords_source must be one of {_ALLOWED_COORDS_SOURCE}."
            )
        self.output = require_str(self.output, "analyze.output")
        self.top_k = require_positive_int(self.top_k, "analyze.top_k")
        self.n = require_positive_int(self.n, "analyze.n")
        self.remove_stopwords = require_bool(
            self.remove_stopwords, "analyze.remove_stopwords"
        )


@dataclass
class GenaiLLMConfig:
    """Root configuration for the genai_llm package."""

    pipeline: Sequence[str] = field(default_factory=lambda: _DEFAULT_PIPELINE)
    cache_dir: str = field(default_factory=_default_cache_dir)
    output_dir: str = "output"
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    visualize: VisualizeConfig = field(default_factory=VisualizeConfig)
    analyze: AnalyzeConfig = field(default_factory=AnalyzeConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GenaiLLMConfig":
        """Create a config from a dictionary."""
        return cls(
            pipeline=tuple(data.get("pipeline", _DEFAULT_PIPELINE)),
            cache_dir=data.get("cache_dir", _default_cache_dir()),
            output_dir=data.get("output_dir", cls.output_dir),
            embeddings=EmbeddingsConfig.from_dict(data.get("embeddings", {})),
            cluster=ClusterConfig.from_dict(data.get("cluster", {})),
            visualize=VisualizeConfig.from_dict(data.get("visualize", {})),
            analyze=AnalyzeConfig.from_dict(data.get("analyze", {})),
        )

    def validate(self) -> None:
        """Validate configuration values."""
        if not isinstance(self.pipeline, (list, tuple)) or not self.pipeline:
            raise ValueError("pipeline must be a non-empty list of steps.")
        for step in self.pipeline:
            if step not in _ALLOWED_PIPELINE:
                raise ValueError(f"Unsupported pipeline step: {step}")
        self.cache_dir = require_str(self.cache_dir, "cache_dir")
        self.output_dir = require_str(self.output_dir, "output_dir")
        self.embeddings.validate()
        self.cluster.validate()
        self.visualize.validate()
        self.analyze.validate()
