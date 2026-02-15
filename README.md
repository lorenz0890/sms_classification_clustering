**Setup**
- Python: 3.11
- Clone and enter the repo:

```bash
git clone https://github.com/lorenz0890/sms_classification_clustering
cd sms_classification_clustering
```

- Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

- Install dependencies:

```bash
pip install -r requirements.txt
```

- Format and lint (optional):

```bash
black .
flake8 .
```

- Dataset: place the SMS Spam Collection file at `data/SMSSpamCollection` (tab-separated `label<TAB>text`).

- Set required environment variables for GenAI runs:

```bash
export OPENAI_API_KEY="your-key"    # OpenAI provider
export GOOGLE_API_KEY="your-key"    # Gemini provider (or GEMINI_API_KEY / GOOGLE_GEMINI_API_KEY)
```

**Run Pipelines**
- Classic ML pipeline:

```bash
python main.py --config configs/classic_ml_config.json
```

- GenAI/LLM pipeline:

```bash
python main.py --config configs/genai_llm_config_hdbscan_pca_openai.json
python main.py --config configs/genai_llm_config_hdbscan_kernelpca_openai.json
python main.py --config configs/genai_llm_config_hdbscan_svd_openai.json
python main.py --config configs/genai_llm_config_dbscan_pca_openai.json
python main.py --config configs/genai_llm_config_dbscan_kernelpca_openai.json
python main.py --config configs/genai_llm_config_dbscan_svd_openai.json
python main.py --config configs/genai_llm_config_kmeans_pca_openai.json
python main.py --config configs/genai_llm_config_kmeans_kernelpca_openai.json
python main.py --config configs/genai_llm_config_kmeans_svd_openai.json
```

Swap `_openai` for `_gemini` in the filenames to run Gemini-backed configs.

- Run all configs in the `configs` folder:

```bash
./run_all_configs.sh
./run_all_configs.sh --clean-cache
./run_all_configs.sh --genai-provider openai
./run_all_configs.sh --genai-provider gemini --clean-cache
./run_all_configs.sh --configs-dir configs --clean-cache
```

**Config Notes**
- `package` must be `classic_ml` or `genai_llm`.
- `pipeline` controls the ordered steps to run.
- Classic ML outputs:
  - `output_dir` controls where word/ngram charts and classifier charts are saved.
  - `cache_dir` controls where classifier results JSON files are saved (defaults to `classic_ml/cache`).
  - `words.output` and `ngrams.output` set the filenames for the charts.
  - `classifier.classifiers` lists the classifiers to train (`naive_bayes`, `svm`, `logistic_regression`).
  - `classifier.seed` controls the train/test shuffle and model randomness for reproducibility.
  - `classifier.output` sets the filename for the classifier comparison chart.
  - `classifier.results_output` sets the filename for cached classifier metrics.
- GenAI embeddings:
  - `embeddings.provider` selects `openai` or `gemini`.
  - `embeddings.model` sets the embedding model name for the provider (Gemini embedContent uses `models/gemini-embedding-001` by default).
  - `embeddings.params` supplies provider-specific options (Gemini supports `task_type` like `clustering` and `output_dimensionality` for smaller vectors).
  - Embedding reference docs:
    - Gemini: https://ai.google.dev/gemini-api/docs/embeddings
    - OpenAI: https://developers.openai.com/api/docs/guides/embeddings/
- GenAI clustering:
  - `cluster.algorithm` selects `hdbscan`, `dbscan`, or `kmeans`.
  - `cluster.params` supplies the algorithm parameters passed to scikit-learn.
  - `cluster.seed` controls reduction and KMeans randomness for reproducible runs.
- GenAI reduction:
  - `cluster.reduce.algorithm` selects `pca`, `kernel_pca` (aka `kernelpca`), or `svd`.
  - `cluster.reduce.params` supplies the reducer parameters passed to scikit-learn.
  - `cluster.reduce.dims` controls the embedding dimensionality used for clustering.
  - `visualize.reduce` and `analyze.reduce` select the reducer used when re-projecting to 2D (`dims` must be 2).
  - `visualize.coords_source` and `analyze.coords_source` choose `stored` (use saved 2D coords) or `reduce` (recompute 2D).
- GenAI outputs:
  - Embeddings and cluster artifacts are cached in `genai_llm/cache` by default.
  - Visualizations are written under the top-level `output` directory by default.
  - Scatter plots are saved to `output/clusters_visualized`.
  - Annotated plots (with n-grams) are saved to `output/clusters_visualized_annotated`.
  - Output filenames include the clustering, reduction, and embedding provider names (e.g., `_hdbscan_pca_openai`).
  - Embedding cache filenames also include the provider to avoid mixing results.
  - Cluster metric reports are written to `genai_llm/cache` with `cluster_metrics_*` filenames.
  - `run_all_configs.sh` aggregates these into `output/genai_cluster_comparison.csv` plus accuracy/F1/silhouette/coverage bar charts.
  - Override `output_dir` in `configs/genai_llm_config_*.json` to change the visualization destination.
  - Override `cache_dir` to move embedding/cluster artifacts elsewhere.

**Technical Design**
- Package split:
  - `classic_ml` contains word/ngram analysis and classifier strategies for SMS spam detection. This keeps classical NLP workflows separate from embedding-based pipelines.
  - `genai_llm` contains embedding generation, clustering, visualization, and cluster n-gram analysis. This isolates model-dependent logic and plotting concerns.
- genai_llm subpackages:
  - `genai_llm/cluster` holds clustering compute/visualize/analyze workflows plus plotting and metrics helpers.
  - `genai_llm/embeddings` holds embedding workflow and provider implementations.
  - `genai_llm/algorithms` holds strategy implementations for clustering and reduction.
- Facade pattern:
  - `classic_ml/ClassicMLFacade` and `genai_llm/GenaiLLMFacade` provide a single `run()` entrypoint.
  - The facades implement a consistent pipeline contract, so `main.py` only needs to load config and delegate execution.
  - This makes orchestration predictable and keeps step-level logic encapsulated.
- Pipeline handler maps:
  - Both facades dispatch pipeline steps via `{step: handler}` maps instead of chained conditionals, making step extension safer.
- Strategy pattern for classifiers:
  - `classic_ml/classifiers` holds the Naive Bayes, SVM, and Logistic Regression implementations.
  - `classifier.classifiers` controls which strategies are trained and compared.
  - Classifier IDs and aliases are centralized in `classic_ml/classifiers/registry.py` to keep config validation and factories in sync.
- Strategy pattern for clustering:
  - `ClusteringStrategy` defines the interface for clustering algorithms, with `HDBSCANStrategy`, `DBSCANStrategy`, and `KMeansStrategy` implementations.
  - The factory selects a strategy based on config, so adding a new algorithm only requires a new strategy class.
  - Visualization titles are driven by the stored algorithm name in metadata, keeping plots accurate without hardcoding.
- Strategy pattern for reduction:
  - `ReductionStrategy` defines the interface for reducers, with `PCAReductionStrategy`, `KernelPCAReductionStrategy`, and `SVDReductionStrategy` implementations.
  - The factory selects a strategy based on config, so swapping reducers only changes JSON settings.
- Shared formatting helpers:
  - `genai_llm/cluster/formatting.py` centralizes reduction/provider naming and output filename construction for cluster plots.
  - This keeps visualize/analyze output naming consistent and avoids duplication.
- Configuration and validation:
  - Typed dataclass configs validate inputs early, preventing silent misconfiguration.
  - Algorithm parameters are passed as a `cluster.params` JSON object, which keeps algorithm-specific knobs isolated.
  - This structure supports future backends (e.g., different embedding models or clusterers) without changing the entrypoint API.
- Caching strategy:
  - Embeddings are expensive (API-bound), so `sms_embeddings.npz` is reused if present.
  - Clustering always recomputes to reflect changed algorithm parameters or reduction settings while still using cached embeddings.
  - Cache and output roots are configurable, with defaults aimed at keeping artifacts organized and reproducible.
- Extensibility and reuse:
  - Embedding providers are isolated behind `EmbeddingProvider` for easier replacement.
  - Dimensionality reduction, metrics, and plotting are in focused modules to reduce coupling.
  - Shared NLP utilities (tokenization, stopwords) live in `utils` to avoid duplication across packages.
  - `np.load` calls use context managers to close files promptly, avoiding file-handle leaks.

**Robust Evaluation**
- Note: the steps below are a theoretical robust evaluation pipeline and are not implemented in the current codebase.
- Classic ML (spam/ham): use nested cross-validation over a fixed grid of hyperparameters and N fixed random seeds to get stable estimates of predictive quality.
- GenAI clustering: rerun clustering across different seeds (bootstrap) to assess stability under different initial states and estimate confidence intervals for cluster assignments.
- If using clustering as a classifier, apply a similar evaluation pipeline to classic ML (nested CV + fixed seeds) for robust results.

**Dataset License**
- SMS Spam Collection v.1 (data/SMSSpamCollection) is provided by Tiago Agostinho de Almeida and Jose Maria Gomez Hidalgo.
- Please cite the dataset paper and page when using the corpus: http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/
- The dataset is provided "as is" with no warranty; you are responsible for use and distribution.
