## scFM_eval

Evaluate single-cell foundation model (scFM) embeddings + downstream classification across multiple datasets/tasks (e.g. BRCA: pre/post, outcome, subtype; LUAD1/LUAD2).

### What it does
- **Load data** from an `.h5ad` file into an `AnnData`
- **QC / preprocessing / HVG** (optional, driven by YAML)
- **Extract embeddings** (PCA, HVG, scVI, Geneformer, scGPT, scFoundation, SCimilarity, CellPLM, mock linear projector for tests, …)
- **Train & evaluate a classifier** on the embeddings (commonly RandomForest)
- **Save outputs** under `OUTPUT_PATH` (see `setup_path.py`)

### Quickstart
From the repo root:

```bash
python run/run_exp.py brca_full/pre_post/scvi.yaml
python run/run_exp.py luad1/scgpt.yaml
```

There are helper scripts with example runs:
- `run/run_brca.sh`
- `run/run_luad.sh`

### Mock embedding example (cancer subtyping)
To exercise the full pipeline (QC, HVG, embeddings, visualization, embedding metrics, and downstream classification) **without** loading a foundation model, use the mock linear projector:

```bash
python run/run_exp.py yaml/examples/mock_subtype_eval.yaml
```

This config is the **cancer subtyping** task (ER+ vs TNBC, Pre), reusing the same dataset and classifier fragments as the real subtype runs:

- `yaml/brca_full/subtype/dataset_shared.yaml`
- `yaml/brca_full/subtype/classification_randomforest.yaml`

The `embedding` block points at `features.mock_extractor.MockEmbeddingExtractor` and `models/mock_embedding_model.py`. Swap that block for a real extractor + params to match production experiments.

### Paths & outputs
Key paths are defined in `setup_path.py`:
- `PARAMS_PATH`: points at `yaml/` (where experiment configs live)
- `DATA_PATH`: prefix for dataset paths in YAML (default `/home/jupyter`)
- `OUTPUT_PATH`: where run artifacts are written (default `/home/jupyter/mnt/__output_clean`)

### YAML experiment configs
An experiment YAML ultimately needs these top-level keys (after composition):
- `run_id`
- `dataset`
- `qc`
- `preprocessing`
- `embedding`
- `classification`
- optionally `hvg`

#### Config composition (to reduce redundancy)
`run/run_exp.py` supports composing configs by deep-merging “fragment” YAMLs before running.

Use these include-tags at the top of an experiment YAML:

```yaml
run_id:
dataset:
  - brca_full/pre_post/dataset_h5ad_tcells_only.yaml
classification:
  - brca_full/pre_post/classification_randomforest_pre_post.yaml

# Optional per-experiment overrides go below
embedding:
  method: scVI
  module: features.scvi_extractor
  class: scVIEmbeddingExtractor
  viz: True
  eval: True
  params: {}
```

Notes:
- **Include-tags vs config blocks**: if `dataset:` is a dict, it’s treated as the dataset config; if it’s a string/list, it’s treated as include paths.
- **Merge order**: `dataset` includes → `classification` includes → `bases/defaults` (legacy) → local overrides.
- **Lists are replaced** during merge; dicts are merged recursively.

#### `label_map` convention
For many tasks, the class mapping is stored under `dataset.label_map` in the dataset fragment.
At runtime, if `classification.params.label_map` is missing, `run/run_exp.py` will inherit it from `dataset.label_map`.

### Unified extractor API (recommended)
Extractors inherit from `features/extractor.py::EmbeddingExtractor` and expose an `output_key`.
The runner now prefers `extractor.output_key` (or `extractor.get_output_key()`) instead of hard-coding
the embedding key in `run/run_exp.py`.

To override the output key in YAML, set:
- `embedding.output_key: X_custom_key`

### Standard run outputs (recommended)
Each run directory now includes:
- `resolved_config.yaml`: fully merged config used for the run
- `run_summary.json`: high-level run metadata
- `metrics.json`: embedding + classification metrics (best-effort)

Additionally, the runner appends one row per run to:
- `metrics_runs.csv` under `OUTPUT_PATH`

### Repository layout (high level)
- `run/`: entrypoints and runner (`run_exp.py`)
- `yaml/`: experiment configurations + fragments
- `data/`: dataset loaders (`H5ADLoader`, etc.)
- `features/`: embedding extractors
- `models/`: classifiers / finetuning modules
- `evaluation/`, `viz/`: evaluation + visualization
