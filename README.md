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

### Pixi environments (optional)

`pixi.toml` defines **separate environments** per model stack (`pixi run -e <name> …`). Examples:

```bash
pixi install
pixi run -e default run-exp yaml/examples/mock_subtype_eval.yaml
pixi run -e scvi run-exp yaml/brca_full/subtype/scvi.yaml
pixi run -e scconcept run-exp …      # after extractors + YAML exist for scConcept
pixi run -e nicheformer run-exp …   # heavy deps (Merlin/dask); prefer Linux + CUDA for upstream workflows
```

- **`default`**: shared evaluation harness (no extra model packages).
- **`geneformer`**: installs `transformers` / `datasets`; the **package** is installed by **`scripts/install_packages.sh`** (clone into `third_party/Geneformer/`, then `pip install -e` into the `geneformer` env).
- **`nicheformer`** / **`scconcept`**: **isolated** envs (`no-default-feature`) with git installs + harness deps for `run_exp` / `scib`. Add YAML + extractors when you wire those models into `features/`.

Commit `pixi.toml` and `pixi.lock`; keep `.pixi/` out of git (see `.gitignore`).

### Setup scripts (packages, weights, data)

| Script | Purpose |
|--------|---------|
| **`scripts/install_packages.sh`** | `pixi install` and editable Python installs (Geneformer: `GIT_LFS_SKIP_SMUDGE=1` clone, then `pip install -e`). Flags: `--pixi-only`, `--geneformer-only`. |
| **`scripts/download_model_weights.sh`** | Large checkpoints: optional `git lfs pull` in `third_party/Geneformer`, optional Hugging Face `snapshot_download` into `MODEL_ROOT`. Configure `scripts/config/model_weights.env` (copy from `model_weights.env.example`). |
| **`scripts/download_data.sh`** | Datasets: `gsutil rsync`, HTTP download, or `rsync`. Configure `scripts/config/data_download.env` (copy from `data_download.env.example`). |

Pixi shortcuts: `pixi run install-packages`, `pixi run download-models`, `pixi run download-data`.

### GCP / CUDA dev VM: script vs Docker

| Approach | Best for |
|----------|-----------|
| **Bootstrap script** | You SSH into the VM, use the same repo layout as locally, want fast iteration. |
| **Docker** | Reproducible image, CI, sharing an exact stack; needs NVIDIA Container Toolkit + `docker run --gpus all`. |

**On a fresh GCP GPU VM:** install `git`, `git-lfs`, NVIDIA driver; run **`bash scripts/bootstrap_vm.sh`**, then **`bash scripts/download_model_weights.sh`** and **`bash scripts/download_data.sh`** after copying/editing the `scripts/config/*.env` examples. Check **`pixi run verify-gpu`**. If `cuda_available=False`, add CUDA PyTorch channels in `pixi.toml` for Linux or use **`Dockerfile`**.

**Docker:** `docker build -t scfm-eval .` then `docker run --gpus all ...`. The image runs `install_packages.sh` at build time when the network is available.

### Paths & outputs

Key paths are defined in `setup_path.py`:

- `PARAMS_PATH`: points at `yaml/` (where experiment configs live)
- `DATA_PATH`: prefix for dataset paths in YAML (default: repo root; override with `SCFM_DATA_PATH`)
- `OUTPUT_PATH`: where run artifacts are written (default: `__output/` under the repo; override with `SCFM_OUTPUT_PATH`)

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

