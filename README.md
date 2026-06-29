## scFM_eval

Evaluate single-cell foundation model (scFM) embeddings + downstream classification across multiple datasets/tasks (e.g. BRCA: pre/post, outcome, subtype; LUAD1/LUAD2).

### What it does

- **Load data** from an `.h5ad` file into an `AnnData`
- **QC / preprocessing / HVG** (optional, driven by YAML)
- **Extract embeddings** (PCA, HVG, scVI, Geneformer, scGPT, scFoundation, SCimilarity, CellPLM, mock linear projector for tests, …)
- **Train & evaluate a classifier** on the embeddings (commonly RandomForest)
- **Save outputs** under `OUTPUT_PATH` (see `setup_path.py`)

### Quickstart

The Python package is **`scfm-cancer-eval`** (import **`scfm_cancer_eval`**). After `pixi install` (recommended) or `pip install -e .`, run from the **repo root**:

```bash
pixi run -e default run-exp exp/geneformer/V1-10M-i2048/brca_cell_type.yaml
# or
python -m scfm_cancer_eval.run.run_exp exp/geneformer/V1-10M-i2048/brca_cell_type.yaml
# or
scfm-cancer-eval exp/geneformer/V1-10M-i2048/brca_cell_type.yaml
```

Paths like `exp/...` are relative to **`PARAMS_PATH`** (bundled `yaml/` in the install, unless you set **`SCFM_PARAMS_PATH`**).

Helper scripts: `run/*.sh` (they `cd` to the repo root and call `python -m scfm_cancer_eval.run.run_exp …`).

### Custom models (your YAML, extractor, and weights)

Use this when you want to **evaluate your own embedding model** on the same tasks/loaders/metrics as the bundled benchmarks, and **compare** runs via the shared outputs (see below).

1. **Your code (extractor / loader)**  
   - Implement an embedding class that follows the same contract as **`scfm_cancer_eval.features.extractor.EmbeddingExtractor`** (subclass it when possible; set **`output_key`** or implement **`get_output_key()`**).  
   - Put it in an **installable** package (e.g. `pip install -e /path/to/my_lab_pkg`) or ensure the package root is on **`PYTHONPATH`**.

2. **YAML `module` / `class`**  
   - In the `embedding` block, set **`module`** to a fully importable module path, e.g. **`my_lab.embedders.my_model`**, and **`class`** to your extractor class name.  
   - Bundled configs use short names like **`features.gf_extractor`**; those are rewritten internally to **`scfm_cancer_eval.features.*`**. Anything else is imported **as-is**, so your package name must be importable.

3. **Weights and checkpoints**  
   - Set **`SCFM_MODELS_PATH`** to the directory that holds your checkpoints.  
   - In YAML, use **absolute** paths or **relative** paths under `SCFM_MODELS_PATH` for keys the extractor resolves (same pattern as the bundled Geneformer / scGPT configs).

4. **Your experiment YAML**  
   - **Option A — absolute entry config:**  
     `python -m scfm_cancer_eval.run.run_exp /path/to/my_experiment.yaml`  
     Includes in that file that look like `./fragment.yaml` or `../shared/dataset.yaml` resolve **relative to each YAML file’s directory**. Other relative includes resolve under **`PARAMS_PATH`**.  
   - **Option B — your own tree as `PARAMS_PATH`:**  
     `export SCFM_PARAMS_PATH=/path/to/my_yaml_root`  
     then `python -m scfm_cancer_eval.run.run_exp exp/my_task.yaml` (relative to that root).  
   - **Reusing bundled fragments** while keeping your entry YAML elsewhere: either copy/include paths that point into the installed package’s `yaml/` (absolute path to that directory), or set `SCFM_PARAMS_PATH` to a directory that contains **both** your files and symlinks to the fragments you need.

5. **Data and outputs**  
   - **`SCFM_DATA_PATH`**: prefix for `dataset.path` in YAML (same as before).  
   - **`SCFM_OUTPUT_PATH`**: all run directories and the global **`metrics_runs.csv`** ledger.  
   - Entry YAML **outside** the repo: run artifacts go under **`OUTPUT_PATH/external/<hash>/<stem>/`** so they stay under your output root.

6. **Comparing to baseline / other runs**  
   - Each run writes **`results.json`** (schema **`scfm_eval.results` v1.1.0** — see `scfm_cancer_eval.utils.results_json`), plus **`run_summary.json`**, **`metrics.json`**, and **`resolved_config.yaml`**.  
   - The runner appends one row per run to **`metrics_runs.csv`** under **`OUTPUT_PATH`** (columns include **`config_path`**, **`resolved_entry_yaml`**, **`save_dir`**, **`embedding_method`**, **`embedding_key`**, and metric fields when present).  
   - To compare your model to **previously evaluated** models: use the same **`OUTPUT_PATH`** (or merge ledgers), or join on stable ids you define in **`run_id`** / dataset / task metadata in **`results.json`**.

7. **Optional validation**  
   - Set **`SCFM_VALIDATE_EXP=1`** to run embedding/data constraint checks after merge (see `scfm_cancer_eval.utils.validate_exp_constraints`).

### Mock embedding example (cancer subtyping)

To exercise the full pipeline **without** loading a foundation model, point an experiment YAML at the mock extractor (see bundled **`embedding`** examples under `src/scfm_cancer_eval/yaml/` or define your own YAML using):

- **`module`:** `features.mock_extractor` (resolved to `scfm_cancer_eval.features.mock_extractor`)
- **`class`:** `MockEmbeddingExtractor`

Use **`python -m scfm_cancer_eval.run.run_exp …`** with a suitable YAML path (bundled or custom per section above).

### Pixi environments (optional)

`pixi.toml` defines **separate environments** per model stack (`pixi run -e <name> …`). Examples:

```bash
pixi install
pixi run -e default run-exp exp/geneformer/V1-10M-i2048/brca_cell_type.yaml
pixi run -e scvi run-exp exp/scvi/default/brca_cell_type.yaml
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

Key paths are defined in **`scfm_cancer_eval.setup_path`** (overridable via environment variables):

| Symbol | Role | Override |
|--------|------|----------|
| **`PARAMS_PATH`** | Root for **relative** experiment YAML paths and includes | Default: bundled `yaml/` inside the package. Set **`SCFM_PARAMS_PATH`** to your own directory. |
| **`DATA_PATH`** | Prefix for **`dataset.path`** in YAML | **`SCFM_DATA_PATH`** (default: `./data` under process CWD at import time). |
| **`MODELS_PATH`** | Prefix for relative checkpoint paths in YAML | **`SCFM_MODELS_PATH`** (default: `./models` under CWD). |
| **`OUTPUT_PATH`** | Run directories + **`metrics_runs.csv`** | **`SCFM_OUTPUT_PATH`** (default: `./output` under CWD). |

Convenience: **`source scripts/set_runtime_paths.sh`** after copying **`scripts/config/runtime_paths.env.example`** (optional **`SCFM_PARAMS_PATH`** there).

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

`scfm_cancer_eval.run.run_exp` supports composing configs by deep-merging “fragment” YAMLs before running.

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
At runtime, if `classification.params.label_map` is missing, the experiment runner will inherit it from `dataset.label_map`.

### Unified extractor API (recommended)

Extractors inherit from **`scfm_cancer_eval.features.extractor.EmbeddingExtractor`** and expose an **`output_key`**.
The runner prefers **`extractor.output_key`** (or **`extractor.get_output_key()`**) instead of hard-coding
the embedding key in **`run_exp.py`**.

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

- `src/scfm_cancer_eval/`: installable package (code + bundled `yaml/`)
- `run/`: shell helpers invoking `python -m scfm_cancer_eval.run.run_exp`
- `scripts/`: VM/bootstrap, downloads, validation

