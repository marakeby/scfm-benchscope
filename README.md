# scFM_eval

**scFM_eval** is an evaluation framework for single-cell foundation models. It runs a standard pipeline so you can compare how well different embedding models perform on the same datasets and tasks.


**Core Functionality**

1. **Loads a dataset** (e.g. an `.h5ad` file) and optional QC/preprocessing (normalization, HVG, train/test splits).
2. **Computes cell embeddings** using one of the supported models (Stack, scGPT, Geneformer, STATE, scConcept, SCSimilarity, etc.). Each model is an “extractor” that turns expression (or counts) into a fixed-size embedding per cell.
3. **Runs evaluations** on those embeddings. Evaluations depend on the task and can include:
   - **Batch integration**: batch-effect metrics (e.g. k-NN LISI), biological signal preservation, annotation (e.g. k-NN accuracy on cell type).
   - **Downstream tasks**: e.g. cancer subtype classification, treatment response, survival prediction, drug response (Depmap), so you can compare models on real benchmarks.

You choose a **YAML config** that specifies: which dataset, which embedding method, and which evaluations to run. Outputs (embeddings, metrics, plots) are written to a central output directory. The same config layout supports both single runs and parallel batch runs over many configs.

## Configuration

All paths are driven by environment variables so the codebase has no hardcoded absolute paths. Use a `.env` file in the repo root (or export variables in your shell).

### Required variables

Set these in `.env` (see `.env.example` for a template):

| Variable | Description |
|----------|-------------|
| `DATASETS_PATH` | Root for datasets; put `.h5ad` files here (or in subdirs). YAML `dataset.path` is relative to this. |
| `MODEL_CHECKPOINTS` | Root for model weights; each model has its own subdir (STACK, scGPT, VCC_checkpoints, etc.). |
| `OUTPUT_PATH` | Directory where all experiment outputs and embeddings are written. |

**Checkpoints vs repos:** `MODEL_CHECKPOINTS` holds the **saved weights**; repo paths (below) point to the **source code** of each model. For extractors that run in a separate env, the repo is added to `PYTHONPATH` so the code can load the weights from the checkpoint path. You need both when using that model.

### Optional variables

| Variable | Description |
|----------|-------------|
| `STACK_REPO_ROOT` | Path to the Stack **repo** (source code; used for stack extractor subprocess). |
| `STATE_REPO_PATH` | Path to the STATE **repo** (for `pip install -e` in extractor env). |
| `SCGPT_REPO_PATH` | Path to the scGPT **repo**. |
| `GENEFORMER_REPO_PATH` | Path to the Geneformer **repo**. |
| `UCE_REPO_PATH` | Path to the UCE **repo** (used in PYTHONPATH). |
| `OMNICELL_CHECKPOINT_PATH` | Full path to Omnicell checkpoint (e.g. `.pkl` or `.pt`). Set once in `.env`; all Omnicell configs use it. If unset, falls back to `MODEL_CHECKPOINTS/Omnicell/omnicell_checkpoint_epoch27.pt`. |
| `OMNICELL_CHECKPOINTS_BASE` | Optional override for Omnicell fallback path root (defaults to `MODEL_CHECKPOINTS` if unset). |
| `OMNICELL_BASE_DIR` | Path to the **cell-types repo** (root or `cell_types` package dir). Required for Omnicell. |
| `OMNICELL_DATA_DIR` | Dir with `protocol_embeddings/genes/` (gene list + gene manager). Required for Omnicell. |
| `OMNICELL_PYTHON` | Python executable for Omnicell subprocess (optional; default: current Python). |
| `SCFM_EVAL_PARAMS_PATH` | Override directory for YAML configs (default: `yaml/` in repo). |

If you have a reference setup with concrete paths, copy from `dotenv.txt` into `.env` and adjust as needed. The app loads `.env` via `python-dotenv` on startup.

## Clone and run (minimal uv setup)

After a fresh clone, you can run the one-shot setup with no manual env creation:

1. **Install uv** (if not already): `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `pip install uv`).
2. **From the repo root**, run:
   ```bash
   bash setup/setup_download_all.sh
   ```
   The script will `uv sync` (create `.venv` and install deps from `pyproject.toml`: PyYAML, python-dotenv), then download datasets/repos/checkpoints. Dataset and model dirs default to **sibling directories** of the repo (`../datasets`, `../model_checkpoints`) so no `.env` is required for this step.
3. **Optional:** copy `.env.example` to `.env` and set `DATASETS_PATH`, `MODEL_CHECKPOINTS`, `OUTPUT_PATH` if you want different paths. The same uv env (from `uv sync` in the setup script) includes the core deps needed to run the pipeline (e.g. `run/run_exp.py` with PCA, integration baselines, scib evaluations); see `pyproject.toml`. Model-specific extractors (Stack, STATE, Omnicell, etc.) may require separate envs or repos.

## One-shot setup (download data, clone repos, fetch checkpoints)

Run the script above; it uses default sibling dirs unless you set `DATASETS_PATH` and `MODEL_CHECKPOINTS` in `.env`. It downloads all evaluation datasets, clones model repos, and fetches checkpoints so you can start running experiments:

```bash
cp .env.example .env
# Edit .env: set DATASETS_PATH, MODEL_CHECKPOINTS, OUTPUT_PATH

bash setup/setup_download_all.sh
```

- **Datasets:** DKD, Tabula Sapiens, Lymph Node Atlas, Lung Atlas (LUCA), Hypomap, GTEx v9 (cellxgene .h5ad) → `$DATASETS_PATH/DATASETS/SCFM_datasets/EvalDatasets/cell_identity_batch/`
- **Repos:** stack, state, geneformer, scGPT, scConcept, scsimilarity → `$REPOS_DIR` (default: sibling of `MODEL_CHECKPOINTS` named `repos`)
- **Checkpoints:** Stack-Large, State SE-600M, scConcept, scSimilarity (and instructions for Geneformer/scGPT) → under `$MODEL_CHECKPOINTS`

Options: `--datasets-only`, `--repos-only`, `--checkpoints-only`. Data and model URLs/paths are defined in `setup/dataset_paths.yaml` and `setup/model_paths.yaml` (read by the script).

## Simplest first run (no model checkpoints)

To run the pipeline with **no external model weights or repos**, use a **PCA** config. You only need the three required env vars and a dataset.

1. **Copy and edit env**
   ```bash
   cp .env.example .env
   ```
   Set in `.env`:
   - `DATASETS_PATH` — directory that contains your `.h5ad` files (or subdirs like `cell_identity_batch/dkd.h5ad`).
   - `MODEL_CHECKPOINTS` — any existing directory (e.g. same as `OUTPUT_PATH`); not used for PCA.
   - `OUTPUT_PATH` — where results will be written.

2. **Check setup**
   ```bash
   python scripts/check_setup.py
   python scripts/check_setup.py yaml/batch_bio_integration/pca_qc/assay/dkd_assay.yaml
   ```
   The second line checks that the config’s dataset path exists under `DATASETS_PATH`. Fix any reported missing paths before running.

3. **Run one PCA experiment**
   ```bash
   python run/run_exp.py yaml/batch_bio_integration/pca_qc/assay/dkd_assay.yaml
   ```
   This runs load data → PCA embedding → evaluations. No STACK/scGPT/Omnicell checkpoints or repos are needed.

After that works, add optional env vars and run configs that use other methods (see Optional variables and config layout below).

## Quick start (with a specific model)

```bash
# 1. Copy env template and set your paths (including any repo/checkpoint paths for your chosen model)
cp .env.example .env
# Edit .env: DATASETS_PATH, MODEL_CHECKPOINTS, OUTPUT_PATH, and e.g. OMNICELL_BASE_DIR/OMNICELL_DATA_DIR for Omnicell

# 2. (Optional) Validate env and dataset for a config
python scripts/check_setup.py yaml/batch_bio_integration/omnicell/assay/dkd_assay.yaml

# 3. Run a single experiment
python run/run_exp.py yaml/batch_bio_integration/omnicell/assay/dkd_assay.yaml
```

YAML configs can use environment variable expansion (e.g. `${MODEL_CHECKPOINTS}/STACK/bc_large_aligned.ckpt`). Values are expanded when configs are loaded.

## Running experiments

- **Single run:**  
  `python run/run_exp.py <config_path>`  
  Example: `python run/run_exp.py yaml/cancer_survival/pca/tcga_outcomes.yaml`

- **Parallel / batch:**  
  Run many configs in parallel with `python -m parallel_experiments.main`. You can restrict **which** configs run and **which GPUs** to use:

  | Option | Meaning | Example |
  |--------|---------|--------|
  | `-w`, `--workers` | Number of parallel workers (default: 4) | `-w 8` |
  | `--tasks` | Only run configs from these task directories (comma-separated). Task = top-level folder under `yaml/` (e.g. `batch_bio_integration`, `cancer_survival`, `drug_response`). | `--tasks drug_response,cancer_survival` |
  | `-m`, `--method` | Only run this embedding method (e.g. `stack`, `scgpt`, `scconcept`, `pca`). | `-m scconcept` |
  | `-d`, `--dataset` | Only run configs for this dataset (matches YAML filename / `dataset_name`). | `-d perturbseq_competition` |
  | `-s`, `--subgroup` | Only run configs under `task/method/<subgroup>/*.yaml` (e.g. `cancer_TME` under batch_bio_integration). | `--subgroup cancer_TME` |
  | `--gpus` | GPU IDs to use. Sets `CUDA_VISIBLE_DEVICES` so only these GPUs are visible to workers (e.g. `0`, `0,1`, `2,3`). | `--gpus 0,1` |
  | `--force` | Rerun even if results already exist (default: skip completed). | `--force` |
  | `--dry-run` | Print which configs would be run without executing. | `--dry-run` |

  **Examples:**

  ```bash
  # All new experiments, 8 workers
  python -m parallel_experiments.main -w 8

  # Only drug_response and cancer_survival tasks, 4 workers, use GPUs 0 and 1
  python -m parallel_experiments.main --tasks drug_response,cancer_survival -w 4 --gpus 0,1

  # Only batch_bio_integration, only configs under cancer_TME (bassez, chan2021, etc.)
  python -m parallel_experiments.main --tasks batch_bio_integration --subgroup cancer_TME -w 4 --gpus 0,1

  # Only scconcept on dataset perturbseq_competition, force rerun
  python -m parallel_experiments.main -m scconcept -d perturbseq_competition --force -w 2

  # See what would run (no execution)
  python -m parallel_experiments.main --tasks cancer_survival --dry-run
  ```

  By default, the runner **skips** experiments that already have results in the output directory. Use `--force` to rerun them. For more options (e.g. `--yaml-dir`, `--output-dir`), run `python -m parallel_experiments.main --help`.

### Structure of configs and results

Configs and outputs follow the same **nested layout**:

- **Task** — top-level benchmark (e.g. `batch_bio_integration`, `cancer_survival`, `drug_response`). Each task has its own evaluation types and datasets.
- **Model (method)** — embedding method (e.g. `stack`, `scgpt`, `scconcept`, `pca`). Under each task you have one folder per method.
- **Subgroup** — optional: some tasks group datasets into subgroups (e.g. `cancer_TME`, `technical_repeats` under `batch_bio_integration`). So you get `task/method/subgroup/` when present.
- **Dataset** — one YAML per dataset (e.g. `bassez_pre.yaml`, `tcga_outcomes.yaml`). Each YAML is one experiment; results are written under the same path.

So paths look like:

- `yaml/<task>/<method>/<dataset>.yaml` or `yaml/<task>/<method>/<subgroup>/<dataset>.yaml`
- Results: `OUTPUT_PATH/<task>/<method>/<dataset>/` or `OUTPUT_PATH/<task>/<method>/<subgroup>/<dataset>/` (each run produces metrics, embeddings, config copy, etc.)

The parallel runner’s `--tasks`, `-m`, `-d`, and `--subgroup` options filter exactly along this hierarchy.

### Results summarizer

Given that hierarchy, the **results summarizer** (`utils/results_summarizer.py`) **brings together results at each level** and compares them. It reads per-dataset metrics (e.g. `all_results_raw.csv` under each `task/method/[subgroup/]dataset/`) and, at every level, produces **comparison tables and boxplots** across the children:

- **Subgroup-level** (if present): compare datasets within `task/method/subgroup/` → writes to `task/method/subgroup/summaries/`
- **Method-level**: compare all datasets (and subgroups) for that method in the task → `task/method/summaries/`
- **Task-level**: compare all methods and datasets for the task → `task/summaries/`
- **Cross-task**: compare all tasks, methods, and datasets → `OUTPUT_PATH/summaries/`

So you get method-vs-method and dataset-vs-dataset comparisons with tables and boxplots at each scope. Metric values stay in their **original range and direction**; a composite **global_score** (mean of normalized metrics, see `utils/metric_definitions.py`) is used for ranking.

The summarizer runs **automatically after each experiment**. You can turn it off with **`SKIP_SUMMARIES=1`** (env) or **`dataset.skip_summaries: true`** in the YAML for that run. To run it manually on an existing output tree (e.g. after many parallel runs):

```bash
python -m utils.results_summarizer $OUTPUT_PATH
python -m utils.results_summarizer $OUTPUT_PATH --no-plots   # tables only, no boxplots
python -m utils.results_summarizer $OUTPUT_PATH/batch_bio_integration --scope task
```

Config files live under `yaml/` by default (or under `SCFM_EVAL_PARAMS_PATH` if set). Each YAML defines dataset, embedding method, and evaluations.

## Writing and reading a YAML config

Each experiment is defined by a single YAML file with these top-level keys:

| Key | Purpose |
|-----|--------|
| `run_id` | Optional identifier for this run (e.g. for logging); can be `null`. |
| `dataset` | How to load the data: loader class, path to the `.h5ad`, and metadata (label key, batch key, filters, train/test split, etc.). |
| `qc` | Quality control: `min_genes`, `min_cells`, or `skip: true` to disable. |
| `preprocessing` | Normalization and transform: e.g. `normalize: true`, `target_sum: 10000`, `apply_log1p: true`, or `skip: true`. |
| `hvg` | Highly variable genes: `n_top_genes`, `batch_key`, `flavor` (e.g. `seurat`), or `skip: true`. |
| `embedding` | Which model to use and its parameters: `method` (e.g. `stack`, `pca`, `scgpt`), `module`/`class`, and `params` (model-specific). |
| `evaluations` | List of evaluation blocks. Each has `type`, `skip`, and `params` (e.g. `batch_effects`, `biological_signal`, `annotation`, `survival`, `drug_response`). |

**Paths in the YAML**

- **Dataset path** (`dataset.path`): Relative to `DATASETS_PATH` (e.g. `cell_identity_batch/dkd.h5ad`) or absolute with a variable, e.g. `path: ${DATASETS_PATH}/cell_identity_batch/dkd.h5ad`.
- **Model/checkpoint paths** (`embedding.params`): Use env vars so the same config works everywhere, e.g. `checkpoint_path: ${MODEL_CHECKPOINTS}/STACK/bc_large_aligned.ckpt`. Any string value is expanded with `os.path.expandvars` when the config is loaded (so `$VAR` and `${VAR}` both work).


## Extractors

Embedding models (Stack, scGPT, Geneformer, STATE, UCE, etc.) are in `extractors/`. Many use **separate virtual environments** and **local repo paths** (set via the optional env vars above).

- Setup: `cd extractors && ./setup_envs.sh [model]`  
  (Ensure the repo path env vars are set or the script is updated to use them.)
- Details: [extractors/README.md](extractors/README.md)

## Requirements

- Python 3.10+
- Dependencies as in the project (e.g. `pyproject.toml` or `requirements.txt` if present).
- `python-dotenv` for loading `.env`.
