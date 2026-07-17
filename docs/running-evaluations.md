# Running evaluations

## Install the environments

Run commands from the repository root:

```bash
pixi install
```

Some models require source packages that are installed separately:

```bash
pixi run install-packages
```

The helper installs the Pixi environments and the Geneformer source package.
scFoundation has its own setup task:

```bash
pixi run install-scfoundation
```

## Configure runtime paths

Set paths before starting Python or running an experiment:

```bash
export SCFM_DATA_PATH=/mnt/data
export SCFM_MODELS_PATH=/mnt/models
export SCFM_OUTPUT_PATH=/mnt/scfm-results
```

- `SCFM_DATA_PATH` is the root for relative `dataset.path` values.
- `SCFM_MODELS_PATH` is the root for relative checkpoint paths.
- `SCFM_OUTPUT_PATH` receives run directories and `metrics_runs.csv`.
- `SCFM_PARAMS_PATH` optionally replaces the bundled YAML configuration root.

You can instead copy `scripts/config/runtime_paths.env.example` and source
`scripts/set_runtime_paths.sh`.

## Run an experiment

Use the environment associated with the model:

```bash
# PCA, HVG, and mock baselines
pixi run -e default run-exp exp/pca/n50/brca_cell_type.yaml

# Geneformer
pixi run -e geneformer run-exp \
  exp/geneformer/V1-10M-i2048/brca_cell_type.yaml

# scVI
pixi run -e scvi run-exp exp/scvi/default/brca_cell_type.yaml
```

Other model environments include `scgpt`, `scimilarity`, `cellplm`, `state`,
`scf`, `nicheformer`, `scconcept`, and `scbert`. The shell helpers under
`run/` show the environment used by each bundled model.

Useful runtime limits:

```bash
pixi run -e default run-exp EXPERIMENT.yaml \
  --seed 42 \
  --max-cells 10000 \
  --max-cells-stratify donor_id
```

Set `SCFM_VALIDATE_EXP=1` to validate model/data constraints before the
evaluation starts.

## Provision data and weights

Configuration templates are available under `scripts/config/`:

```bash
cp scripts/config/data_download.env.example scripts/config/data_download.env
cp scripts/config/model_weights.env.example scripts/config/model_weights.env

# Edit both files, then run:
pixi run download-data
pixi run download-models
```

The download scripts support the repository's current GCS, HTTP, rsync, Git
LFS, and Hugging Face workflows. You can also place files manually under
`SCFM_DATA_PATH` and `SCFM_MODELS_PATH`.

## Find the results

The command prints `save_dir` when the run starts. A successful run contains:

- `results.json`: validated `scfm_eval.results` v1.1.0 record
- `resolved_config.yaml`: exact configuration used
- `run_summary.json`: paths, identifiers, and timestamps
- `metrics.json`: compact metric summary
- `embedding_metrics.csv`: embedding metrics when enabled
- `data.h5ad`: generated embedding matrix and observations
- classifier CSV files and plots when classification is enabled

The output root also contains `metrics_runs.csv`, with one summary row per
completed run.

## External configurations

An absolute experiment path is supported:

```bash
pixi run -e default run-exp /path/to/my_experiment.yaml
```

Includes beginning with `./` or `../` resolve relative to the including file.
Other relative includes resolve under the bundled YAML root or
`SCFM_PARAMS_PATH`. External runs are stored below
`$SCFM_OUTPUT_PATH/external/<config-hash>/`.

## GPU machines

For an interactive GPU VM, install the NVIDIA driver, Git, and Git LFS, then
run:

```bash
bash scripts/bootstrap_vm.sh
pixi run verify-gpu
```

Use the repository `Dockerfile` when you need a containerized CUDA runtime.
The host must have NVIDIA Container Toolkit installed for
`docker run --gpus all`.
