# Running evaluations

Work from the repository root.

## Install environments

```bash
pixi install
```

Some models need extra source packages:

```bash
pixi run install-packages
```

That command does a frozen Pixi install. Geneformer is already pinned in the
`geneformer` environment. Use
`scripts/install_packages.sh --with-geneformer-source` only if you are
developing against an editable upstream checkout.

scFoundation has its own step:

```bash
pixi run install-scfoundation
```

## Set data, weight, and output paths

Set these before you start Python or run an experiment:

```bash
export SCFM_DATA_PATH=/mnt/data
export SCFM_MODELS_PATH=/mnt/models
export SCFM_OUTPUT_PATH=/mnt/scfm-results
```

| Variable | Used for |
| --- | --- |
| `SCFM_DATA_PATH` | Relative `dataset.path` values |
| `SCFM_MODELS_PATH` | Relative checkpoint paths |
| `SCFM_OUTPUT_PATH` | Run directories and `metrics_runs.csv` |
| `SCFM_PARAMS_PATH` | Optional replacement for the bundled YAML root |

You can copy `scripts/config/runtime_paths.env.example` and source
`scripts/set_runtime_paths.sh` instead.

## Download data and weights

```bash
cp scripts/config/data_download.env.example scripts/config/data_download.env
cp scripts/config/model_weights.env.example scripts/config/model_weights.env

# Edit both files, then:
pixi run download-data
pixi run download-models
```

The scripts cover GCS, HTTP, rsync, Git LFS, and Hugging Face. You can also
copy files yourself into `SCFM_DATA_PATH` and `SCFM_MODELS_PATH`.

## Run an experiment

Use the Pixi environment that matches the model:

```bash
# PCA, HVG, and mock baselines
pixi run -e default run-exp exp/pca/n50/brca_cell_type.yaml

# Geneformer
pixi run -e geneformer run-exp \
  exp/geneformer/V1-10M-i2048/brca_cell_type.yaml

# scVI
pixi run -e scvi run-exp exp/scvi/default/brca_cell_type.yaml
```

Other environments: `scgpt`, `scimilarity`, `cellplm`, `state`, `scf`,
`nicheformer`, `scconcept`, `scbert`. The helpers under `run/` show which
environment each bundled model uses.

Optional limits:

```bash
pixi run -e default run-exp EXPERIMENT.yaml \
  --seed 42 \
  --max-cells 10000 \
  --max-cells-stratify donor_id
```

Set `SCFM_VALIDATE_EXP=1` to check model and data constraints before the run
starts.

An absolute experiment path works:

```bash
pixi run -e default run-exp /path/to/my_experiment.yaml
```

Includes that start with `./` or `../` are relative to the file that includes
them. Other relative includes resolve under the bundled YAML tree or
`SCFM_PARAMS_PATH`. External runs land in
`$SCFM_OUTPUT_PATH/external/<config-hash>/`.

## Find the results

The command prints `save_dir` when the run starts. A finished run usually has:

- `results.json` — metrics and provenance
- `resolved_config.yaml` — the exact config that ran
- `run_summary.json` — paths and timestamps
- `metrics.json` — compact metric summary
- `embedding_metrics.csv` — embedding metrics, if enabled
- `data.h5ad` — embeddings and observations
- classifier CSVs and plots, if classification is enabled

The output root also has `metrics_runs.csv`, one row per finished run.

Build a browser report:

```bash
scfm-eval report "$SCFM_OUTPUT_PATH"
```

That writes `report/report.html`, `report/comparison.json`, and
`report/comparison.csv`. See [Compare results](reporting.md) for more options.

## GPU machines

On an interactive GPU VM, install the NVIDIA driver, Git, and Git LFS, then:

```bash
bash scripts/bootstrap_vm.sh
pixi run verify-gpu
```

Or use Docker:

```bash
docker build -t scfm-eval:default .
docker run --rm \
  -v /path/to/data:/data:ro \
  -v /path/to/models:/models:ro \
  -v "$PWD/output":/output \
  scfm-eval:default exp/pca/n50/brca_cell_type.yaml
```

Add `--build-arg SCFM_PIXI_ENV=<model-env>` for a pretrained model stack.
`--gpus all` needs NVIDIA Container Toolkit on the host. See
[Installation](installation.md) for more Docker examples.
