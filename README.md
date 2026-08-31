<p align="center">
  <img src="docs/assets/scfm-benchscope-logo.png" alt="scFM BenchScope" width="420">
</p>

# scFM BenchScope

Evaluate single-cell foundation model embeddings and downstream classifiers
using reproducible datasets, tasks, and metrics.

See [Installation](docs/installation.md) for pip and Docker alternatives.

## Quick start

Pixi is the recommended way to run the repository because model dependencies
are isolated in separate environments.

```bash
pixi install --frozen

export SCFM_DATA_PATH=/path/to/datasets
export SCFM_MODELS_PATH=/path/to/model-weights
export SCFM_OUTPUT_PATH="$PWD/output"

# scGPT; expects the BRCA dataset and weights referenced by this config.
pixi run -e scgpt run-exp exp/scgpt/cancer-i2048/brca_cell_type.yaml
```

Experiment paths such as `exp/scgpt/...` refer to the bundled configuration tree
under `src/scfm_cancer_eval/yaml/`.

For pretrained models, select the matching Pixi environment and experiment:

```bash
pixi run -e geneformer run-exp \
  exp/geneformer/V1-10M-i2048/brca_cell_type.yaml
```

Model code, weights, and datasets must be available at the paths expected by
the selected configuration. See [Running evaluations](docs/running-evaluations.md)
for setup scripts and environment details.

## Results

The runner prints the run directory. Its primary files are:

- `results.json`: validated metrics, provenance, inputs, and artifacts
- `resolved_config.yaml`: the exact merged configuration
- `metrics.json`: compact embedding and classification metrics
- `data.h5ad`: generated embeddings

Completed runs also append a summary row to
`$SCFM_OUTPUT_PATH/metrics_runs.csv`.

Create an offline report across completed runs:

```bash
scfm-eval report "$SCFM_OUTPUT_PATH"
```

Open `$SCFM_OUTPUT_PATH/report/report.html` to filter runs, compare metrics, and
download stable JSON or CSV exports.

## Evaluate your own model

An installed model can be evaluated without adding it to the repository. Pass
an adapter object that provides `output_key` and `fit_transform(loader)`:

```python
from scfm_cancer_eval import evaluate

result = evaluate(
    model=MyModelAdapter(checkpoint="/models/my-model"),
    dataset="datasets/brca/cell_type/dataset_full.yaml",
    task="embedding",
    output_dir="./output",
)

print(result.results_path)
print(result.status)
```

See [Adding a model](docs/adding-a-model.md) to plug in your own adapter.

## More documentation

[docs/README.md](docs/README.md) is the index. Short version:

- [Install](docs/installation.md)
- [Run an experiment](docs/running-evaluations.md)
- [Compare results](docs/reporting.md)
- [Add your model](docs/adding-a-model.md)
- [Write experiment YAML](docs/experiment-configuration.md)
- [Find models](docs/discovery-agent.md)
- [Onboard a model with the integration agent](docs/integration/README.md)

