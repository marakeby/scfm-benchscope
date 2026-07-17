# scFM_eval

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

# Weight-free baseline; expects the BRCA dataset referenced by this config.
pixi run -e default run-exp exp/pca/n50/brca_cell_type.yaml
```

Experiment paths such as `exp/pca/...` refer to the bundled configuration tree
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

See [Adding a model](docs/adding-a-model.md) for the adapter contract,
importable model configurations, custom YAML, and checkpoint handling.

## More documentation

- [Installation](docs/installation.md)
- [Running evaluations](docs/running-evaluations.md)
- [Comparing and visualizing results](docs/reporting.md)
- [Adding a model](docs/adding-a-model.md)
- [Experiment configuration](docs/experiment-configuration.md)

