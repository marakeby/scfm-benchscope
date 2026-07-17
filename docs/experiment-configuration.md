# Experiment configuration

Experiments are YAML mappings assembled from reusable dataset, model, and task
fragments.

## Required sections

After composition, the runner expects:

- `dataset`
- `qc`
- `preprocessing`
- `embedding`
- `classification`

`run_id`, `hvg`, and `task` are optional, although a stable `run_id` is useful
for comparing runs.

## Composed experiment

```yaml
run_id: my_model_brca_cell_type

dataset:
  - datasets/brca/cell_type/dataset_full.yaml

model:
  - models/my_model/default.yaml

classification:
  skip: true

embedding:
  viz: false
  eval: true
```

When `dataset`, `model`, or `classification` contains a string or list, its
values are include paths. When it contains a mapping, it is the configuration
section itself.

Legacy `datasets`, `models`, `classifications`, `bases`, and `defaults`
includes remain supported.

## Merge behavior

Fragments are merged in this order:

1. dataset includes
2. model includes
3. classification includes
4. `bases` or `defaults`
5. values in the current file

Mappings are merged recursively. Lists and scalar values are replaced by the
later value.

Absolute includes are used unchanged. Paths beginning with `./` or `../`
resolve relative to the including YAML file. Other paths resolve under
`SCFM_PARAMS_PATH`, or the bundled `src/scfm_cancer_eval/yaml/` tree when that
variable is unset.

The loader rejects cyclic includes.

## Dataset paths

Relative `dataset.path` values resolve under `SCFM_DATA_PATH`. The data loader
maps the configured `label_key` and `batch_key` to the canonical `label` and
`batch` observation columns used by evaluators.

Many dataset fragments store class values under `dataset.label_map`. When a
classifier does not provide its own `params.label_map`, the runner inherits
the dataset or task mapping.

## Model paths

Relative checkpoint parameters declared by an extractor's
`MODELS_PATH_KEYS` resolve under `SCFM_MODELS_PATH`. Other string parameters
are passed to the adapter unchanged.

## Validation

Enable model/data compatibility checks for a run:

```bash
SCFM_VALIDATE_EXP=1 pixi run -e default run-exp EXPERIMENT.yaml
```

Validate the bundled configuration collection without running models:

```bash
pixi run -e default python scripts/validate_all_yaml.py
```

Every run stores its fully merged configuration as `resolved_config.yaml`.
