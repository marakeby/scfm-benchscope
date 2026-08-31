# Experiment configuration

An experiment is a YAML file built from reusable dataset, model, and task
pieces.

## What the runner needs

After all includes are merged, the file must have:

- `dataset`
- `qc`
- `preprocessing`
- `embedding`
- `classification`

`run_id`, `hvg`, and `task` are optional. A stable `run_id` makes comparisons
easier.

## Example

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

If `dataset`, `model`, or `classification` is a string or list, those values
are include paths. If it is a mapping, it is the section itself.

Older include keys still work: `datasets`, `models`, `classifications`,
`bases`, and `defaults`.

## How includes merge

Order:

1. dataset includes
2. model includes
3. classification includes
4. `bases` or `defaults`
5. values in the current file

Later values win. Nested maps are merged. Lists and single values are
replaced.

- Absolute paths are used as-is
- Paths starting with `./` or `../` are relative to the file that includes them
- Other paths resolve under `SCFM_PARAMS_PATH`, or
  `src/scfm_cancer_eval/yaml/` if that variable is unset

Cycles are rejected.

## Dataset and model paths

Relative `dataset.path` values resolve under `SCFM_DATA_PATH`. The loader
maps `label_key` and `batch_key` to the `label` and `batch` columns used by
evaluators.

If a classifier does not set `params.label_map`, it inherits
`dataset.label_map` from the dataset or task fragment.

Relative checkpoint parameters listed in an extractor’s `MODELS_PATH_KEYS`
resolve under `SCFM_MODELS_PATH`. Other string parameters are passed through
unchanged.

## Validate YAML

Check one experiment before it runs:

```bash
SCFM_VALIDATE_EXP=1 pixi run -e default run-exp EXPERIMENT.yaml
```

Check the bundled collection without running models:

```bash
pixi run -e default python scripts/validate_all_yaml.py
```

Every run writes the fully merged config as `resolved_config.yaml`.
