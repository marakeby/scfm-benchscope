# Adding a model

You only need a Python adapter that returns embeddings. Cloning the model
repo, creating an environment, and downloading weights are separate steps.

## Smallest option: pass an adapter object

The adapter needs:

- `output_key` — the `AnnData.obsm` key for the embeddings
- `fit_transform(loader)` — one embedding row per input cell

```python
import numpy as np


class MyModelAdapter:
    output_key = "X_my_model"

    def __init__(self, checkpoint):
        self.model = load_my_model(checkpoint)

    def fit_transform(self, loader):
        embeddings = self.model.encode(loader.adata.X)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        loader.adata.obsm[self.output_key] = embeddings
        return embeddings
```

```python
from scfm_cancer_eval import evaluate

result = evaluate(
    model=MyModelAdapter("/models/my-model"),
    dataset="datasets/brca/cell_type/dataset_full.yaml",
    task="embedding",
    output_dir="./output",
)
```

No registry entry or model YAML is required.

## Serializable option: import path

Put the adapter in an installed package and point at it:

```python
from scfm_cancer_eval import EvaluationModelConfig, evaluate

model = EvaluationModelConfig(
    model_id="my-model",
    adapter="my_lab.scfm.MyModelExtractor",
    output_key="X_my_model",
    params={"checkpoint": "/models/my-model"},
)

result = evaluate(
    model=model,
    dataset="datasets/brca/cell_type/dataset_full.yaml",
    output_dir="./output",
)
```

The easiest way to match the constructor is to subclass
`scfm_cancer_eval.features.extractor.EmbeddingExtractor`.

## YAML option

```yaml
embedding:
  method: my-model
  module: my_lab.scfm
  class: MyModelExtractor
  output_key: X_my_model
  viz: false
  eval: true
  params:
    checkpoint: /models/my-model
```

The module must be importable in the Pixi environment you run. Short bundled
paths such as `features.pca_extractor` are rewritten to `scfm_cancer_eval`.
Other module paths are imported as written.

## Checkpoints

A direct adapter can take any absolute path. Extractors that subclass
`EmbeddingExtractor` can list `MODELS_PATH_KEYS`; relative values for those
keys resolve under `SCFM_MODELS_PATH`.

Keep large weights outside the package. Record a model version or checksum
in your own notes when you compare runs over time.

## Conflicting dependencies

If the model’s packages clash with the core framework, put them in a
separate Pixi environment. See `pixi.toml` for examples. The adapter API
does not change.

## Compare with bundled models

Use the same dataset and task for each model. `results.json` stores the
resolved inputs and metrics. `metrics_runs.csv` under `SCFM_OUTPUT_PATH`
is the flat run index.
