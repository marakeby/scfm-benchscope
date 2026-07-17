# Adding a model

The evaluation framework only requires an installed model adapter. Repository
cloning, environment creation, and weight downloading are separate concerns.

## Direct Python adapter

The smallest adapter provides:

- `output_key`: the `AnnData.obsm` key for its embeddings
- `fit_transform(loader)`: returns one embedding row per input cell

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

Pass the object directly to the library:

```python
from scfm_cancer_eval import evaluate

result = evaluate(
    model=MyModelAdapter("/models/my-model"),
    dataset="datasets/brca/cell_type/dataset_full.yaml",
    task="embedding",
    output_dir="./output",
)
```

This path does not require a registry entry or model YAML.

## Importable adapter configuration

For a serializable evaluation request, put the adapter in an installed Python
package and use its full import path:

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

Importable adapters are constructed with the existing embedding configuration
mapping. Subclassing
`scfm_cancer_eval.features.extractor.EmbeddingExtractor` is the easiest way to
follow that constructor contract.

## Existing experiment YAML

Adapters can also be selected in YAML:

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

The module must be importable in the selected Pixi environment. Bundled short
paths such as `features.pca_extractor` are rewritten to the
`scfm_cancer_eval` package; external module paths are imported unchanged.

## Checkpoints

Direct adapters may use any absolute checkpoint path. Extractors that subclass
`EmbeddingExtractor` can declare `MODELS_PATH_KEYS`; relative values for those
parameter names are then resolved under `SCFM_MODELS_PATH`.

Keep large weights outside the Python package and experiment repository.
Record an immutable model version or checksum in your own run metadata when
comparing results over time.

## Model-specific dependencies

Install the model and adapter in an isolated Pixi environment when their
dependencies conflict with the core framework. Existing environments in
`pixi.toml` provide examples. The adapter API remains the same regardless of
how the environment was created.

## Comparing with bundled models

Use the same dataset and task configuration for each model. `results.json`
records the resolved inputs and metrics, while `metrics_runs.csv` provides a
flat run index under `SCFM_OUTPUT_PATH`.
