# Installation

Choose one installation path. Pixi is recommended for running bundled
pretrained models; pip is convenient for library integration; Docker provides
the most isolated runtime.

## Pip

Install the package from a repository checkout:

```bash
python -m pip install .
scfm-eval --help
```

For development:

```bash
python -m pip install -e .
python -m scfm_cancer_eval --help
```

The wheel contains the built-in YAML configurations and result JSON Schema.
Pip installs the core scientific dependencies, but it does not solve the
incompatible dependency stacks of every pretrained model. Use Pixi or a
model-specific Docker image for those models.

To build a wheel without installing it:

```bash
python -m pip wheel . --no-deps --wheel-dir dist
```

## Pixi

Pixi uses the committed `pixi.lock` and separate environments for incompatible
models:

```bash
pixi install --frozen
pixi run -e default check-imports
pixi run -e default test
```

Run a baseline:

```bash
pixi run -e default run-exp exp/pca/n50/brca_cell_type.yaml
```

Run a pretrained model in its environment:

```bash
pixi run -e geneformer run-exp \
  exp/geneformer/V1-10M-i2048/brca_cell_type.yaml
```

The workspace currently targets Linux x86-64. Model weights and datasets are
not stored in the environments.

## Docker

Build the core image:

```bash
docker build -t scfm-eval:default .
docker run --rm scfm-eval:default --help
```

Build a specific locked Pixi environment:

```bash
docker build \
  --build-arg SCFM_PIXI_ENV=geneformer \
  -t scfm-eval:geneformer .
```

Run an evaluation with read-only data and model mounts:

```bash
docker run --rm --gpus all \
  -v /path/to/data:/data:ro \
  -v /path/to/models:/models:ro \
  -v "$PWD/output":/output \
  scfm-eval:geneformer \
  exp/geneformer/V1-10M-i2048/brca_cell_type.yaml
```

The image uses:

- `/data` as `SCFM_DATA_PATH`
- `/models` as `SCFM_MODELS_PATH`
- `/output` as `SCFM_OUTPUT_PATH`

The default command is `--help`, so starting the image without an experiment
is safe. NVIDIA Container Toolkit is required only when using `--gpus`.

## Verify an installation

For source or Pixi installs:

```bash
pixi run -e default check-imports
pixi run -e default validate-yaml-syntax
pixi run -e default test
```

For Docker:

```bash
docker run --rm --entrypoint pixi \
  scfm-eval:default run -e default check-imports
```
