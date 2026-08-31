# Installation

Pick one path:

- **Pixi** — best for the bundled pretrained models
- **pip** — best if you only need the Python library
- **Docker** — best for an isolated runtime

## Pixi

From the repository root:

```bash
pixi install --frozen
pixi run -e default check-imports
pixi run -e default test
```

Run a baseline:

```bash
pixi run -e default run-exp exp/pca/n50/brca_cell_type.yaml
```

Run a pretrained model in its own environment:

```bash
pixi run -e geneformer run-exp \
  exp/geneformer/V1-10M-i2048/brca_cell_type.yaml
```

Pixi uses the committed `pixi.lock`. Weights and datasets are not inside the
environments. This workspace targets Linux x86-64.

## Pip

```bash
python -m pip install .
scfm-eval --help
```

For local development:

```bash
python -m pip install -e .
python -m scfm_cancer_eval --help
```

Pip installs the core package and the built-in YAML configs. It does not
install every pretrained model's dependencies. Use Pixi or a model-specific
Docker image for those.

Build a wheel without installing:

```bash
python -m pip wheel . --no-deps --wheel-dir dist
```

## Docker

Core image:

```bash
docker build -t scfm-eval:default .
docker run --rm scfm-eval:default --help
```

One locked Pixi environment:

```bash
docker build \
  --build-arg SCFM_PIXI_ENV=geneformer \
  -t scfm-eval:geneformer .
```

Run with data and weights mounted read-only:

```bash
docker run --rm --gpus all \
  -v /path/to/data:/data:ro \
  -v /path/to/models:/models:ro \
  -v "$PWD/output":/output \
  scfm-eval:geneformer \
  exp/geneformer/V1-10M-i2048/brca_cell_type.yaml
```

Inside the image:

- `/data` is `SCFM_DATA_PATH`
- `/models` is `SCFM_MODELS_PATH`
- `/output` is `SCFM_OUTPUT_PATH`

The default command is `--help`, so starting the image with no experiment is
safe. `--gpus` needs NVIDIA Container Toolkit.

## Check that it works

Pixi or a source checkout:

```bash
pixi run -e default check-imports
pixi run -e default validate-yaml-syntax
pixi run -e default test
```

Docker:

```bash
docker run --rm --entrypoint pixi \
  scfm-eval:default run -e default check-imports
```
