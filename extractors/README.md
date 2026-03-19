# Per-Model Extractor Environments

Each model runs in its own isolated environment to avoid dependency conflicts.

## Architecture

```
extractors/
├── stack/
│   ├── env/              # Virtual environment
│   ├── extract.py        # Standalone extraction script
│   └── requirements.txt  # Model-specific dependencies
├── scgpt/
│   ├── env/
│   ├── extract.py
│   └── requirements.txt
└── ...
```

## Usage

Each extractor can be called standalone:

```bash
# Activate model-specific environment
source extractors/stack/env/bin/activate

# Run extraction
python extractors/stack/extract.py \
    --input /path/to/data.h5ad \
    --output /path/to/embeddings.npy \
    --checkpoint /path/to/model.ckpt
```

Or via the dispatcher:

```bash
python run_extractor.py \
    --model stack \
    --input data.h5ad \
    --output embeddings.npy \
    --config config.yaml
```

## Environment Setup

Each model is installed directly from its local GitHub repo using editable installs.
The repo's own `setup.py` or `pyproject.toml` defines the dependencies.

```bash
# Setup all environments
cd extractors/
./setup_envs.sh

# Or setup a specific model
./setup_envs.sh stack
./setup_envs.sh scgpt
./setup_envs.sh geneformer
```

The setup script:
1. Creates a fresh virtual environment
2. Installs the model package with `pip install -e /path/to/repo`
3. The model's dependencies are automatically installed

## Why Separate Environments?

| Model | Conflicting Dependency |
|-------|----------------------|
| scGPT | torchtext ABI, older torch |
| Geneformer | transformers 4.x (huggingface-hub<1.0) |
| scConcept | huggingface-hub>=1.0.1 |
| STATE | scipy>=1.15 |
| AIDO.Cell | transformers (huggingface-hub conflict) |

## Common Interface

All extractors implement the same interface:

```python
# Input: AnnData file path
# Output: numpy array saved to file

python extract.py \
    --input input.h5ad \
    --output embeddings.npy \
    --label_key cell_type \
    --batch_key batch \
    [--model_specific_args ...]
```

Output format:
- `embeddings.npy`: Shape (n_cells, embedding_dim)
- `metadata.json`: Model info, parameters used
