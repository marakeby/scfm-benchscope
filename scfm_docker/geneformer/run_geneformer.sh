#!/usr/bin/env bash
set -euo pipefail

IMAGE="geneformer:cu128"
docker build -t "$IMAGE" .

# Use a local cache for Hugging Face so you don't re-download models each time
mkdir -p "$HOME/.cache/huggingface"

docker run --rm -it \
  --gpus all \
  --shm-size=16g \
  -v "$PWD":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  "$IMAGE"