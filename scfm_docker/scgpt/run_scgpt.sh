#!/usr/bin/env bash
set -euo pipefail

IMAGE="scgpt:cu128"
docker build -t "$IMAGE" .

docker run --rm -it \
  --gpus all \
  --shm-size=16g \
  -v "$PWD":/workspace \
  "$IMAGE"