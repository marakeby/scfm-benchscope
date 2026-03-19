#!/usr/bin/env bash
# One-shot setup: download datasets, clone model repos, and fetch checkpoints.
# Reads dataset_paths.yaml and model_paths.yaml from this setup/ directory.
#
# If uv is installed and pyproject.toml exists at repo root, runs `uv sync` then
# `uv run python ...` so you need no pre-created env. Otherwise uses system python3.
#
# Default dirs (if DATASETS_PATH/MODEL_CHECKPOINTS not set in .env): sibling of
# scFM_eval, so the three sit at the same level:
#   parent_dir/
#     scFM_eval/
#     datasets/           <- DATASETS_PATH
#     model_checkpoints/  <- MODEL_CHECKPOINTS
# Override in .env or export before running.
#
# Usage:
#   cd /path/to/scFM_eval && bash setup/setup_download_all.sh
#   bash setup/setup_download_all.sh --datasets-only
#   bash setup/setup_download_all.sh --repos-only
#   bash setup/setup_download_all.sh --checkpoints-only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Load .env so DATASETS_PATH, MODEL_CHECKPOINTS, REPOS_DIR are available
if [[ -f .env ]]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi

# Defaults: sibling dirs of scFM_eval (so scFM_eval, datasets, model_checkpoints sit at same level)
PARENT_DIR="$(dirname "$REPO_ROOT")"
export DATASETS_PATH="${DATASETS_PATH:-$PARENT_DIR/datasets}"
export MODEL_CHECKPOINTS="${MODEL_CHECKPOINTS:-$PARENT_DIR/model_checkpoints}"

# Use uv if available (minimal env from pyproject.toml); else system python
if command -v uv &>/dev/null && [[ -f "$REPO_ROOT/pyproject.toml" ]]; then
  uv sync --no-dev
  exec uv run python "$SCRIPT_DIR/setup_download_all.py" "$@"
else
  exec python3 "$SCRIPT_DIR/setup_download_all.py" "$@"
fi
