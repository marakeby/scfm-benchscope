#!/usr/bin/env bash
# Small Geneformer finetune hyperparameter grid search — pixi env: geneformer
#
# Usage (from repo root):
#   bash run/grid_search_gf_finetune.sh
#   bash run/grid_search_gf_finetune.sh grids/gf_finetune_v2_small.yaml
#   bash run/grid_search_gf_finetune.sh grids/gf_finetune_small.yaml --dry-run
#   bash run/grid_search_gf_finetune.sh grids/gf_finetune_small.yaml --max-trials 2
set -euo pipefail
cd "$(dirname "$0")/.."

GRID="${1:-grids/gf_finetune_small.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

pixi run -e geneformer python -m scfm_cancer_eval.run.grid_search_finetune \
  "${GRID}" \
  "$@"
