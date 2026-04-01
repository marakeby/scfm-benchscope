#!/usr/bin/env bash
# (2) Download or refresh large model checkpoints (Git LFS, Hugging Face Hub, etc.).
#     Complements install_packages.sh (Python package installs).
#
# Config:
#   cp scripts/config/model_weights.env.example scripts/config/model_weights.env
#   $EDITOR scripts/config/model_weights.env
#   bash scripts/download_model_weights.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${MODEL_WEIGHTS_CONFIG:-$ROOT/scripts/config/model_weights.env}"
if [[ -f "$CONFIG" ]]; then
  echo "Loading $CONFIG"
  set -a
  # shellcheck source=/dev/null
  source "$CONFIG"
  set +a
else
  echo "No $CONFIG — using defaults. Copy scripts/config/model_weights.env.example to model_weights.env to customize."
fi

MODEL_ROOT="${MODEL_ROOT:-$HOME/mnt/MODELS}"
GENEFORMER_SRC="${GENEFORMER_SRC:-$ROOT/third_party/Geneformer}"
mkdir -p "$MODEL_ROOT"

PIXI_BIN="${PIXI_HOME:-$HOME/.pixi}/bin"
export PATH="$PIXI_BIN:$PATH"

hf_snapshot() {
  local repo_id="$1"
  local dest="$2"
  echo "HF snapshot_download: $repo_id -> $dest"
  export HF_SNAPSHOT_REPO="$repo_id"
  export HF_SNAPSHOT_DEST="$dest"
  if command -v pixi >/dev/null 2>&1; then
    pixi run -e default python -c "
import os
from pathlib import Path
from huggingface_hub import snapshot_download
repo = os.environ['HF_SNAPSHOT_REPO']
dest = Path(os.environ['HF_SNAPSHOT_DEST'])
dest.parent.mkdir(parents=True, exist_ok=True)
tok = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
snapshot_download(repo_id=repo, local_dir=str(dest), token=tok)
print('OK', dest)
"
  else
    python3 -c "
import os
from pathlib import Path
from huggingface_hub import snapshot_download
repo = os.environ['HF_SNAPSHOT_REPO']
dest = Path(os.environ['HF_SNAPSHOT_DEST'])
dest.parent.mkdir(parents=True, exist_ok=True)
tok = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
snapshot_download(repo_id=repo, local_dir=str(dest), token=tok)
print('OK', dest)
"
  fi
}

if [[ "${PULL_GENEFORMER_LFS:-0}" == "1" ]] && [[ -d "$GENEFORMER_SRC/.git" ]]; then
  echo "git lfs pull in $GENEFORMER_SRC …"
  (cd "$GENEFORMER_SRC" && git lfs install && git lfs pull)
fi

if [[ "${HF_DOWNLOAD_GENEFORMER_V2_104M:-0}" == "1" ]]; then
  REPO="${HF_REPO_GENEFORMER_V2_104M:-ctheodoris/Geneformer-V2-104M}"
  DEST="$MODEL_ROOT/${REPO##*/}"
  hf_snapshot "$REPO" "$DEST"
fi

if [[ "${HF_DOWNLOAD_GENEFORMER_DICTS:-0}" == "1" ]] && [[ -n "${HF_REPO_GENEFORMER_DICTS:-}" ]]; then
  DICT_DEST="${DICT_DEST:-$HOME/mnt/gf_dicts_v2}"
  mkdir -p "$DICT_DEST"
  hf_snapshot "${HF_REPO_GENEFORMER_DICTS}" "$DICT_DEST"
fi

echo "download_model_weights.sh done. Set MODEL_ROOT=$MODEL_ROOT"
