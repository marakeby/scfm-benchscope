#!/usr/bin/env bash
# (1) Install Pixi envs from pixi.lock + Python packages not fully covered by PyPI wheels
#     (Geneformer editable install, optional scGPT source, etc.).
#
# Usage:
#   bash scripts/install_packages.sh              # pixi install + Geneformer package
#   bash scripts/install_packages.sh --pixi-only
#   bash scripts/install_packages.sh --geneformer-only
#
# Large weights: use scripts/download_model_weights.sh after this (or git lfs pull there).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PIXI_BIN="${PIXI_HOME:-$HOME/.pixi}/bin"
export PATH="$PIXI_BIN:$PATH"

DO_PIXI=1
DO_GENEFORMER=1
while [[ $# -gt 0 ]]; do
  case $1 in
    --pixi-only) DO_GENEFORMER=0 ;;
    --geneformer-only) DO_PIXI=0 ;;
    -h|--help)
      echo "Usage: $0 [--pixi-only] [--geneformer-only]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
  shift
done

if [[ $DO_PIXI -eq 1 ]]; then
  if ! command -v pixi >/dev/null 2>&1; then
    echo "Installing Pixi …"
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="${PIXI_HOME:-$HOME/.pixi}/bin:$PATH"
  fi
  echo "pixi install (from lockfile) …"
  pixi install
fi

GENEFORMER_SRC="${GENEFORMER_SRC:-$ROOT/third_party/Geneformer}"

if [[ $DO_GENEFORMER -eq 1 ]]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "error: git is required for Geneformer."
    exit 1
  fi
  if ! command -v git-lfs >/dev/null 2>&1; then
    echo "error: git-lfs is required (install OS package, then: git lfs install)."
    exit 1
  fi

  mkdir -p "$(dirname "$GENEFORMER_SRC")"
  if [[ ! -d "$GENEFORMER_SRC/.git" ]]; then
    echo "Cloning Geneformer (code first; use download_model_weights.sh for full LFS assets) …"
    # Smaller initial clone; run download_model_weights.sh with PULL_GENEFORMER_LFS=1 for weights.
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/ctheodoris/Geneformer "$GENEFORMER_SRC"
  else
    echo "Geneformer repo exists at $GENEFORMER_SRC"
  fi

  if ! command -v pixi >/dev/null 2>&1; then
    echo "error: pixi required to install into the geneformer env."
    exit 1
  fi

  echo "pip install -e Geneformer into Pixi env 'geneformer' …"
  pixi run -e geneformer pip install -U pip setuptools wheel
  pixi run -e geneformer pip install -e "$GENEFORMER_SRC"
  echo "Geneformer package OK. If models fail to load, run: bash scripts/download_model_weights.sh"
fi

echo "install_packages.sh done."
