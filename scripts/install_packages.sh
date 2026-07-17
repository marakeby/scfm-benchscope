#!/usr/bin/env bash
# Install the locked Pixi environment. An optional pinned Geneformer source
# checkout is available for developers who need to edit upstream code.
#
# Usage:
#   bash scripts/install_packages.sh
#   bash scripts/install_packages.sh --with-geneformer-source
#   bash scripts/install_packages.sh --geneformer-only  # compatibility alias
#
# Large weights: use scripts/download_model_weights.sh after this (or git lfs pull there).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PIXI_BIN="${PIXI_HOME:-$HOME/.pixi}/bin"
export PATH="$PIXI_BIN:$PATH"

DO_PIXI=1
DO_GENEFORMER_SOURCE=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --pixi-only) DO_GENEFORMER_SOURCE=0 ;;
    --with-geneformer-source) DO_GENEFORMER_SOURCE=1 ;;
    --geneformer-only)
      DO_PIXI=0
      DO_GENEFORMER_SOURCE=1
      ;;
    -h|--help)
      echo "Usage: $0 [--pixi-only] [--with-geneformer-source] [--geneformer-only]"
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
  echo "pixi install --frozen …"
  pixi install --frozen
fi

GENEFORMER_SRC="${GENEFORMER_SRC:-$ROOT/third_party/Geneformer}"
GENEFORMER_REF="${GENEFORMER_REF:-fcd26c45fc30fba1989e586bdc46bc366dda8655}"

if [[ $DO_GENEFORMER_SOURCE -eq 1 ]]; then
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
    echo "Cloning pinned Geneformer source (weights remain separate) …"
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/ctheodoris/Geneformer "$GENEFORMER_SRC"
  else
    echo "Geneformer repo exists at $GENEFORMER_SRC"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$GENEFORMER_SRC" fetch origin "$GENEFORMER_REF"
  fi
  git -C "$GENEFORMER_SRC" checkout --detach "$GENEFORMER_REF"

  if ! command -v pixi >/dev/null 2>&1; then
    echo "error: pixi required to install into the geneformer env."
    exit 1
  fi

  echo "Installing pinned editable Geneformer source without changing locked dependencies …"
  pixi run -e geneformer pip install --no-deps -e "$GENEFORMER_SRC"
  echo "Geneformer package OK. If models fail to load, run: bash scripts/download_model_weights.sh"
fi

echo "install_packages.sh done."
