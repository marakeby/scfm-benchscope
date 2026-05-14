#!/usr/bin/env bash
# Clone biomap-research/scFoundation, add a minimal pyproject.toml under model/, and
# pip install -e so `from pretrainmodels.select_model import select_model` works.
#
# Usage (from repo root):
#   bash scripts/install_scfoundation_model.sh
# With Pixi (recommended — uses env `scf`):
#   pixi run install-scfoundation
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCF_SRC="${SCF_SRC:-$ROOT/third_party/scFoundation}"
TEMPLATE="$ROOT/scripts/templates/scfoundation_model_pyproject.toml"

if [[ ! -f "$TEMPLATE" ]]; then
  echo "error: missing template: $TEMPLATE"
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "error: git is required."
  exit 1
fi

mkdir -p "$(dirname "$SCF_SRC")"
if [[ ! -d "$SCF_SRC/.git" ]]; then
  echo "Cloning scFoundation → $SCF_SRC …"
  git clone https://github.com/biomap-research/scFoundation.git "$SCF_SRC"
else
  echo "Using existing repo at $SCF_SRC"
fi

if [[ ! -d "$SCF_SRC/model/pretrainmodels" ]]; then
  echo "error: $SCF_SRC/model/pretrainmodels not found (unexpected clone layout)."
  exit 1
fi

cp "$TEMPLATE" "$SCF_SRC/model/pyproject.toml"
echo "Wrote $SCF_SRC/model/pyproject.toml (from template)."

export PATH="${PIXI_HOME:-$HOME/.pixi}/bin:$PATH"

if command -v pixi >/dev/null 2>&1; then
  echo "pip install -e (Pixi env scf) …"
  pixi run -e scf pip install -U pip setuptools wheel
  pixi run -e scf pip install -e "$SCF_SRC/model"
else
  echo "Pixi not found; install with: pip install -e \"$SCF_SRC/model\""
  echo "Or install Pixi and re-run this script."
  pip install -U pip setuptools wheel
  pip install -e "$SCF_SRC/model"
fi

echo "scfoundation-model install done."
