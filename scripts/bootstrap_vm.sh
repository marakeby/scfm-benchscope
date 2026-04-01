#!/usr/bin/env bash
# One-shot setup for a Linux dev machine (e.g. GCP GPU VM): Pixi + package installs (Geneformer, …).
# For weights and data, run scripts/download_model_weights.sh and scripts/download_data.sh separately.
#
# Usage (from repo root):
#   bash scripts/bootstrap_vm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash "$ROOT/scripts/install_packages.sh"

echo ""
echo "Smoke checks:"
pixi run check-imports || true
pixi run verify-gpu || true

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
else
  echo "nvidia-smi not found (OK for CPU-only nodes)."
fi

echo ""
echo "Next: configure scripts/config/model_weights.env + data_download.env, then:"
echo "  bash scripts/download_model_weights.sh"
echo "  bash scripts/download_data.sh"
echo "Adjust setup_path.py (DATA_PATH, OUTPUT_PATH), then e.g."
echo "  pixi run -e default run-exp yaml/examples/mock_subtype_eval.yaml"
