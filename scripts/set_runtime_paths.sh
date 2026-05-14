#!/usr/bin/env bash
# Export SCFM_DATA_PATH, SCFM_OUTPUT_PATH, SCFM_MODELS_PATH for setup_path.py.
# Optionally set SCFM_PARAMS_PATH (custom YAML root); see README "Custom models".
#
# Usage (from repo root or elsewhere):
#   source scripts/set_runtime_paths.sh
#
# Optional: copy scripts/config/runtime_paths.env.example to runtime_paths.env and edit.
# Override config path:
#   SCFM_RUNTIME_CONFIG=/path/to/runtime_paths.env source scripts/set_runtime_paths.sh
#
# Pre-set variables are left unchanged. Values from the config file apply to any still unset.
# If no config file exists, defaults match the previous repo layout under $HOME/mnt/.

if [[ "${BASH_SOURCE[0]:-$0}" == "$0" ]]; then
  echo "Run:  source ${BASH_SOURCE[0]}" >&2
  echo "(do not execute this script directly — it runs in a subshell and would not affect your shell)" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${SCFM_RUNTIME_CONFIG:-$ROOT/scripts/config/runtime_paths.env}"

if [[ -f "$CONFIG" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$CONFIG"
  set +a
fi

export SCFM_DATA_PATH="${SCFM_DATA_PATH:-$HOME/mnt/DATA}"
export SCFM_OUTPUT_PATH="${SCFM_OUTPUT_PATH:-$HOME/mnt/__output_v6}"
export SCFM_MODELS_PATH="${SCFM_MODELS_PATH:-$HOME/mnt/MODELS}"
