#!/usr/bin/env bash
# Run every evaluation script under run/ (all yaml/exp model × dataset jobs).
# Total runtime can be very large (GPU queues, many models).
#
# Usage (from repo root):
#   bash run/eval_all.sh
#
# Stop on first failure:
#   bash run/eval_all.sh --fail-fast
#
# Environment:
#   EVAL_CONTINUE_ON_ERROR=1  — default: continue after a failed script and report counts at the end.
#   With --fail-fast, exits on first non-zero status (equivalent to EVAL_CONTINUE_ON_ERROR=0).
#
# Repo root is derived from this file's path so it works when invoked as
#   bash scFM_eval/run/eval_all.sh
# from any working directory (relative $0 would otherwise resolve against cwd).

_THIS="${BASH_SOURCE[0]:-$0}"
[[ "$_THIS" != /* ]] && _THIS="${PWD}/${_THIS}"
_EVAL_ALL_DIR="$(cd "$(dirname "$_THIS")" && pwd)" || exit 1
_REPO_ROOT="$(cd "$_EVAL_ALL_DIR/.." && pwd)" || exit 1
cd "$_REPO_ROOT" || exit 1

FAIL_FAST=0
if [[ "${1:-}" == "--fail-fast" ]]; then
  FAIL_FAST=1
fi

# If set to 0 or empty, stop on first error when FAIL_FAST is 1; otherwise continue.
: "${EVAL_CONTINUE_ON_ERROR:=1}"

# Paths are relative to the repo root (same layout as in the README: `bash run/eval_*.sh`).
SCRIPTS=(
  run/eval_pca.sh
  run/eval_hvg.sh
  run/eval_scvi.sh
  run/eval_cellplm.sh
  run/eval_scimilarity.sh
  run/eval_scfoundation.sh
  run/eval_scgpt_human.sh
  run/eval_scgpt_cancer.sh
  run/eval_state.sh
  run/eval_scconcept.sh
  run/eval_nicheformer.sh
  run/eval_geneformer_V1-10M-i2048.sh
  run/eval_geneformer_V2-104M-i4096.sh
  run/eval_geneformer_V2-316M-i4096.sh
  run/eval_geneformer_V2-104M_CLcancer-i4096.sh
  run/eval_geneformer_V1-10M-i2048_finetune.sh
  run/eval_geneformer_V2-104M-i4096_finetune.sh
)

failed=0
ran=0
for s in "${SCRIPTS[@]}"; do
  spath="$_REPO_ROOT/$s"
  if [[ ! -f "$spath" ]]; then
    echo "WARN: missing $spath — skip" >&2
    continue
  fi
  echo ""
  echo "========================================"
  echo "  $s"
  echo "========================================"
  ran=$((ran + 1))
  if bash "$spath"; then
    echo "OK: $s"
  else
    echo "FAIL: $s (exit $?)" >&2
    failed=$((failed + 1))
    if [[ "$FAIL_FAST" -eq 1 ]] || [[ "${EVAL_CONTINUE_ON_ERROR:-1}" == "0" ]]; then
      echo "Stopping: fail-fast or EVAL_CONTINUE_ON_ERROR=0" >&2
      exit 1
    fi
  fi
done

echo ""
echo "eval_all: finished $ran script(s), $failed failed."
[[ "$failed" -eq 0 ]]
exit $?
