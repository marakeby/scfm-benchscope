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

cd "$(dirname "$0")"

FAIL_FAST=0
if [[ "${1:-}" == "--fail-fast" ]]; then
  FAIL_FAST=1
fi

# If set to 0 or empty, stop on first error when FAIL_FAST is 1; otherwise continue.
: "${EVAL_CONTINUE_ON_ERROR:=1}"

SCRIPTS=(
  eval_pca.sh
  eval_hvg.sh
  eval_scvi.sh
  eval_cellplm.sh
  eval_scimilarity.sh
  eval_scfoundation.sh
  eval_scgpt_human.sh
  eval_scgpt_cancer.sh
  eval_state.sh
  eval_scconcept.sh
  eval_nicheformer.sh
  eval_geneformer_V1-10M-i2048.sh
  eval_geneformer_V2-104M-i4096.sh
  eval_geneformer_V2-316M-i4096.sh
  eval_geneformer_V2-104M_CLcancer-i4096.sh
  eval_geneformer_V1-10M-i2048_finetune.sh
  eval_geneformer_V2-104M-i4096_finetune.sh
)

failed=0
ran=0
for s in "${SCRIPTS[@]}"; do
  if [[ ! -f "$s" ]]; then
    echo "WARN: missing $s — skip" >&2
    continue
  fi
  echo ""
  echo "========================================"
  echo "  $s"
  echo "========================================"
  ran=$((ran + 1))
  if bash "$s"; then
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
