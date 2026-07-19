#!/usr/bin/env bash
# Interactive checklist: walk scGPT through both human approval gates.
#
# Defaults match this machine (A100 GPU VM) and the scFM_eval checkout.
# Edit the CONFIG block if paths or identity change.
#
# Usage (from repo root):
#   bash scripts/rollout_scgpt_checklist.sh              # interactive
#   bash scripts/rollout_scgpt_checklist.sh --dry-run     # print commands only
#   TRANSPORT=fake bash scripts/rollout_scgpt_checklist.sh
#   TRANSPORT=local bash scripts/rollout_scgpt_checklist.sh
#
# Human gates (PR review / scientific accept) always pause for confirmation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------------------
# CONFIG — filled for va-ml-sc-a100-gpu-spotb / haitham
# ---------------------------------------------------------------------------
MODEL_ID="scgpt"
CANDIDATE="${CANDIDATE:-$ROOT/examples/models/candidates/scgpt.json}"
PLANNING_DIR="${PLANNING_DIR:-$ROOT/planning/${MODEL_ID}}"
BUNDLE_DIR="${BUNDLE_DIR:-$ROOT/approvals/${MODEL_ID}/${MODEL_ID}-attempt-1}"
BUNDLE_REPO_PATH="approvals/${MODEL_ID}/${MODEL_ID}-attempt-1"
APPROVAL_JSON="${APPROVAL_JSON:-$ROOT/execution-approvals/${MODEL_ID}-attempt-1.json}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/${MODEL_ID}-attempt-1}"
PUBLISHED_DIR="${PUBLISHED_DIR:-$ROOT/published/${MODEL_ID}-attempt-1}"
DRAFT_DIR="${DRAFT_DIR:-$ROOT/draft-reports/${MODEL_ID}-attempt-1}"

# Planner
PLANNER_PROVIDER="${PLANNER_PROVIDER:-openai}"
PLANNER_MODEL="${PLANNER_MODEL:-}"

# Pre-run budget (A100 on this VM). Adjust before opening the PR.
GPU_TYPE="${GPU_TYPE:-A100-SXM4-40GB}"
GPU_COUNT="${GPU_COUNT:-1}"
DISK_GB="${DISK_GB:-120}"
MAX_RUNTIME_MINUTES="${MAX_RUNTIME_MINUTES:-180}"
HOURLY_RATE_USD="${HOURLY_RATE_USD:-3.50}"
MAX_BUDGET_USD="${MAX_BUDGET_USD:-10.50}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
MANIFEST_ID="${MANIFEST_ID:-${MODEL_ID}-attempt-1}"

# Identity / GitHub
REVIEWER_IDENTITY="${REVIEWER_IDENTITY:-haitham}"
SCIENTIST_IDENTITY="${SCIENTIST_IDENTITY:-haitham}"
GITHUB_REPO="${GITHUB_REPO:-marakeby/scFM_eval}"
PR_URL="${PR_URL:-}"          # set after opening the PR, or pass interactively
MERGE_COMMIT="${MERGE_COMMIT:-}"  # 40-char SHA after merge

# Execution transport:
#   fake  — contract smoke only (no install / GPU)
#   local — run on this VM under LOCAL_ROOT (recommended here)
#   ssh   — copy job to REMOTE_* (for Actions or a laptop)
TRANSPORT="${TRANSPORT:-local}"
LOCAL_ROOT="${LOCAL_ROOT:-$HOME/mnt/scfm-jobs}"

# SSH defaults point at this VM's internal DNS (use from another host / GHA).
SSH_HOST="${SSH_HOST:-va-ml-sc-a100-gpu-spotb.c.vanallen-ml-sc.internal}"
SSH_USER="${SSH_USER:-haitham}"
SSH_REMOTE_ROOT="${SSH_REMOTE_ROOT:-/home/haitham/mnt/scfm-jobs}"
SSH_IDENTITY_FILE="${SSH_IDENTITY_FILE:-$HOME/.ssh/id_ed25519}"

# Runtime data/output mounts used by the evaluation harness on this VM
RUNTIME_ENV="${RUNTIME_ENV:-$ROOT/scripts/config/runtime_paths.env}"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
step=0
next_step() {
  step=$((step + 1))
  echo ""
  echo "============================================================"
  echo "STEP ${step}: $1"
  echo "============================================================"
}

pause() {
  local msg="${1:-Press Enter to continue (Ctrl-C to abort)...}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] $msg"
    return 0
  fi
  read -r -p "$msg " _
}

run() {
  echo "+" "$@"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  "$@"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

# Prefer an installed console script; fall back to this repo's Pixi env.
if command -v scfm-eval >/dev/null 2>&1; then
  SCFM_EVAL=(scfm-eval)
elif command -v pixi >/dev/null 2>&1; then
  SCFM_EVAL=(pixi run -e default -- scfm-eval)
else
  echo "Neither scfm-eval nor pixi is available on PATH." >&2
  exit 1
fi

scfm() {
  run "${SCFM_EVAL[@]}" "$@"
}

# ---------------------------------------------------------------------------
# STEP 0 — environment check
# ---------------------------------------------------------------------------
next_step "Check environment"
echo "Repo:            $ROOT"
echo "Candidate:       $CANDIDATE"
echo "Planning dir:    $PLANNING_DIR"
echo "Approval bundle: $BUNDLE_DIR"
echo "Run dir:         $RUN_DIR"
echo "Transport:       $TRANSPORT"
echo "GPU (requested): ${GPU_COUNT}x ${GPU_TYPE}"
echo "Budget:          \$${MAX_BUDGET_USD} / ${MAX_RUNTIME_MINUTES} min / ${MAX_ATTEMPTS} attempts"
if [[ -f "$RUNTIME_ENV" ]]; then
  # shellcheck source=/dev/null
  source "$RUNTIME_ENV"
  echo "SCFM_DATA_PATH:  ${SCFM_DATA_PATH:-unset}"
  echo "SCFM_OUTPUT_PATH:${SCFM_OUTPUT_PATH:-unset}"
  echo "SCFM_MODELS_PATH:${SCFM_MODELS_PATH:-unset}"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi
need_cmd pixi
[[ -f "$CANDIDATE" ]] || {
  echo "Candidate not found: $CANDIDATE" >&2
  exit 1
}
echo "Using: ${SCFM_EVAL[*]}"
scfm candidate validate "$CANDIDATE"
pause "Environment looks OK. Continue to planner?"

# ---------------------------------------------------------------------------
# STEP 1 — AI integration plan (proposal only)
# ---------------------------------------------------------------------------
next_step "Generate proposal-only integration plan"
if [[ -d "$PLANNING_DIR" ]]; then
  echo "Planning directory already exists: $PLANNING_DIR"
  echo "Reuse it, or move it aside before re-planning."
  pause "Continue with existing planning dir?"
else
  plan_args=(plan "$CANDIDATE" --provider "$PLANNER_PROVIDER" -o "$PLANNING_DIR")
  if [[ -n "$PLANNER_MODEL" ]]; then
    plan_args+=(--model "$PLANNER_MODEL")
  fi
  echo "Requires ${PLANNER_PROVIDER} credentials in the environment."
  pause "Run planner now?"
  scfm "${plan_args[@]}"
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  status_file="$PLANNING_DIR/planning-status.json"
  if [[ -f "$status_file" ]]; then
    python - <<'PY' "$status_file"
import json, sys
status = json.load(open(sys.argv[1], encoding="utf-8"))
print("Planning status:", status.get("status"))
for issue in status.get("issues") or []:
    print("Issue:", issue)
if status.get("status") != "ready":
    raise SystemExit(
        "Planner is not ready. Resolve unresolved fields and re-run plan."
    )
PY
  fi
  test -f "$PLANNING_DIR/model-spec.json"
  test -f "$PLANNING_DIR/integration-plan.json"
  test ! -e "$PLANNING_DIR/pixi.lock"
  test ! -e "$PLANNING_DIR/execution-manifest.json"
fi
pause "Review the planning workspace by hand, then continue."

# ---------------------------------------------------------------------------
# STEP 2 — Materialize lockfile + execution manifest
# ---------------------------------------------------------------------------
next_step "Prepare immutable pre-run approval bundle"
if [[ -d "$BUNDLE_DIR" ]]; then
  echo "Bundle already exists: $BUNDLE_DIR"
  pause "Reuse existing bundle?"
else
  pause "Run pixi lock + manifest materialization?"
  scfm approval prepare \
    "$CANDIDATE" \
    "$PLANNING_DIR" \
    --output "$BUNDLE_DIR" \
    --manifest-id "$MANIFEST_ID" \
    --gpu-type "$GPU_TYPE" \
    --gpu-count "$GPU_COUNT" \
    --disk-gb "$DISK_GB" \
    --max-runtime-minutes "$MAX_RUNTIME_MINUTES" \
    --hourly-rate-usd "$HOURLY_RATE_USD" \
    --max-budget-usd "$MAX_BUDGET_USD" \
    --max-attempts "$MAX_ATTEMPTS"
fi
scfm approval verify "$BUNDLE_DIR"
pause "Bundle verified. Continue to open the approval PR?"

# ---------------------------------------------------------------------------
# STEP 3 — Human pre-run gate (PR)
# ---------------------------------------------------------------------------
next_step "Human pre-run approval (one PR, one bundle)"
cat <<EOF
Manual checklist:
  [ ] Create branch: git switch -c approve/${MODEL_ID}-attempt-1
  [ ] Commit ONLY: ${BUNDLE_REPO_PATH}/
  [ ] Push and open PR against main using .github/PULL_REQUEST_TEMPLATE/model-evaluation.md
  [ ] Title suggestion: Approve ${MODEL_ID} attempt-1 execution manifest
  [ ] Wait for Validate model approval + human review
  [ ] Merge the PR (this is the pre-run approval boundary)

Repo: https://github.com/${GITHUB_REPO}
EOF
pause "After the PR is merged, press Enter."

if [[ -z "$PR_URL" ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    PR_URL="https://github.com/${GITHUB_REPO}/pull/NNN"
  else
    read -r -p "Merged PR URL: " PR_URL
  fi
fi
if [[ -z "$MERGE_COMMIT" ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    MERGE_COMMIT="0123456789abcdef0123456789abcdef01234567"
  else
    read -r -p "Merge commit SHA (40 hex chars): " MERGE_COMMIT
  fi
fi

# ---------------------------------------------------------------------------
# STEP 4 — Record execution approval
# ---------------------------------------------------------------------------
next_step "Record immutable execution-approval after merge"
mkdir -p "$(dirname "$APPROVAL_JSON")"
if [[ -f "$APPROVAL_JSON" ]]; then
  echo "Approval record already exists: $APPROVAL_JSON"
else
  scfm approval grant "$BUNDLE_DIR" \
    --output "$APPROVAL_JSON" \
    --approval-id "${MANIFEST_ID}-approval" \
    --identity "$REVIEWER_IDENTITY" \
    --method github_pr \
    --pr-url "$PR_URL" \
    --merge-commit "$MERGE_COMMIT" \
    --bundle-path "$BUNDLE_REPO_PATH"
fi
scfm contract validate execution-approval "$APPROVAL_JSON"
pause "Pre-run gate recorded. Continue to execution?"

# ---------------------------------------------------------------------------
# STEP 5 — Execute approved manifest
# ---------------------------------------------------------------------------
next_step "Execute approved manifest (transport=${TRANSPORT})"
if [[ -d "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR"
  echo "Choose a new RUN_DIR or remove it before re-executing."
  pause "Skip execution and continue to review?"
else
  case "$TRANSPORT" in
    fake)
      pause "Run fake transport smoke execution?"
      scfm execute "$BUNDLE_DIR" \
        --approval "$APPROVAL_JSON" \
        --output "$RUN_DIR" \
        --transport fake
      ;;
    local)
      if [[ -f "$RUNTIME_ENV" ]]; then
        # shellcheck source=/dev/null
        source "$RUNTIME_ENV"
      fi
      mkdir -p "$LOCAL_ROOT"
      pause "Run LOCAL execution on this A100 VM under $LOCAL_ROOT?"
      scfm execute "$BUNDLE_DIR" \
        --approval "$APPROVAL_JSON" \
        --output "$RUN_DIR" \
        --transport local \
        --local-root "$LOCAL_ROOT"
      ;;
    ssh)
      pause "Run SSH execution on ${SSH_USER}@${SSH_HOST}:${SSH_REMOTE_ROOT}?"
      scfm execute "$BUNDLE_DIR" \
        --approval "$APPROVAL_JSON" \
        --output "$RUN_DIR" \
        --transport ssh \
        --ssh-host "$SSH_HOST" \
        --ssh-user "$SSH_USER" \
        --ssh-remote-root "$SSH_REMOTE_ROOT" \
        --ssh-identity-file "$SSH_IDENTITY_FILE"
      ;;
    *)
      echo "Unknown TRANSPORT=$TRANSPORT (use fake|local|ssh)" >&2
      exit 1
      ;;
  esac
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  test -f "$RUN_DIR/execution-record.json"
  python - <<'PY' "$RUN_DIR/execution-record.json"
import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
print("Execution status:", record.get("status"))
print("Review status:   ", record.get("review_status"))
print("Estimated USD:   ", record.get("resources", {}).get("estimated_cost_usd"))
if record.get("status") != "completed_unreviewed":
    raise SystemExit("Execution did not complete successfully.")
PY
fi

echo "Official report must still fail before scientific acceptance:"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "+ ${SCFM_EVAL[*]} report $RUN_DIR --accepted-only -o ${PUBLISHED_DIR}-too-early"
else
  if "${SCFM_EVAL[@]}" report "$RUN_DIR" --accepted-only -o "${PUBLISHED_DIR}-too-early"; then
    echo "ERROR: accepted-only report succeeded before scientific review" >&2
    exit 1
  else
    echo "Good: unreviewed run cannot publish."
  fi
fi
pause "Inspect results under $RUN_DIR, then continue to scientific review."

# ---------------------------------------------------------------------------
# STEP 6 — Human post-run gate
# ---------------------------------------------------------------------------
next_step "Human scientific review"
cat <<EOF
Manual checklist:
  [ ] Inspect results.json, plots, and execution-record.json in:
      $RUN_DIR
  [ ] Decide: accepted | needs_tuning | rejected
EOF

if [[ "$DRY_RUN" -eq 1 ]]; then
  DECISION="accepted"
  RATIONALE="Dry-run placeholder rationale."
else
  read -r -p "Decision [accepted/needs_tuning/rejected]: " DECISION
  read -r -p "Rationale: " RATIONALE
fi

decide_args=(
  review decide "$RUN_DIR"
  --decision-id "${MANIFEST_ID}-${DECISION}"
  --decision "$DECISION"
  --identity "$SCIENTIST_IDENTITY"
  --rationale "$RATIONALE"
)

if [[ "$DECISION" == "needs_tuning" ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    decide_args+=(
      --change "dry-run tuning change"
      --expected-improvement "dry-run expected improvement"
      --max-additional-budget-usd 3.0
    )
  else
    read -r -p "Tuning change (one line): " TUNING_CHANGE
    read -r -p "Expected improvement: " EXPECTED
    read -r -p "Max additional budget USD: " EXTRA_BUDGET
    decide_args+=(
      --change "$TUNING_CHANGE"
      --expected-improvement "$EXPECTED"
      --max-additional-budget-usd "$EXTRA_BUDGET"
    )
  fi
fi

pause "Write the review decision now?"
scfm "${decide_args[@]}"

# ---------------------------------------------------------------------------
# STEP 7 — Reports
# ---------------------------------------------------------------------------
next_step "Build draft and (if accepted) official reports"
scfm report "$RUN_DIR" --output "$DRAFT_DIR" --title "Draft scGPT attempt-1"
if [[ "$DECISION" == "accepted" ]]; then
  scfm report "$RUN_DIR" \
    --accepted-only \
    --output "$PUBLISHED_DIR" \
    --title "Accepted scGPT attempt-1"
  echo "Official report: $PUBLISHED_DIR/report.html"
else
  echo "Decision was $DECISION — official accepted-only report is intentionally skipped."
  if [[ "$DECISION" == "needs_tuning" ]]; then
    echo "Material changes still require a NEW approval PR before re-execution."
  fi
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
next_step "Rollout checklist complete"
cat <<EOF
Artifacts:
  candidate:   $CANDIDATE
  planning:    $PLANNING_DIR
  bundle:      $BUNDLE_DIR
  approval:    $APPROVAL_JSON
  run:         $RUN_DIR
  draft:       $DRAFT_DIR
  published:   $PUBLISHED_DIR  (only if accepted)

Recommended next run order on this VM:
  1) TRANSPORT=fake   bash scripts/rollout_scgpt_checklist.sh
  2) TRANSPORT=local  bash scripts/rollout_scgpt_checklist.sh
  3) Only then wire GitHub Actions SSH secrets to:
       host=$SSH_HOST
       user=$SSH_USER
       root=$SSH_REMOTE_ROOT
EOF
