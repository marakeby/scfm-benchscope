# Scientific review

A finished job starts as `completed_unreviewed`. It cannot appear in official
reports until a person records a decision.

## Record a decision

```bash
scfm-eval review decide runs/scgpt-attempt-1 \
  --decision-id scgpt-attempt-1-accepted \
  --decision accepted \
  --identity alice \
  --rationale "Cell-type metrics and controls look scientifically sound."
```

Choices:

- `accepted` — allowed in official reports
- `needs_tuning` — blocked; records what to change
- `rejected` — blocked

`needs_tuning` also needs proposed changes and remaining budget:

```bash
scfm-eval review decide runs/scgpt-attempt-1 \
  --decision-id scgpt-attempt-1-tuning \
  --decision needs_tuning \
  --identity alice \
  --rationale "Batch effect remains too strong." \
  --change "retune preprocessing" \
  --change "add donor covariate" \
  --expected-improvement "Higher batch-mixed NMI" \
  --max-additional-budget-usd 3.0
```

The command writes `review-decision.json` and updates
`execution-record.json`. Tuning also writes `tuning-lineage.json`.

If you change dependencies, sources, weights, datasets, commands, or
budget, open a new [pre-run approval](pre-run-approval.md) before running
again.

## Draft vs official reports

A draft includes every valid run and labels each with `review_status`:

```bash
scfm-eval report "$SCFM_OUTPUT_PATH"
```

Official reports keep only accepted runs:

```bash
scfm-eval report "$SCFM_OUTPUT_PATH" \
  --accepted-only \
  --output ./published-report \
  --title "Accepted model comparison"
```

Local YAML runs with no execution record are marked `local`. They show up
in drafts and are left out of `--accepted-only`.
