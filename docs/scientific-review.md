# Scientific review

A technically completed execution starts as `completed_unreviewed`. It cannot
enter official reports until a human records an immutable review decision bound
to the exact `results.json` checksum and manifest fingerprint.

## Record a decision

```bash
scfm-eval review decide runs/scgpt-attempt-1 \
  --decision-id scgpt-attempt-1-accepted \
  --decision accepted \
  --identity alice \
  --rationale "Cell-type metrics and controls look scientifically sound."
```

Allowed decisions:

- `accepted` — eligible for official reports when publication is enabled
- `needs_tuning` — writes tuning lineage; blocks publication
- `rejected` — blocks publication

`needs_tuning` also requires proposed changes and remaining budget:

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
`execution-record.json`. For tuning it also writes `tuning-lineage.json`.
Material dependency, source, weight, dataset, command, or budget changes still
require a new pre-run approval pull request before re-execution.

## Draft vs official reports

Draft reports include every valid run and label each with `review_status`
(`local`, `completed_unreviewed`, `accepted`, `needs_tuning`, `rejected`):

```bash
scfm-eval report "$SCFM_OUTPUT_PATH"
```

Official publication keeps only accepted runs:

```bash
scfm-eval report "$SCFM_OUTPUT_PATH" \
  --accepted-only \
  --output ./published-report \
  --title "Accepted model comparison"
```

Local YAML evaluations without an execution record are marked `local`. They
appear in draft reports and are excluded from `--accepted-only` publication.
