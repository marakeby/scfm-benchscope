# Pre-run approval

A ready plan is still a draft. Before anything runs, prepare a lockfile and
a budget sheet, then have a person merge them in a pull request.

## Prepare the bundle

```bash
scfm-eval approval prepare \
  docs/candidates/2026-07-17/scgpt.json \
  planning/scgpt \
  --output approvals/scgpt/scgpt-attempt-1 \
  --manifest-id scgpt-attempt-1 \
  --gpu-type A10G \
  --gpu-count 1 \
  --disk-gb 80 \
  --max-runtime-minutes 120 \
  --hourly-rate-usd 1.20 \
  --max-budget-usd 2.40
```

This runs `pixi lock --no-install`. It does not install packages, download
weights, or run model code. Existing output directories are never
overwritten.

The new folder holds the candidate, planner files, generated files,
`pixi.lock`, `execution-manifest.json`, and `approval-request.json`.

The budget must be at least:

```text
GPU count × runtime hours × hourly rate × maximum attempts
```

Add `--secret`, `--allow-host`, `--retryable-step`, or `--experiment-path`
only if the model needs them. Repository and weight hosts are allowed
automatically.

## Review and merge

Check the bundle locally:

```bash
scfm-eval approval verify approvals/scgpt/scgpt-attempt-1
```

Commit exactly one new bundle on a model-specific branch. Open a pull
request with the `model-evaluation.md` template. The approval workflow
rejects extra bundles or edits to existing ones.

Merging the pull request approves that exact manifest fingerprint. On the
default branch, require:

- at least one human approval
- the **Validate model approval** check
- stale reviews to be dismissed when new commits are pushed

GitHub settings live in the repo settings, not in this package. Without
them, a merge is not a reliable approval.

If you change code, dependencies, sources, weights, datasets, permissions,
runtime, retries, or budget, make a new bundle and a new pull request.

After merge, [record the grant and run](approved-execution.md). Results still
need [scientific review](scientific-review.md) before official reports.
