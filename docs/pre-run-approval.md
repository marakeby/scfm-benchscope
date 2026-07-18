# Pre-run approval

A ready AI plan is still only a proposal. Before model code can run, the
approval command resolves its Pixi lockfile and creates a bounded execution
manifest for human review.

## Prepare one approval bundle

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

The command runs `pixi lock --no-install`. It may resolve package metadata, but
it does not create an environment, install dependencies, download weights, or
execute generated model code.

The new directory contains the candidate, original planner proposal and status,
proposed and finalized integration plans, model specification, generated
files, `pixi.lock`, `execution-manifest.json`, and `approval-request.json`.
Existing output directories are never overwritten.

The manifest rejects a budget below:

```text
GPU count × runtime hours × hourly rate × maximum attempts
```

Use `--secret`, `--allow-host`, `--retryable-step`, or `--experiment-path` only
when the model needs them. Repository and weight hosts are allowlisted
automatically.

## Review and approve

Verify locally before opening the pull request:

```bash
scfm-eval approval verify approvals/scgpt/scgpt-attempt-1
```

Commit exactly one new bundle on a model-specific branch and open a pull
request with the `model-evaluation.md` template. The approval workflow verifies
all fingerprints and rejects changed existing bundles or multiple bundles in
one pull request.

Merging the pull request approves exactly the manifest fingerprint in that
bundle. Configure branch protection for the default branch to require:

- at least one approving human review;
- the **Validate model approval** status check;
- dismissal of stale reviews when new commits are pushed.

The repository cannot enforce those GitHub settings itself. Without branch
protection, a merge is not a reliable human-approval boundary.

Any change to code, dependencies, sources, weights, datasets, permissions,
runtime, retries, or budget requires a new manifest and a new pull request.
After merge, record the approval and run the job with
[Approved execution](approved-execution.md). Result publication remains
separately blocked until post-run scientific review.
