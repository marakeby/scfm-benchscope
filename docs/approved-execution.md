# Approved execution

Only a verified approval bundle plus a separate execution-approval record may
spend evaluation budget. The executor never invents steps, expands retries, or
publishes scientific results.

## Record a merged pull-request approval

```bash
scfm-eval approval grant approvals/scgpt/scgpt-attempt-1 \
  --output execution-approvals/scgpt-attempt-1.json \
  --approval-id scgpt-attempt-1-approval \
  --identity github-user \
  --pr-url https://github.com/org/scFM_eval/pull/123 \
  --merge-commit 0123456789abcdef0123456789abcdef01234567 \
  --bundle-path approvals/scgpt/scgpt-attempt-1
```

The grant command re-verifies the bundle and writes an immutable approval
record bound to the manifest fingerprint. Comments or labels are not enough.

## Execute one approved manifest

Dry run with the fake transport (CI and local smoke tests):

```bash
scfm-eval execute approvals/scgpt/scgpt-attempt-1 \
  --approval execution-approvals/scgpt-attempt-1.json \
  --output runs/scgpt-attempt-1 \
  --transport fake
```

Local job directory:

```bash
scfm-eval execute approvals/scgpt/scgpt-attempt-1 \
  --approval execution-approvals/scgpt-attempt-1.json \
  --output runs/scgpt-attempt-1 \
  --transport local \
  --local-root /tmp/scfm-jobs
```

GPU VM over SSH:

```bash
scfm-eval execute approvals/scgpt/scgpt-attempt-1 \
  --approval execution-approvals/scgpt-attempt-1.json \
  --output runs/scgpt-attempt-1 \
  --transport ssh \
  --ssh-host gpu.example.org \
  --ssh-user evaluator \
  --ssh-remote-root /data/scfm-jobs \
  --ssh-identity-file ~/.ssh/scfm_eval
```

The executor:

1. verifies the approval bundle fingerprints;
2. requires an `approved` execution-approval for the same manifest fingerprint;
3. runs the fixed step order from the manifest;
4. retries only allowlisted steps within `max_attempts`;
5. stops when estimated GPU cost exceeds `max_budget_usd`;
6. writes `execution-record.json` with status `completed_unreviewed`.

Successful technical completion is not scientific acceptance. Stage 19 records
`accepted`, `needs_tuning`, or `rejected` before report publication.
