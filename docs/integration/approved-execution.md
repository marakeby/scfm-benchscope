# Approved execution

To spend evaluation budget you need two things: the verified bundle, and a
separate grant file from the merged pull request.

## Record the grant

```bash
scfm-eval approval grant approvals/scgpt/scgpt-attempt-1 \
  --output execution-approvals/scgpt-attempt-1.json \
  --approval-id scgpt-attempt-1-approval \
  --identity github-user \
  --pr-url https://github.com/org/scFM_eval/pull/123 \
  --merge-commit 0123456789abcdef0123456789abcdef01234567 \
  --bundle-path approvals/scgpt/scgpt-attempt-1
```

The command checks the bundle again and writes a grant bound to the
manifest fingerprint. A comment or label is not enough.

## Run the approved plan

Smoke test (no install, no GPU):

```bash
scfm-eval execute approvals/scgpt/scgpt-attempt-1 \
  --approval execution-approvals/scgpt-attempt-1.json \
  --output runs/scgpt-attempt-1 \
  --transport fake
```

On this machine:

```bash
scfm-eval execute approvals/scgpt/scgpt-attempt-1 \
  --approval execution-approvals/scgpt-attempt-1.json \
  --output runs/scgpt-attempt-1 \
  --transport local \
  --local-root /tmp/scfm-jobs
```

On a GPU VM over SSH:

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

The runner:

1. checks the bundle fingerprints
2. requires a matching `approved` grant
3. runs the steps in the manifest, in order
4. retries only allowlisted steps, up to `max_attempts`
5. stops if estimated GPU cost exceeds `max_budget_usd`
6. writes `execution-record.json` with status `completed_unreviewed`

A finished job is not a published result. Record accepted, needs tuning, or
rejected in [Scientific review](scientific-review.md) before official
reports.
