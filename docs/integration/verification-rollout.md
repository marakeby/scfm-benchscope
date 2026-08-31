# Verification and rollout

Use this page to test the integration path and to walk one real model
through it. The steps themselves are in
[Integration agent](README.md).

Command details: [planner](ai-integration-planner.md),
[pre-run approval](pre-run-approval.md),
[execution](approved-execution.md),
[scientific review](scientific-review.md).

## Run the tests

CI installs the package, validates the packaged examples, runs the unit
tests, and checks YAML syntax.

Locally:

```bash
pixi run -e default test
# or
python -m unittest discover -s tests -p 'test_*.py' -v
```

`tests/test_dual_gate_pipeline.py` walks one fake candidate through plan →
approve → fake-host run → accept → `--accepted-only` report. It does not
need an LLM or a GPU.

`tests/test_adversarial_gates.py` checks that the runner refuses missing
approvals, changed bundles, invented lockfiles, over-budget manifests,
extra retries, and unreviewed results in official reports.

## Turn this on for real models

1. Keep discovery and the planner as “write files only” in CI. Those jobs
   must not install environments or start evaluations.
2. Protect approval pull requests: human review, the **Validate model
   approval** check, and dismiss stale reviews.
3. Walk one real model: prepare → merge → grant → execute → review →
   `report --accepted-only`. The scGPT script below does that.
4. Only then turn on scheduled discovery publishing. Finding models is
   safe to automate. Running them and publishing results is not.

## Walk scGPT

`scripts/rollout_scgpt_checklist.sh` prints each command and pauses at the
human steps (read the plan, merge the PR, judge the results).

Defaults in the script match the A100 host `va-ml-sc-a100-gpu-spotb`, this
checkout, and `scripts/config/runtime_paths.env`. Override paths or budget
with environment variables. See the CONFIG block at the top of the script.

```bash
# Print commands only
bash scripts/rollout_scgpt_checklist.sh --dry-run

# Fake run (no install / GPU)
TRANSPORT=fake bash scripts/rollout_scgpt_checklist.sh

# Real run on this machine (after the approval PR is merged)
TRANSPORT=local bash scripts/rollout_scgpt_checklist.sh
```
