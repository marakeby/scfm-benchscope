# Verification and rollout

Stage 20 locks the dual-gate workflow with CI, adversarial coverage, and one
synthetic model path that never needs an LLM or GPU.

## Continuous integration

The `CI` workflow installs the package, validates packaged candidate and
planning examples, runs the full unittest suite, and checks YAML syntax.

Locally:

```bash
pixi run -e default test
# or
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Synthetic dual-gate path

`tests/test_dual_gate_pipeline.py` walks one candidate through:

1. proposal-only planner workspace
2. deterministic lock/manifest materialization
3. pre-run execution approval
4. fake-host execution to `completed_unreviewed`
5. scientific acceptance
6. `--accepted-only` report publication

## Adversarial gates

`tests/test_adversarial_gates.py` proves that the platform refuses:

- execution without an approval record or with a fingerprint mismatch
- planner-invented `pixi.lock` / execution manifests
- mutable repository revisions
- invented secret names that are not environment-variable identifiers
- over-budget manifests
- altered bundles after grant
- retries beyond the approved attempt budget
- unreviewed or rejected results in official reports
- reordered execution steps and unexpected bundle files

## Rollout order

1. Keep discovery and planner proposal-only in CI.
2. Require branch protection on approval PRs.
3. Run one real model through `approval prepare` → PR merge →
   `approval grant` → `execute` → `review decide` →
   `report --accepted-only`.
4. Only then enable periodic discovery publishing on a schedule.
