# Planning and approval contracts

These JSON files pin facts after discovery. Validating them does not install
packages or run a model.

Each file has a versioned schema and a SHA-256 fingerprint. Unknown fields
and other versions are rejected. Add a new version if the format must change.

## Model specification

What the planner decided the model is: candidate link, license, pinned Git
commit, weight files and checksums, adapter path, tasks, and hardware.

```bash
scfm-eval contract validate model-spec \
  examples/models/planning/model-spec.json
```

## Integration plan

The planner’s draft: Pixi dependencies, generated files, smoke tests,
resource estimates, assumptions, and open questions. It is structured data,
not a shell script.

```bash
scfm-eval contract validate integration-plan \
  examples/models/planning/integration-plan.json
```

## Execution manifest

The run sheet a person approves. It pins:

- model-spec and integration-plan fingerprints
- commit, weight checksums, generated-file checksums, and the Pixi lockfile
- adapter, experiment, tasks, and step order
- allowed hosts, secret names, and read-only dataset access
- GPU, disk, timeout, hourly rate, total budget, and retry limit

A budget below GPU count × hours × rate × max attempts is rejected. A valid
manifest is still not approved. Record the grant with
`scfm-eval approval grant` before [you run it](approved-execution.md).

```bash
scfm-eval contract validate execution-manifest \
  examples/models/planning/execution-manifest.json
```

## Review decision

A person’s decision on one finished run, bound to that run’s `results.json`
checksum and manifest fingerprint.

Allowed values: `accepted`, `needs_tuning`, `rejected`. Only accepted runs
go in official reports. `needs_tuning` also needs proposed changes and an
extra budget. Write the decision with `scfm-eval review decide`. See
[Scientific review](scientific-review.md).

```bash
scfm-eval contract validate review-decision \
  examples/models/planning/review-decision.json
```

## Python API

```python
from scfm_cancer_eval.onboarding import (
    load_execution_manifest,
    load_integration_plan,
    load_model_candidate,
    load_model_spec,
    load_review_decision,
    validate_planning_chain,
)

candidate = load_model_candidate("candidate.json")
model = load_model_spec("model-spec.json")
plan = load_integration_plan("integration-plan.json")
manifest = load_execution_manifest("execution-manifest.json")
decision = load_review_decision("review-decision.json")

validate_planning_chain(candidate, model, plan, manifest)
print(model.fingerprint)
print(plan.fingerprint)
print(manifest.fingerprint)
print(decision.fingerprint)
```
