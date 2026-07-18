# Planning and approval contracts

The onboarding workflow uses four immutable JSON documents after discovery.
Each document has a versioned JSON Schema, deterministic runtime validation,
and a SHA-256 fingerprint of its canonical representation.

These contracts define data boundaries only. Creating or validating them does
not clone repositories, install dependencies, download weights, invoke an AI
agent, or run an evaluation.

## Model specification

`scfm_eval.model_spec` v1.0.0 records the integration planner's enriched model
identity:

- originating candidate ID and fingerprint
- model identity and license review
- HTTPS repository URL pinned to a full Git commit
- exact weight files and SHA-256 checksums
- adapter import path and output key
- supported tasks and hardware constraints

Validate an example:

```bash
scfm-eval contract validate model-spec \
  examples/models/planning/model-spec.json
```

## Integration plan

`scfm_eval.integration_plan` v1.0.0 is the AI planner's proposal. It contains
the Pixi dependency proposal, package installation mode, generated files and
their hashes, smoke tests, resource estimates, assumptions, risks, and
unresolved fields.

The plan intentionally contains structured fields instead of arbitrary shell
scripts. The future executor will derive its operations from an approved
manifest rather than execute free-form agent output.

```bash
scfm-eval contract validate integration-plan \
  examples/models/planning/integration-plan.json
```

## Execution manifest

`scfm_eval.execution_manifest` v1.0.0 is the exact input proposed for human
approval. It pins:

- model-spec and integration-plan fingerprints
- repository commit, weight checksums, and generated-file checksums
- Pixi lockfile
- adapter, experiment, tasks, and expected outputs
- fixed execution order
- network hosts, secret names, and read-only dataset access
- GPU, disk, timeout, hourly rate, total budget, and bounded retries

Validation rejects plans whose worst-case retry cost exceeds the approved
budget. A valid manifest is still not approved; `scfm-eval approval grant`
records approval of its exact fingerprint before
[approved execution](approved-execution.md).

```bash
scfm-eval contract validate execution-manifest \
  examples/models/planning/execution-manifest.json
```

## Review decision

`scfm_eval.review_decision` v1.0.0 binds a human decision to an exact run,
result checksum, and manifest fingerprint.

Allowed decisions are:

- `accepted`
- `needs_tuning`
- `rejected`

Only accepted runs may be included in published reports or promoted as
baselines. `needs_tuning` requires proposed changes and an additional budget;
other decisions cannot contain tuning instructions.

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

Unknown fields and unsupported schema versions are rejected. Contract changes
must add a new version instead of silently changing v1.0.0.
