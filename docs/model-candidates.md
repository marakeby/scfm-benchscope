# Model candidates

A model candidate is the handoff from discovery to AI-assisted integration
planning. It records evidence about a potentially relevant model; it is not an
installation recipe, execution plan, approval, or evaluation request.

The current contract is `scfm_eval.model_candidate` v1.0.0. A complete example
is available at
[`examples/models/candidates/scgpt.json`](../examples/models/candidates/scgpt.json).

## Required information

Each candidate records:

- a stable lowercase `candidate_id`
- discovery time, agent identity, source type, and confidence
- the model name and optional summary
- paper, repository, and weight evidence
- suggested evaluation tasks
- information the integration planner still needs to resolve

At least one paper, repository, or weight source is required. Missing sources
remain `null` or empty and should be named in `unresolved_fields`; discovery
agents must not invent values to make a candidate appear complete.

Links must be public HTTPS URLs without embedded credentials. Candidate
repository revisions are only hints. The later integration planner must resolve
an immutable commit and exact weight files before proposing execution.

## Validate discovery output

```bash
scfm-eval candidate validate examples/models/candidates/scgpt.json
```

Successful validation prints the candidate identity and a SHA-256 fingerprint
of its canonical JSON representation. Validation performs no network access,
repository checkout, installation, or model execution.

Python callers can use the same contract:

```python
from scfm_cancer_eval.onboarding import load_model_candidate

candidate = load_model_candidate("candidate.json")
print(candidate.candidate_id)
print(candidate.fingerprint)
```

`ModelCandidate` stores an immutable canonical representation. `to_dict()`
returns a new copy for downstream planning.

## Versioning

Producers must set both schema fields:

```json
{
  "schema": {
    "name": "scfm_eval.model_candidate",
    "version": "1.0.0"
  }
}
```

Unknown fields and unsupported schema versions are rejected. Future contract
changes should add a new schema version and an explicit migration instead of
silently changing v1.0.0.
