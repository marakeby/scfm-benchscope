# Model candidates

A candidate is the file discovery hands to the planner. It records what is
known about a model. It is not an install recipe and not permission to run
anything.

Example: [`examples/models/candidates/scgpt.json`](../../examples/models/candidates/scgpt.json).

## What belongs in the file

- a stable lowercase `candidate_id`
- when it was found, by which agent, and how confident the find is
- model name and optional summary
- paper, repository, and weight links
- suggested evaluation tasks
- `unresolved_fields` for anything still missing

At least one paper, repository, or weight source is required. Leave missing
values empty and list them in `unresolved_fields`. Do not invent links.

Use public HTTPS URLs with no passwords in them. A repository revision in
the candidate is only a hint. The planner must pin an exact commit and
weight files later.

## Validate a candidate

```bash
scfm-eval candidate validate examples/models/candidates/scgpt.json
```

A valid file prints the candidate id and a SHA-256 fingerprint. This command
does not download anything or run the model.

In Python:

```python
from scfm_cancer_eval.onboarding import load_model_candidate

candidate = load_model_candidate("candidate.json")
print(candidate.candidate_id)
print(candidate.fingerprint)
```

`to_dict()` returns a copy you can pass to the planner.

## Schema

Set both schema fields:

```json
{
  "schema": {
    "name": "scfm_eval.model_candidate",
    "version": "1.0.0"
  }
}
```

Unknown fields and other versions are rejected. Add a new version if the
format needs to change.
