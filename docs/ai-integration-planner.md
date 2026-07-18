# AI integration planner

The integration planner turns one validated `ModelCandidate` into a reviewable
workspace. It researches the paper, repository, dependencies, model-loading
code, and weight documentation through a selected AI provider.

The planner is proposal-only. It does not clone repositories, install packages,
download weights, run generated code, create a Pixi environment, or start an
evaluation.

## Run with OpenAI

```bash
python -m pip install openai
export OPENAI_API_KEY=...

scfm-eval plan candidate.json \
  --provider openai \
  --model gpt-4o-search-preview \
  --output planning/my-model
```

`--model` is optional. The default can also be configured with
`SCFM_PLANNER_OPENAI_MODEL`.

## Run with Anthropic

```bash
python -m pip install anthropic
export ANTHROPIC_API_KEY=...

scfm-eval plan candidate.json \
  --provider anthropic \
  --output planning/my-model
```

The Anthropic model can be configured through `--model` or
`SCFM_PLANNER_ANTHROPIC_MODEL`.

Set `SCFM_PLANNER_PROVIDER` to change the default provider without changing
commands or application code.

## Planner output

A ready proposal contains:

- `proposal.json`: original provider output
- `planning-status.json`: provider, status, issues, and fingerprints
- `model-spec.json`: validated, candidate-linked model specification
- `integration-plan.json`: validated environment and integration proposal
- `pixi.toml`
- generated adapter, experiment YAML, and optional test files

Generated files are limited to `pixi.toml`, `integrations/`, `experiments/`,
and `tests/`. Paths, sizes, fingerprints, source commits, weight checksums, and
contract fields are validated before the workspace is written.

If required facts cannot be verified, the provider should return
`needs_input`. The workspace then contains only the proposal and status files;
no adapter or environment files are accepted.

The planner does not generate an execution manifest or `pixi.lock`. Those
require deterministic materialization and human review in the next approval
stage.

## Add another provider

A provider only needs `name`, `model`, and `generate(prompt)`:

```python
class MyPlannerProvider:
    name = "my-provider"

    def __init__(self, model=None):
        self.model = model or "default-model"

    def generate(self, prompt):
        return my_api_call_that_returns_a_dict(prompt)
```

Load it without changing the planner:

```bash
scfm-eval plan candidate.json \
  --provider my_package.providers:MyPlannerProvider
```

Provider SDKs and API keys remain optional. Core validation and tests do not
require either OpenAI or Anthropic.
