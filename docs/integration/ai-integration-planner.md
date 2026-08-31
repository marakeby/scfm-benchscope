# AI integration planner

The planner reads one candidate and writes a draft workspace: how to identify
the model, which commit and weights to use, and how you would install and
evaluate it.

It does not clone the repo, install packages, download weights, or start a
run. Next step: [Pre-run approval](pre-run-approval.md).

## Run with OpenAI

```bash
python -m pip install openai
export OPENAI_API_KEY=...

scfm-eval plan candidate.json \
  --provider openai \
  --output planning/my-model
```

OpenAI uses the Responses API with `web_search`. The default model is
`gpt-5.5`. Override it with `--model` or `SCFM_PLANNER_OPENAI_MODEL`.

## Run with Anthropic

```bash
python -m pip install anthropic
export ANTHROPIC_API_KEY=...

scfm-eval plan candidate.json \
  --provider anthropic \
  --output planning/my-model
```

Override the model with `--model` or `SCFM_PLANNER_ANTHROPIC_MODEL`.

Set `SCFM_PLANNER_PROVIDER` to change the default provider.

## What a ready plan contains

- `proposal.json` — raw model output
- `planning-status.json` — status, issues, and fingerprints
- `model-spec.json` — identity, pinned commit, weight files
- `integration-plan.json` — environment and integration draft
- `pixi.toml`
- generated adapter, experiment YAML, and optional tests

Generated files may only land in `pixi.toml`, `integrations/`,
`experiments/`, and `tests/`. If the agent cannot verify required facts, the
status is `needs_input` and only the proposal and status files are written.

The planner does not write `pixi.lock` or an execution manifest. Those come
from `approval prepare`.

## Add another provider

A provider needs `name`, `model`, and `generate(prompt)`:

```python
class MyPlannerProvider:
    name = "my-provider"

    def __init__(self, model=None):
        self.model = model or "default-model"

    def generate(self, prompt):
        return my_api_call_that_returns_a_dict(prompt)
```

```bash
scfm-eval plan candidate.json \
  --provider my_package.providers:MyPlannerProvider
```

Core tests do not need OpenAI or Anthropic installed.
