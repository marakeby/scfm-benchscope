# Model discovery agent

The existing discovery agents remain responsible for the public model catalog:

- `docs/agent.py`: scheduled OpenAI web-search agent
- `docs/agent_claude.py`: manually triggered Anthropic alternative

Discovery does not install model packages, download model files, run model
code, spend evaluation compute, or approve scientific results. It may therefore
publish catalog updates automatically.

## Outputs

Each successful run updates:

- `docs/models.json`: classified model catalog
- `docs/models.html`: static searchable catalog
- `docs/candidates/YYYY-MM-DD/*.json`: immutable candidate evidence added by
  that run

Catalog classifications stay in `models.json`. Candidate records contain only
the smaller evidence contract needed by the integration planner: model name,
paper, repository, weights, provenance, confidence, and unresolved fields.

Invalid or non-public links are omitted from candidate records and marked
unresolved. The model can still appear in the discovery catalog. Missing
architecture or biological classifications remain empty rather than receiving
invented defaults.

## Automation

`.github/workflows/update.yml` runs the OpenAI agent every three days and
publishes changed catalog and candidate files. It requires `OPENAI_API_KEY`.

`.github/workflows/update_claude.yml` is a manual fallback requiring
`ANTHROPIC_API_KEY`. Keeping only one scheduled provider avoids simultaneous
catalog writes and duplicate API spending.

Both workflows install the evaluation package without scientific dependencies;
candidate validation itself uses only the Python standard library.

## Local run

From the repository root:

```bash
python -m pip install "openai>=1.76.0"
python -m pip install --no-deps -e .
python docs/agent.py
```

The agent automatically publishes factual discovery output. Human approval
begins later, when the integration planner proposes executable files,
permissions, and a compute budget.
