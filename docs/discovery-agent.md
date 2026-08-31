# Model discovery agent

Discovery finds public models and updates the catalog. It does not install
packages, download weights, or run evaluations.

Two scripts:

- `docs/agent.py` — scheduled OpenAI web-search agent
- `docs/agent_claude.py` — manual Anthropic fallback

## Run it locally

From the repository root:

```bash
python -m pip install "openai>=1.76.0"
python -m pip install --no-deps -e .
python docs/agent.py
```

You need `OPENAI_API_KEY`.

## What it writes

- `docs/models.json` and `docs/models.html` — the public catalog
- `docs/candidates/YYYY-MM-DD/*.json` — one evidence card per new model

The catalog can list a model even if some fields are missing. Candidate
files keep only what the [integration planner](integration/README.md) needs:
name, paper, repository, weights, and open questions. Bad or private links
are left unresolved instead of invented.

See [Model candidates](integration/model-candidates.md) for the file format.

## GitHub automation

`.github/workflows/update.yml` runs the OpenAI agent every three days
(`OPENAI_API_KEY`). `.github/workflows/update_claude.yml` is a manual
fallback (`ANTHROPIC_API_KEY`). Run only one scheduled provider so the
catalog is not written twice.

Those workflows install the evaluation package without scientific
dependencies. Validating a candidate uses only the Python standard library.

Human approval starts later, when you plan and run a model. See
[Integration agent](integration/README.md).
