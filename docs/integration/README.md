# Integration agent

Use this path when you want the AI to draft how to evaluate a new model.
The agent writes a plan. It does not install the model or start a GPU job.

You still approve twice:

1. **Before the run** — a person reviews the budget and merges a pull request
2. **After the run** — a person looks at the metrics and accepts or rejects them

[Discovery](../discovery-agent.md) only finds models and writes a candidate
file. Start here after that file exists.

## What to do

**1. Check the candidate.**
A [candidate](model-candidates.md) is a short note: paper, code, weights,
and gaps. Validate it, then pass it to the planner.

```bash
scfm-eval candidate validate examples/models/candidates/scgpt.json
```

**2. Ask for a plan.**
The [planner](ai-integration-planner.md) writes a workspace: which commit
and weight files to use, and how you would install and evaluate the model.
Treat the output as a draft.

```bash
scfm-eval plan candidate.json --provider openai --output planning/my-model
```

**3. Build the approval bundle.**
[Pre-run approval](pre-run-approval.md) locks the real packages and writes
a run sheet (GPU, time, money, retries). Still no install and no run.
[Contracts](planning-contracts.md) explains the JSON files if you need to
validate them by hand.

**4. Merge one pull request, then record the grant.**
A person reviews the bundle and merges it. Then
[record the grant](approved-execution.md) so the runner has a written
approval.

**5. Run that plan.**
[Execution](approved-execution.md) follows the run sheet step for step.
When it finishes, the status is “done, not yet reviewed.”

**6. Review the science, then report.**
[Scientific review](scientific-review.md) records accepted, needs tuning,
or rejected. Official reports keep only accepted runs.

```bash
scfm-eval report "$SCFM_OUTPUT_PATH" --accepted-only --output ./published-report
```

To test the whole path, or walk scGPT on a real machine, see
[Verification and rollout](verification-rollout.md).
