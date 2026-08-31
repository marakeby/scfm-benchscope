# Comparing and visualizing results

Every finished evaluation writes a `results.json`. Reports read those files,
whether the run came from YAML, the Python API, Pixi, or Docker.

## Build a report

```bash
scfm-eval report "$SCFM_OUTPUT_PATH"
```

This creates `$SCFM_OUTPUT_PATH/report/` with:

- `report.html` — open this in a browser (no server needed)
- `comparison.json` — structured records
- `comparison.csv` — spreadsheet-friendly table

The HTML report can filter by text, dataset, and evaluation kind. Click a
column header to sort. Use the metric selector to plot values.

Write the report somewhere else, or give it a title:

```bash
scfm-eval report ./output \
  --output ./published-report \
  --title "BRCA model comparison"
```

A normal report includes every valid run and labels each with
`review_status`. To publish only scientifically accepted runs:

```bash
scfm-eval report ./output \
  --accepted-only \
  --output ./published-report \
  --title "Accepted BRCA model comparison"
```

See [Scientific review](integration/scientific-review.md) for how a run
becomes accepted.

## Compare specific runs

Pass `results.json` files, run directories, or a mix:

```bash
scfm-eval compare \
  output/api/model_a/results.json \
  output/api/model_b \
  --output ./model-comparison
```

Directories are searched recursively. Duplicate paths are ignored.

Invalid or missing files are skipped by default. Problems are listed in
`comparison.json` and `report.html`. Use `--strict` if you want the command
to fail when any requested result is bad:

```bash
scfm-eval compare run-a/results.json run-b/results.json --strict
```

The command always fails if it finds no valid result.

## Collect dashboard tables

```bash
scfm-eval report "$SCFM_OUTPUT_PATH" --collect --output ./dashboard-metrics
```

This writes embedding and classification CSVs/JSON. Copy them into
`docs/results/` so the static site can load them:

- [`classification.html`](classification.html) loads
  `docs/results/classification.metrics.json` (then the CSV if JSON is missing)
- [`embeddings.html`](embeddings.html) loads
  `docs/results/embedding_bootstrap/embedding.metrics.bootstrap.json`

Use `--kind embedding` or `--kind classification` to collect only one family.

## Bootstrap embedding metrics

Repeat subsampled embedding evaluation and write aggregate tables for
[`embeddings.html`](embeddings.html):

```bash
scfm-eval report "$SCFM_OUTPUT_PATH" --bootstrap \
  --output ./dashboard-metrics/embedding_bootstrap
```

Copy the bootstrap JSON/CSV into `docs/results/embedding_bootstrap/`, or use
**Load other file** in the sidebar.

## What a comparison row contains

There is one row per entry in a run's `evaluations` list. A run with no
evaluations still gets one row so embedding-only or extraction-only runs
stay visible.

Each row has the run, model, dataset, task, kind, variant, split, status,
source path, review status, and metrics. CSV metric columns are named
`metric__<name>`.

## Python API

```python
from scfm_cancer_eval.reporting import create_report_bundle

bundle = create_report_bundle(
    ["./output"],
    "./published-report",
    strict=False,
    title="Evaluation comparison",
)

print(bundle.html_path)
print(bundle.discovery.valid_count)
```
