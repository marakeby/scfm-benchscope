# Comparing and visualizing results

Every completed evaluation writes a validated `results.json`. Reporting reads
those files directly, so it works for runs created through YAML, the Python API,
Pixi, or Docker.

## Build a report for an output root

```bash
scfm-eval report "$SCFM_OUTPUT_PATH"
```

By default this creates `$SCFM_OUTPUT_PATH/report/` containing:

- `report.html`: self-contained interactive report
- `comparison.json`: versioned structured comparison records
- `comparison.csv`: flat records for spreadsheets and analysis tools

Open `report.html` directly in a browser. It does not require a server or an
internet connection. The report can filter by search text, dataset, and
evaluation kind; table headers sort the records, and the metric selector
visualizes aggregate metric values.

Choose another destination or title when publishing a report:

```bash
scfm-eval report ./output \
  --output ./published-report \
  --title "BRCA model comparison"
```

Draft reports include every valid run and label each with `review_status`.
Official publication keeps only scientifically accepted runs:

```bash
scfm-eval report ./output \
  --accepted-only \
  --output ./published-report \
  --title "Accepted BRCA model comparison"
```

See [Scientific review](scientific-review.md) for the post-run decision gate.

## Compare selected runs

Pass individual `results.json` files, run directories, or a mixture:

```bash
scfm-eval compare \
  output/api/model_a/results.json \
  output/api/model_b \
  --output ./model-comparison
```

Directories are searched recursively. Repeated paths are deduplicated.

## Invalid results

The default mode skips missing or invalid files, keeps valid runs, and records
each problem in `comparison.json` and `report.html`. Use `--strict` for
automation that should fail when any requested result is invalid:

```bash
scfm-eval compare run-a/results.json run-b/results.json --strict
```

The command always fails if it cannot find at least one valid result.

## Comparison record shape

There is one comparison record per entry in a run's `evaluations` array. A run
without evaluation entries receives one fallback record so that successful
embedding-only or extraction-only runs remain visible.

Records carry the run, model, dataset, task, evaluation kind, variant, split,
status, source path, and aggregate metrics. CSV metric columns use the
`metric__<name>` prefix and are sorted for stable downstream imports. The JSON
export uses the `scfm_eval.comparison` v1.1.0 contract and includes
`review_status` on every record.

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
