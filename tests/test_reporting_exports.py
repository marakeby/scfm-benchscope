from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scfm_cancer_eval.reporting import (
    COMPARISON_SCHEMA_NAME,
    build_comparison_records,
    discover_results,
    write_comparison_exports,
)
from scfm_cancer_eval.utils.results_json import write_results_json


def _payload(
    run_id: str,
    *,
    model: str,
    evaluations: list[dict],
) -> dict:
    return {
        "schema": {"name": "scfm_eval.results", "version": "1.1.0"},
        "run": {
            "run_id": run_id,
            "started_at": "2026-07-17T00:00:00Z",
            "finished_at": "2026-07-17T00:01:00Z",
            "status": "success",
            "errors": [],
        },
        "provenance": {},
        "inputs": {
            "embedding": {"method": model},
            "dataset": {
                "path": "shared.h5ad",
                "label_key": "cell_type",
            },
            "task": {},
        },
        "artifacts": {},
        "evaluations": evaluations,
        "timing": {},
    }


def _evaluation(kind: str, metrics: dict) -> dict:
    return {
        "kind": kind,
        "variant": "base",
        "split": "all",
        "target": {},
        "aggregate": {"metrics": metrics},
        "folds": [],
        "primary": None,
        "artifacts": {},
        "status": "success",
        "errors": [],
    }


class ComparisonExportTests(unittest.TestCase):
    def test_builds_one_record_per_evaluation_and_run_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first" / "results.json"
            second = root / "second" / "results.json"
            first.parent.mkdir()
            second.parent.mkdir()
            write_results_json(
                str(first),
                _payload(
                    "run-a",
                    model="model-a",
                    evaluations=[
                        _evaluation("classification", {"accuracy": 0.8}),
                        _evaluation(
                            "embedding",
                            {"NMI": 0.5, "details": {"neighbors": 15}},
                        ),
                    ],
                ),
            )
            write_results_json(
                str(second),
                _payload("run-b", model="model-b", evaluations=[]),
            )

            records = build_comparison_records(discover_results([root]))

            self.assertEqual(len(records), 3)
            self.assertEqual(
                [record.evaluation_kind for record in records],
                ["classification", "embedding", "run"],
            )
            self.assertEqual(records[1].metrics["NMI"], 0.5)

    def test_writes_stable_json_and_flat_csv_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_path = root / "runs" / "one" / "results.json"
            results_path.parent.mkdir(parents=True)
            write_results_json(
                str(results_path),
                _payload(
                    "run-a",
                    model="model-a",
                    evaluations=[
                        _evaluation(
                            "embedding",
                            {"NMI": 0.5, "ARI": 0.4},
                        )
                    ],
                ),
            )
            invalid = root / "runs" / "broken" / "results.json"
            invalid.parent.mkdir(parents=True)
            invalid.write_text("{}", encoding="utf-8")
            discovered = discover_results([root / "runs"])

            artifacts = write_comparison_exports(
                discovered,
                root / "report",
            )

            payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
            with artifacts.csv_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(payload["schema"]["name"], COMPARISON_SCHEMA_NAME)
            self.assertEqual(payload["summary"]["run_count"], 1)
            self.assertEqual(payload["summary"]["issue_count"], 1)
            self.assertEqual(artifacts.record_count, 1)
            self.assertEqual(rows[0]["model_id"], "model-a")
            self.assertEqual(rows[0]["metric__ARI"], "0.4")
            self.assertEqual(rows[0]["metric__NMI"], "0.5")
            self.assertEqual(
                list(rows[0]).index("metric__ARI")
                < list(rows[0]).index("metric__NMI"),
                True,
            )


if __name__ == "__main__":
    unittest.main()
