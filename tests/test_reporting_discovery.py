from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scfm_cancer_eval.reporting import (
    ResultDiscoveryError,
    discover_results,
)
from scfm_cancer_eval.utils.results_json import write_results_json


def _payload(
    run_id: str,
    *,
    model: str,
    dataset: str,
    task: str,
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
            "dataset": {"path": dataset, "label_key": task},
            "task": {},
        },
        "artifacts": {},
        "evaluations": [],
        "timing": {},
    }


class ResultDiscoveryTests(unittest.TestCase):
    def test_discovers_valid_runs_recursively_in_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "z" / "results.json"
            second = root / "a" / "nested" / "results.json"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            write_results_json(
                str(first),
                _payload(
                    "run-b",
                    model="model-b",
                    dataset="dataset-b.h5ad",
                    task="cell_type",
                ),
            )
            write_results_json(
                str(second),
                _payload(
                    "run-a",
                    model="model-a",
                    dataset="dataset-a.h5ad",
                    task="subtype",
                ),
            )

            discovered = discover_results([root])

            self.assertEqual(
                [summary.run_id for summary in discovered.runs],
                ["run-a", "run-b"],
            )
            self.assertEqual(discovered.runs[0].model_id, "model-a")
            self.assertEqual(
                discovered.runs[0].dataset_path,
                "dataset-a.h5ad",
            )
            self.assertEqual(discovered.runs[0].task_id, "subtype")
            self.assertEqual(discovered.valid_count, 2)
            self.assertEqual(discovered.issues, ())

    def test_reports_invalid_and_missing_inputs_without_losing_valid_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "valid" / "results.json"
            invalid = root / "invalid" / "results.json"
            valid.parent.mkdir(parents=True)
            invalid.parent.mkdir(parents=True)
            write_results_json(
                str(valid),
                _payload(
                    "valid",
                    model="mock",
                    dataset="tiny.h5ad",
                    task="label",
                ),
            )
            invalid.write_text('{"schema": {}}', encoding="utf-8")

            discovered = discover_results([root, root / "missing"])

            self.assertEqual([run.run_id for run in discovered.runs], ["valid"])
            self.assertEqual(len(discovered.issues), 2)
            self.assertTrue(
                any("path does not exist" in issue.message for issue in discovered.issues)
            )
            self.assertTrue(
                any("ResultsValidationError" in issue.message for issue in discovered.issues)
            )

    def test_strict_mode_raises_aggregated_discovery_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing"

            with self.assertRaises(ResultDiscoveryError) as raised:
                discover_results([missing], strict=True)

            self.assertEqual(len(raised.exception.issues), 1)
            self.assertIn("path does not exist", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
