from __future__ import annotations

import json
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

from scfm_cancer_eval import ResultsValidationError, RunResult
from scfm_cancer_eval.utils.results_json import (
    RESULTS_SCHEMA_NAME,
    RESULTS_SCHEMA_VERSION,
    read_results_json,
    validate_results_payload,
    write_results_json,
)


def _valid_payload(run_id: str = "test-run") -> dict:
    return {
        "schema": {
            "name": RESULTS_SCHEMA_NAME,
            "version": RESULTS_SCHEMA_VERSION,
        },
        "run": {
            "run_id": run_id,
            "started_at": "2026-07-17T00:00:00Z",
            "finished_at": "2026-07-17T00:01:00Z",
            "status": "success",
            "errors": [],
        },
        "provenance": {},
        "inputs": {},
        "artifacts": {"run_dir": "/tmp/test-run", "files": []},
        "evaluations": [
            {
                "kind": "embedding",
                "variant": "base",
                "split": "all",
                "target": {"embedding_key": "X_test"},
                "aggregate": {"metrics": {"NMI": 0.5}},
                "folds": [],
                "primary": None,
                "artifacts": {},
                "status": "success",
                "errors": [],
            }
        ],
        "timing": {},
    }


class ResultsContractTests(unittest.TestCase):
    def test_round_trip_returns_validated_run_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_path = Path(tmp) / "results.json"
            write_results_json(str(results_path), _valid_payload())

            loaded = read_results_json(str(results_path))
            result = RunResult.from_path(results_path, expected_run_id="test-run")

            self.assertEqual(loaded["run"]["status"], "success")
            self.assertEqual(result.status, "success")
            self.assertEqual(result.evaluations[0]["kind"], "embedding")
            self.assertEqual(result.output_dir, Path(tmp))

    def test_rejects_unsupported_or_incomplete_payloads(self) -> None:
        wrong_version = _valid_payload()
        wrong_version["schema"]["version"] = "9.0.0"
        with self.assertRaisesRegex(ResultsValidationError, "schema.version"):
            validate_results_payload(wrong_version)

        missing_run = _valid_payload()
        del missing_run["run"]
        with self.assertRaisesRegex(ResultsValidationError, r"\$\.run"):
            validate_results_payload(missing_run)

    def test_invalid_payload_does_not_replace_existing_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_path = Path(tmp) / "results.json"
            original = '{"existing": true}\n'
            results_path.write_text(original, encoding="utf-8")

            with self.assertRaises(ResultsValidationError):
                write_results_json(str(results_path), {"schema": {}})

            self.assertEqual(results_path.read_text(encoding="utf-8"), original)

    def test_serialization_failure_cleans_up_and_preserves_result(self) -> None:
        class NotJsonSerializable:
            pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_path = root / "results.json"
            original = json.dumps(_valid_payload())
            results_path.write_text(original, encoding="utf-8")
            payload = _valid_payload()
            payload["timing"]["invalid"] = NotJsonSerializable()

            with self.assertRaises((TypeError, ValueError)):
                write_results_json(str(results_path), payload)

            self.assertEqual(results_path.read_text(encoding="utf-8"), original)
            self.assertEqual(list(root.glob(".results.json.*.tmp")), [])

    def test_packaged_json_schema_matches_runtime_contract(self) -> None:
        schema_text = (
            files("scfm_cancer_eval")
            .joinpath("schemas/results-v1.1.0.json")
            .read_text(encoding="utf-8")
        )
        schema = json.loads(schema_text)

        self.assertEqual(
            schema["properties"]["schema"]["properties"]["name"]["const"],
            RESULTS_SCHEMA_NAME,
        )
        self.assertEqual(
            schema["properties"]["schema"]["properties"]["version"]["const"],
            RESULTS_SCHEMA_VERSION,
        )

    def test_runner_propagates_required_result_write_failure(self) -> None:
        from scfm_cancer_eval.run.run_exp import Experiment

        class Logger:
            def info(self, *args, **kwargs):
                return None

            def exception(self, *args, **kwargs):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            experiment = Experiment.__new__(Experiment)
            experiment.save_dir = tmp
            experiment.output_root = tmp
            experiment.run_id = "failed-result"
            experiment.embedding_key = None
            experiment.embedding_metrics = None
            experiment.classification_metrics = None
            experiment.resolved_config = {}
            experiment.data_config = {}
            experiment.feat_config = {}
            experiment.task_config = None
            experiment.rng_seed = 42
            experiment.run_summary = {
                "config_path": None,
                "started_at": "2026-07-17T00:00:00Z",
            }
            experiment.log = Logger()

            with patch(
                "scfm_cancer_eval.run.run_exp.write_results_json",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    experiment._write_standard_reports()

            self.assertFalse((Path(tmp) / "metrics_runs.csv").exists())


if __name__ == "__main__":
    unittest.main()
