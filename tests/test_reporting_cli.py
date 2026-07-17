from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scfm_cancer_eval import cli
from scfm_cancer_eval.utils.results_json import write_results_json


def _payload(run_id: str, model: str) -> dict:
    return {
        "schema": {"name": "scfm_eval.results", "version": "1.1.0"},
        "run": {
            "run_id": run_id,
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
        },
        "artifacts": {},
        "evaluations": [],
        "timing": {},
    }


class ReportingCliTests(unittest.TestCase):
    def test_report_command_discovers_output_root_and_writes_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_path = root / "api" / "run-a" / "results.json"
            results_path.parent.mkdir(parents=True)
            write_results_json(str(results_path), _payload("run-a", "model-a"))
            output_dir = root / "published"
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "report",
                        str(root),
                        "--output",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "report.html").is_file())
            self.assertTrue((output_dir / "comparison.json").is_file())
            self.assertTrue((output_dir / "comparison.csv").is_file())
            self.assertIn("Discovered 1 valid run(s)", stdout.getvalue())

    def test_compare_command_accepts_selected_result_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_paths = []
            for run_id, model in (("run-a", "model-a"), ("run-b", "model-b")):
                path = root / run_id / "results.json"
                path.parent.mkdir()
                write_results_json(str(path), _payload(run_id, model))
                result_paths.append(path)
            output_dir = root / "comparison"

            exit_code = cli.main(
                [
                    "compare",
                    *(str(path) for path in result_paths),
                    "--output",
                    str(output_dir),
                    "--strict",
                    "--title",
                    "Selected models",
                ]
            )

            self.assertEqual(exit_code, 0)
            report = (output_dir / "report.html").read_text(encoding="utf-8")
            self.assertIn("<title>Selected models</title>", report)
            self.assertIn("run-a", report)
            self.assertIn("run-b", report)

    def test_legacy_yaml_invocation_and_explicit_run_are_forwarded(self) -> None:
        with patch("scfm_cancer_eval.run.run_exp.main") as run_main:
            self.assertEqual(cli.main(["experiment.yaml", "--seed", "7"]), 0)
            run_main.assert_called_once_with(["experiment.yaml", "--seed", "7"])

        with patch("scfm_cancer_eval.run.run_exp.main") as run_main:
            self.assertEqual(cli.main(["run", "experiment.yaml"]), 0)
            run_main.assert_called_once_with(["experiment.yaml"])


if __name__ == "__main__":
    unittest.main()
