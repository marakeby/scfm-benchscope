from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from scfm_cancer_eval.reporting import (
    discover_results,
    render_html_report,
    write_comparison_exports,
    write_html_report,
)
from scfm_cancer_eval.utils.results_json import write_results_json


def _payload(run_id: str) -> dict:
    return {
        "schema": {"name": "scfm_eval.results", "version": "1.1.0"},
        "run": {
            "run_id": run_id,
            "status": "success",
            "errors": [],
        },
        "provenance": {},
        "inputs": {
            "embedding": {"method": "tiny"},
            "dataset": {
                "path": "tiny.h5ad",
                "label_key": "cell_type",
            },
        },
        "artifacts": {},
        "evaluations": [
            {
                "kind": "embedding",
                "variant": "base",
                "split": "all",
                "target": {},
                "aggregate": {"metrics": {"NMI": 0.75}},
                "folds": [],
                "artifacts": {},
                "status": "success",
                "errors": [],
            }
        ],
        "timing": {},
    }


class HtmlReportTests(unittest.TestCase):
    def test_report_is_self_contained_and_embeds_safe_valid_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_path = root / "run" / "results.json"
            results_path.parent.mkdir()
            write_results_json(
                str(results_path),
                _payload("</script><script>unsafe()</script>"),
            )
            discovery = discover_results([root])

            html = render_html_report(discovery, title="A & B <comparison>")

            self.assertIn("<title>A &amp; B &lt;comparison&gt;</title>", html)
            self.assertNotIn("</script><script>unsafe()", html)
            self.assertNotIn("https://", html)
            self.assertNotIn("http://", html)
            self.assertNotIn("<script src=", html)
            self.assertIn('id="dataset-filter"', html)
            self.assertIn('id="metric-filter"', html)
            self.assertIn('href="comparison.csv"', html)

            match = re.search(
                r'<script id="report-data" type="application/json">(.*?)</script>',
                html,
                re.DOTALL,
            )
            self.assertIsNotNone(match)
            embedded = json.loads(match.group(1))
            self.assertEqual(
                embedded["records"][0]["run_id"],
                "</script><script>unsafe()</script>",
            )
            self.assertEqual(embedded["records"][0]["metrics"]["NMI"], 0.75)

    def test_writes_report_beside_machine_readable_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_path = root / "run" / "results.json"
            results_path.parent.mkdir()
            write_results_json(str(results_path), _payload("run-a"))
            discovery = discover_results([root])
            output_dir = root / "report"

            write_comparison_exports(discovery, output_dir)
            report_path = write_html_report(discovery, output_dir)

            self.assertTrue(report_path.is_file())
            self.assertTrue((output_dir / "comparison.csv").is_file())
            self.assertTrue((output_dir / "comparison.json").is_file())
            self.assertIn("run-a", report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
