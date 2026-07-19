from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scfm_cancer_eval import cli


def _write_embedding_run(root: Path, *, model: str, exp: str = "brca_cell_type") -> Path:
    run_dir = root / f"{exp}_{model}_{exp}"
    run_dir.mkdir(parents=True)
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "run_id": f"{model}_{exp}",
                "config_path": f"yaml/exp/{exp}.yaml",
            }
        ),
        encoding="utf-8",
    )
    metrics = pd.DataFrame({"X_pca": [0.5, 0.6]}, index=["NMI_cluster/label", "ARI_cluster/label"])
    metrics_path = run_dir / "embedding_metrics.csv"
    metrics.to_csv(metrics_path)
    return run_dir


def _write_classification_run(
    root: Path, *, model: str, exp: str = "brca_subtype"
) -> Path:
    run_dir = root / f"{exp}_{model}_{exp}"
    cv_dir = run_dir / "cv"
    cv_dir.mkdir(parents=True)
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "run_id": f"{model}_{exp}",
                "config_path": f"yaml/exp/{exp}.yaml",
            }
        ),
        encoding="utf-8",
    )
    metrics = pd.DataFrame(
        {
            "Metrics": ["AUPRC", "AUPRC", "AUC", "AUC"],
            "fold": [0, 1, 0, 1],
            "randomforest": [0.7, 0.8, 0.75, 0.85],
        }
    )
    metrics.to_csv(cv_dir / "mil_cv_metrics.csv", index=False)
    return run_dir


class CollectMetricsCliTests(unittest.TestCase):
    def test_report_collect_writes_embedding_and_classification_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "exp"
            _write_embedding_run(root, model="pca_n50")
            _write_classification_run(root, model="pca_n50")
            output_dir = Path(tmp) / "report"
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "report",
                        str(root),
                        "--collect",
                        "--output",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            emb_csv = output_dir / "embedding.metrics.csv"
            emb_json = output_dir / "embedding.metrics.json"
            cls_csv = output_dir / "classification.metrics.csv"
            cls_json = output_dir / "classification.metrics.json"
            self.assertTrue(emb_csv.is_file())
            self.assertTrue(emb_json.is_file())
            self.assertTrue(cls_csv.is_file())
            self.assertTrue(cls_json.is_file())

            emb = pd.read_csv(emb_csv)
            self.assertEqual(len(emb), 1)
            self.assertEqual(emb.loc[0, "model"], "pca_n50")
            self.assertIn("NMI_cluster/label", emb.columns)

            cls = pd.read_csv(cls_csv)
            self.assertGreaterEqual(len(cls), 2)
            self.assertEqual(cls.loc[0, "strategy"], "MIL")
            self.assertIn("AUPRC", cls.columns)

            payload = json.loads(cls_json.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["schema"]["name"], "scfm_eval.classification_metrics"
            )
            self.assertGreater(payload["row_count"], 0)

    def test_report_collect_kind_embedding_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "exp"
            _write_embedding_run(root, model="hvg_seurat_4096")
            output_dir = Path(tmp) / "out"

            exit_code = cli.main(
                [
                    "report",
                    str(root),
                    "--collect",
                    "--kind",
                    "embedding",
                    "--output",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "embedding.metrics.csv").is_file())
            self.assertFalse((output_dir / "classification.metrics.csv").is_file())


if __name__ == "__main__":
    unittest.main()
