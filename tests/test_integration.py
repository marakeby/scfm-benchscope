from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from scfm_cancer_eval import EvaluationOptions, RunResult, evaluate


class TinyAdapter:
    method = "tiny"
    output_key = "X_tiny"

    def fit_transform(self, loader):
        values = np.asarray(loader.adata.X, dtype=np.float32)
        embedding = values[:, :2]
        loader.adata.obsm[self.output_key] = embedding
        return embedding


def _write_tiny_dataset(path: Path) -> None:
    values = np.arange(32, dtype=np.float32).reshape(8, 4)
    obs = pd.DataFrame(
        {
            "cell_type": ["A", "A", "B", "B", "A", "B", "A", "B"],
            "donor": ["d1", "d1", "d1", "d1", "d2", "d2", "d2", "d2"],
        },
        index=[f"cell-{index}" for index in range(8)],
    )
    var = pd.DataFrame(index=[f"gene-{index}" for index in range(4)])
    ad.AnnData(X=values, obs=obs, var=var).write_h5ad(path)


def _dataset_config(path: Path) -> dict:
    return {
        "dataset": {
            "module": "data.data_loader",
            "class": "H5ADLoader",
            "path": str(path),
            "load_raw": False,
            "layer_name": "X",
            "label_key": "cell_type",
            "batch_key": "donor",
        },
        "qc": {"skip": True},
        "preprocessing": {"skip": True},
    }


class EvaluationIntegrationTests(unittest.TestCase):
    def test_public_api_runs_direct_adapter_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "tiny.h5ad"
            output_root = root / "output"
            _write_tiny_dataset(dataset_path)

            result = evaluate(
                model=TinyAdapter(),
                dataset=_dataset_config(dataset_path),
                output_dir=output_root,
                options=EvaluationOptions(
                    seed=17,
                    evaluate_embedding=False,
                    visualize=False,
                ),
            )

            self.assertEqual(result.status, "success")
            self.assertEqual(result.run_id, "tiny_evaluation")
            self.assertEqual(
                result.payload["inputs"]["embedding"]["output_key"],
                "X_tiny",
            )
            self.assertTrue(result.results_path.is_file())
            self.assertTrue(result.metrics_path.is_file())
            self.assertTrue((result.output_dir / "resolved_config.yaml").is_file())
            self.assertTrue((output_root / "metrics_runs.csv").is_file())

            embedded = ad.read_h5ad(result.output_dir / "data.h5ad")
            self.assertEqual(embedded.obsm["X_tiny"].shape, (8, 2))
            np.testing.assert_array_equal(
                embedded.obsm["X_tiny"],
                np.arange(32, dtype=np.float32).reshape(8, 4)[:, :2],
            )

    def test_legacy_yaml_cli_runs_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "tiny.h5ad"
            output_root = root / "output"
            config_path = root / "mock.yaml"
            _write_tiny_dataset(dataset_path)

            config = _dataset_config(dataset_path)
            config.update(
                {
                    "run_id": "mock_cli_smoke",
                    "embedding": {
                        "method": "mock",
                        "module": "features.mock_extractor",
                        "class": "MockEmbeddingExtractor",
                        "output_key": "X_mock",
                        "viz": False,
                        "eval": False,
                        "params": {"out_dim": 3, "seed": 5},
                    },
                    "classification": {"skip": True},
                }
            )
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )

            repo_root = Path(__file__).resolve().parents[1]
            env = os.environ.copy()
            env["SCFM_OUTPUT_PATH"] = str(output_root)
            env["PYTHONPATH"] = os.pathsep.join(
                filter(
                    None,
                    [str(repo_root / "src"), env.get("PYTHONPATH", "")],
                )
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scfm_cancer_eval.run.run_exp",
                    str(config_path),
                    "--seed",
                    "23",
                ],
                cwd=repo_root,
                env=env,
                text=True,
                capture_output=True,
                timeout=90,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )

            result_paths = list(output_root.rglob("results.json"))
            self.assertEqual(len(result_paths), 1)
            result = RunResult.from_path(result_paths[0])
            self.assertEqual(result.run_id, "mock_cli_smoke")
            self.assertEqual(result.status, "success")
            self.assertEqual(
                result.payload["inputs"]["embedding"]["class"],
                "MockEmbeddingExtractor",
            )
            self.assertTrue((result.output_dir / "timing.csv").is_file())

            embedded = ad.read_h5ad(result.output_dir / "data.h5ad")
            self.assertEqual(embedded.obsm["X_mock"].shape, (8, 3))


if __name__ == "__main__":
    unittest.main()
