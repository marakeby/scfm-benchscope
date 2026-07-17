from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scfm_cancer_eval import (
    EvaluationModelConfig,
    EvaluationOptions,
    ModelAdapter,
    evaluate,
)
from scfm_cancer_eval import api


class FakeAdapter:
    method = "new_model"
    output_key = "X_new_model"

    def fit_transform(self, loader):
        embedding = np.ones((loader.adata.n_obs, 2), dtype=np.float32)
        loader.adata.obsm[self.output_key] = embedding
        return embedding


class PublicApiTests(unittest.TestCase):
    def test_direct_adapter_builds_embedding_only_config(self) -> None:
        adapter = FakeAdapter()

        config = api._build_evaluation_config(
            model=adapter,
            experiment=None,
            dataset={
                "dataset": {"path": "cells.h5ad"},
                "qc": {"skip": True},
                "preprocessing": {"skip": True},
            },
            task="embedding",
            options=EvaluationOptions(),
        )

        self.assertIsInstance(adapter, ModelAdapter)
        self.assertEqual(config["embedding"]["method"], "new_model")
        self.assertEqual(config["embedding"]["output_key"], "X_new_model")
        self.assertTrue(config["embedding"]["eval"])
        self.assertTrue(config["classification"]["skip"])
        self.assertEqual(config["run_id"], "new_model_evaluation")

    def test_serializable_model_config_uses_import_path(self) -> None:
        model = EvaluationModelConfig(
            model_id="example",
            adapter="example_package.adapters.ExampleAdapter",
            params={"checkpoint": "/models/example"},
            output_key="X_example",
        )

        embedding = model.to_embedding_config(visualize=True)

        self.assertEqual(embedding["module"], "example_package.adapters")
        self.assertEqual(embedding["class"], "ExampleAdapter")
        self.assertEqual(embedding["params"]["checkpoint"], "/models/example")
        self.assertEqual(embedding["output_key"], "X_example")
        self.assertTrue(embedding["viz"])

    def test_evaluate_injects_adapter_and_returns_run_paths(self) -> None:
        adapter = FakeAdapter()

        class FakeExperiment:
            last_instance = None

            def __init__(self, config_path, **kwargs):
                self.config_path = config_path
                self.kwargs = kwargs
                self.run_id = kwargs["resolved_config"]["run_id"]
                self.save_dir = str(Path(kwargs["output_dir"]) / self.run_id)
                self.ran = False
                self.reported = False
                FakeExperiment.last_instance = self

            def run(self):
                self.ran = True

            def _write_standard_reports(self):
                self.reported = True

        seeded = []
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(
                api,
                "_runner_components",
                return_value=(FakeExperiment, seeded.append),
            ):
                result = evaluate(
                    model=adapter,
                    dataset={"dataset": {"path": "cells.h5ad"}},
                    output_dir=tmp,
                )

        run = FakeExperiment.last_instance
        self.assertIsNotNone(run)
        self.assertIs(run.kwargs["model_adapter"], adapter)
        self.assertTrue(run.ran)
        self.assertTrue(run.reported)
        self.assertEqual(seeded, [42])
        self.assertEqual(result.run_id, "new_model_evaluation")
        self.assertEqual(result.results_path.name, "results.json")

    def test_requires_exactly_one_config_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            evaluate(model=FakeAdapter())
        with self.assertRaisesRegex(ValueError, "exactly one"):
            evaluate(
                model=FakeAdapter(),
                dataset={"dataset": {}},
                experiment={"dataset": {}},
            )

    def test_runner_extracts_with_injected_adapter(self) -> None:
        import anndata as ad

        from scfm_cancer_eval.run.run_exp import Experiment

        class Loader:
            def __init__(self):
                self.adata = ad.AnnData(
                    X=np.arange(12, dtype=np.float32).reshape(4, 3)
                )

        class Logger:
            def info(self, *args, **kwargs):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            experiment = Experiment.__new__(Experiment)
            experiment.log = Logger()
            experiment.feat_config = {
                "method": "new_model",
                "viz": False,
                "eval": True,
                "params": {},
            }
            experiment.model_adapter = FakeAdapter()
            experiment.save_dir = tmp
            experiment.loader = Loader()
            experiment.embedding_key = None

            experiment.extract_embeddings()

            self.assertEqual(experiment.embedding_key, "X_new_model")
            self.assertEqual(experiment.embedding.shape, (4, 2))
            self.assertTrue((Path(tmp) / "data.h5ad").is_file())


if __name__ == "__main__":
    unittest.main()
