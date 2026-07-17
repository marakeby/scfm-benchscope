from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scfm_cancer_eval.utils import exp_yaml_merge


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class ExperimentConfigMergeTests(unittest.TestCase):
    def test_composes_sections_in_documented_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_yaml(
                root / "dataset.yaml",
                {
                    "dataset": {"path": "cells.h5ad", "options": {"source": "dataset"}},
                    "shared": {"winner": "dataset", "nested": {"dataset": True}},
                },
            )
            _write_yaml(
                root / "model.yaml",
                {
                    "embedding": {"method": "mock"},
                    "shared": {"winner": "model", "nested": {"model": True}},
                },
            )
            _write_yaml(
                root / "classification.yaml",
                {
                    "classification": {"skip": True},
                    "shared": {"winner": "classification"},
                },
            )
            _write_yaml(
                root / "base.yaml",
                {
                    "qc": {"min_genes": 10},
                    "shared": {"winner": "base", "nested": {"base": True}},
                },
            )
            _write_yaml(
                root / "experiment.yaml",
                {
                    "dataset": "dataset.yaml",
                    "model": ["model.yaml"],
                    "classification": "classification.yaml",
                    "defaults": "base.yaml",
                    "run_id": "test_run",
                    "shared": {"winner": "local", "nested": {"local": True}},
                },
            )

            with patch.object(exp_yaml_merge, "PARAMS_PATH", str(root)):
                merged = exp_yaml_merge.load_merged_experiment_config("experiment.yaml")

            self.assertEqual(merged["run_id"], "test_run")
            self.assertEqual(merged["dataset"]["path"], "cells.h5ad")
            self.assertEqual(merged["embedding"]["method"], "mock")
            self.assertTrue(merged["classification"]["skip"])
            self.assertEqual(merged["qc"]["min_genes"], 10)
            self.assertEqual(merged["shared"]["winner"], "local")
            self.assertEqual(
                merged["shared"]["nested"],
                {"dataset": True, "model": True, "base": True, "local": True},
            )
            for include_key in (
                "model",
                "models",
                "datasets",
                "classifications",
                "bases",
                "defaults",
            ):
                self.assertNotIn(include_key, merged)

    def test_dot_include_is_relative_to_including_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry_dir = root / "experiments" / "nested"
            _write_yaml(entry_dir / "fragment.yaml", {"qc": {"min_cells": 3}})
            _write_yaml(
                entry_dir / "experiment.yaml",
                {"bases": "./fragment.yaml", "run_id": "relative"},
            )

            merged = exp_yaml_merge.load_merged_experiment_config(
                str(entry_dir / "experiment.yaml")
            )

            self.assertEqual(merged["run_id"], "relative")
            self.assertEqual(merged["qc"]["min_cells"], 3)

    def test_rejects_cyclic_includes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_yaml(root / "a.yaml", {"bases": "./b.yaml"})
            _write_yaml(root / "b.yaml", {"bases": "./a.yaml"})

            with self.assertRaisesRegex(ValueError, "Cyclic YAML bases detected"):
                exp_yaml_merge.load_merged_experiment_config(str(root / "a.yaml"))

    def test_loads_existing_bundled_experiment(self) -> None:
        merged = exp_yaml_merge.load_merged_experiment_config(
            "exp/pca/n50/brca_cell_type.yaml"
        )

        self.assertEqual(merged["run_id"], "pca_n50_brca_cell_type")
        self.assertEqual(merged["dataset"]["class"], "H5ADLoader")
        self.assertEqual(merged["embedding"]["method"], "PCA")
        self.assertEqual(merged["embedding"]["params"]["n_components"], 50)
        self.assertTrue(merged["classification"]["skip"])

    def test_runner_wrapper_preserves_existing_tuple_contract(self) -> None:
        from scfm_cancer_eval.run.run_exp import get_configs

        (
            run_id,
            dataset,
            qc,
            preprocessing,
            hvg,
            embedding,
            classification,
            resolved,
            task,
        ) = get_configs("exp/pca/n50/brca_cell_type.yaml")

        self.assertEqual(run_id, "pca_n50_brca_cell_type")
        self.assertEqual(dataset, resolved["dataset"])
        self.assertEqual(qc, resolved["qc"])
        self.assertEqual(preprocessing, resolved["preprocessing"])
        self.assertEqual(hvg, resolved["hvg"])
        self.assertEqual(embedding, resolved["embedding"])
        self.assertEqual(classification, resolved["classification"])
        self.assertIsNone(task)


if __name__ == "__main__":
    unittest.main()
