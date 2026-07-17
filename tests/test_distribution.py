from __future__ import annotations

import subprocess
import sys
import unittest
from importlib.metadata import distribution
from importlib.resources import files


class DistributionTests(unittest.TestCase):
    def test_package_metadata_and_cli_aliases(self) -> None:
        installed = distribution("scfm-cancer-eval")
        scripts = {
            entry.name: entry.value
            for entry in installed.entry_points
            if entry.group == "console_scripts"
        }

        self.assertEqual(installed.version, "0.1.0")
        self.assertEqual(installed.metadata["License-Expression"], "GPL-3.0-only")
        self.assertEqual(
            scripts["scfm-cancer-eval"],
            "scfm_cancer_eval.cli:main",
        )
        self.assertEqual(
            scripts["scfm-eval"],
            "scfm_cancer_eval.cli:main",
        )

    def test_wheel_resources_are_declared(self) -> None:
        package = files("scfm_cancer_eval")

        self.assertTrue(
            package.joinpath("schemas/results-v1.1.0.json").is_file()
        )
        self.assertTrue(
            package.joinpath("yaml/exp/pca/n50/brca_cell_type.yaml").is_file()
        )

    def test_package_module_exposes_cli_help(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "scfm_cancer_eval", "--help"],
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn("Run scFM_eval experiment from YAML", completed.stdout)
        self.assertIn("report", completed.stdout)
        self.assertIn("compare", completed.stdout)


if __name__ == "__main__":
    unittest.main()
