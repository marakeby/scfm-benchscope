from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.validate_approval_bundles as approval_ci
from scfm_cancer_eval import cli
from scfm_cancer_eval.onboarding import (
    ApprovalError,
    ApprovalOptions,
    IntegrationPlan,
    ModelSpec,
    load_model_candidate,
    prepare_approval_bundle,
    verify_approval_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = ROOT / "examples/models/candidates/scgpt.json"
PLANNING_EXAMPLES = ROOT / "examples/models/planning"
CREATED_AT = "2026-07-18T00:00:00Z"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _planning_workspace(root: Path) -> Path:
    workspace = root / "planning"
    workspace.mkdir()
    candidate = load_model_candidate(CANDIDATE_PATH)

    model_payload = _json(PLANNING_EXAMPLES / "model-spec.json")
    model_payload["candidate"] = {
        "candidate_id": candidate.candidate_id,
        "fingerprint": candidate.fingerprint,
    }
    model_spec = ModelSpec.from_payload(model_payload)

    files = {
        "pixi.toml": (
            '[workspace]\nname = "example-cell-model"\n'
            'channels = ["conda-forge"]\nplatforms = ["linux-64"]\n'
        ),
        "integrations/example_cell_model.py": (
            "class ExampleCellModelAdapter:\n"
            '    output_key = "X_example"\n'
        ),
        "experiments/example_cell_model.yaml": (
            "run_id: example-cell-model\n"
        ),
    }
    generated = []
    for relative_path, content in files.items():
        path = workspace / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        generated.append(
            {
                "path": relative_path,
                "purpose": f"Generated {relative_path}",
                "sha256": _sha256(content),
            }
        )

    plan_payload = _json(PLANNING_EXAMPLES / "integration-plan.json")
    plan_payload["candidate_fingerprint"] = candidate.fingerprint
    plan_payload["model_spec_fingerprint"] = model_spec.fingerprint
    plan_payload["generated_files"] = generated
    integration_plan = IntegrationPlan.from_payload(plan_payload)

    (workspace / "model-spec.json").write_text(
        json.dumps(model_spec.to_dict()),
        encoding="utf-8",
    )
    (workspace / "integration-plan.json").write_text(
        json.dumps(integration_plan.to_dict()),
        encoding="utf-8",
    )
    (workspace / "proposal.json").write_text(
        json.dumps({"research_notes": ["Static test proposal."]}),
        encoding="utf-8",
    )
    (workspace / "planning-status.json").write_text(
        json.dumps({"status": "ready"}),
        encoding="utf-8",
    )
    return workspace


class FakeLockMaterializer:
    def materialize(self, pixi_toml: Path, lock_path: Path) -> None:
        if not pixi_toml.is_file():
            raise AssertionError("pixi.toml was not copied before locking")
        lock_path.write_text(
            "version: 6\nenvironments: {}\n",
            encoding="utf-8",
        )


def _options(**overrides) -> ApprovalOptions:
    values = {
        "manifest_id": "example-cell-model-attempt-1",
        "gpu_type": "A10G",
        "gpu_count": 1,
        "disk_gb": 50,
        "max_runtime_minutes": 60,
        "hourly_rate_usd": 2,
        "max_budget_usd": 4,
        "max_attempts": 2,
    }
    values.update(overrides)
    return ApprovalOptions(**values)


class ApprovalTests(unittest.TestCase):
    def test_prepare_materializes_and_verifies_review_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = _planning_workspace(root)
            output = root / "approval"

            bundle = prepare_approval_bundle(
                CANDIDATE_PATH,
                workspace,
                output,
                _options(),
                lock_materializer=FakeLockMaterializer(),
                created_at=CREATED_AT,
            )
            verified = verify_approval_bundle(output)
            manifest = verified.manifest.to_dict()

            self.assertEqual(bundle.root, output)
            self.assertEqual(
                manifest["environment"]["lock_sha256"],
                _sha256("version: 6\nenvironments: {}\n"),
            )
            self.assertIn(
                "example.org",
                manifest["permissions"]["network_hosts"],
            )
            self.assertEqual(manifest["resources"]["max_budget_usd"], 4)
            self.assertEqual(
                _json(output / "approval-request.json")["status"],
                "pending_human_review",
            )

    def test_rejects_budget_below_worst_case_attempt_cost(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = _planning_workspace(root)
            output = root / "approval"

            with self.assertRaisesRegex(
                ApprovalError,
                "worst-case approved cost",
            ):
                prepare_approval_bundle(
                    CANDIDATE_PATH,
                    workspace,
                    output,
                    _options(max_budget_usd=3),
                    lock_materializer=FakeLockMaterializer(),
                    created_at=CREATED_AT,
                )
            self.assertFalse(output.exists())

    def test_verify_rejects_changed_generated_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "approval"
            prepare_approval_bundle(
                CANDIDATE_PATH,
                _planning_workspace(root),
                output,
                _options(),
                lock_materializer=FakeLockMaterializer(),
                created_at=CREATED_AT,
            )
            (output / "integrations/example_cell_model.py").write_text(
                "# changed after review\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ApprovalError,
                "checksum changed",
            ):
                verify_approval_bundle(output)

    def test_verify_rejects_unreviewed_extra_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "approval"
            prepare_approval_bundle(
                CANDIDATE_PATH,
                _planning_workspace(root),
                output,
                _options(),
                lock_materializer=FakeLockMaterializer(),
                created_at=CREATED_AT,
            )
            (output / "run-extra.py").write_text(
                "raise RuntimeError\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ApprovalError,
                "unexpected file set",
            ):
                verify_approval_bundle(output)

    def test_pixi_materializer_uses_lock_only_command(self) -> None:
        from scfm_cancer_eval.onboarding.approval import PixiLockMaterializer

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "pixi.toml"
            lock = root / "pixi.lock"
            manifest.write_text("[workspace]\nname='x'\n", encoding="utf-8")

            def fake_run(command, **kwargs):
                self.assertIn("--no-install", command)
                self.assertEqual(kwargs["cwd"], root)
                lock.write_text("version: 6\n", encoding="utf-8")
                return type(
                    "Completed",
                    (),
                    {"returncode": 0, "stderr": "", "stdout": ""},
                )()

            with patch(
                "scfm_cancer_eval.onboarding.approval.subprocess.run",
                side_effect=fake_run,
            ):
                PixiLockMaterializer().materialize(manifest, lock)

    def test_cli_can_prepare_and_verify_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "approval"
            workspace = _planning_workspace(root)
            stdout = io.StringIO()

            def fake_lock(self, pixi_toml: Path, lock_path: Path) -> None:
                FakeLockMaterializer().materialize(pixi_toml, lock_path)

            with patch(
                "scfm_cancer_eval.onboarding.approval."
                "PixiLockMaterializer.materialize",
                new=fake_lock,
            ), contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "approval",
                        "prepare",
                        str(CANDIDATE_PATH),
                        str(workspace),
                        "--output",
                        str(output),
                        "--manifest-id",
                        "example-cell-model-attempt-1",
                        "--gpu-type",
                        "A10G",
                        "--gpu-count",
                        "1",
                        "--disk-gb",
                        "50",
                        "--max-runtime-minutes",
                        "60",
                        "--hourly-rate-usd",
                        "2",
                        "--max-budget-usd",
                        "4",
                        "--max-attempts",
                        "2",
                    ]
                )
                verify_code = cli.main(
                    ["approval", "verify", str(output)]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(verify_code, 0)
            self.assertIn("pending human review", stdout.getvalue())

    def test_pull_request_check_requires_one_new_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            approvals = root / "approvals"
            bundle_path = approvals / "example" / "attempt-1"
            prepare_approval_bundle(
                CANDIDATE_PATH,
                _planning_workspace(root),
                bundle_path,
                _options(),
                lock_materializer=FakeLockMaterializer(),
                created_at=CREATED_AT,
            )
            added = [
                ("A", path)
                for path in bundle_path.rglob("*")
                if path.is_file()
            ]

            with patch.object(
                approval_ci,
                "_changed_paths",
                return_value=added,
            ):
                self.assertEqual(
                    approval_ci.validate(approvals, base="base-sha"),
                    1,
                )

            with patch.object(
                approval_ci,
                "_changed_paths",
                return_value=[("M", bundle_path / "pixi.toml")],
            ), self.assertRaisesRegex(ApprovalError, "immutable"):
                approval_ci.validate(approvals, base="base-sha")


if __name__ == "__main__":
    unittest.main()
