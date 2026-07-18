"""Prove unapproved, mutated, over-budget, and unreviewed paths cannot pass."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scfm_cancer_eval.onboarding import (
    ApprovalError,
    ContractValidationError,
    ExecutionError,
    ExecutionManifest,
    FakeHost,
    ModelSpec,
    PlannerError,
    ReviewOptions,
    execute_approved_bundle,
    load_model_candidate,
    plan_candidate,
    prepare_approval_bundle,
    record_review,
    validate_execution_manifest,
    verify_approval_bundle,
)
from scfm_cancer_eval.reporting import create_report_bundle
from pipeline_fixtures import (
    CANDIDATE,
    complete_unreviewed_run,
    grant_bundle,
    prepare_bundle,
    read_json,
)
from test_ai_planner import FakeProvider, _ready_proposal
from test_approval import (
    CREATED_AT,
    FakeLockMaterializer,
    _options,
    _planning_workspace,
)


EXAMPLES = Path(__file__).resolve().parents[1] / "examples/models/planning"


class AdversarialGateTests(unittest.TestCase):
    def test_unapproved_bundle_cannot_execute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = prepare_bundle(root)
            with self.assertRaises(Exception):
                execute_approved_bundle(
                    bundle,
                    root / "missing-approval.json",
                    root / "run",
                    FakeHost(),
                )

    def test_fingerprint_mismatch_cannot_execute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = prepare_bundle(root)
            wrong = EXAMPLES / "execution-approval.json"
            with self.assertRaisesRegex(ExecutionError, "fingerprint"):
                execute_approved_bundle(
                    bundle,
                    wrong,
                    root / "run",
                    FakeHost(),
                )

    def test_planner_cannot_emit_lockfile_or_execution_manifest(self) -> None:
        candidate = load_model_candidate(CANDIDATE)
        for bad_path in ("pixi.lock", "execution-manifest.json"):
            proposal = _ready_proposal()
            proposal["files"].append(
                {
                    "path": bad_path,
                    "purpose": "Invented executable artifact",
                    "content": "version = 1\n",
                }
            )
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(
                    PlannerError,
                    "outside the allowed workspace paths",
                ):
                    plan_candidate(
                        candidate,
                        FakeProvider(proposal),
                        Path(tmp) / "workspace",
                    )

    def test_approval_rejects_proposal_that_already_contains_lockfile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = _planning_workspace(root)
            plan = read_json(workspace / "integration-plan.json")
            plan["generated_files"].append(
                {
                    "path": "pixi.lock",
                    "purpose": "AI invented lock",
                    "sha256": "3" * 64,
                }
            )
            (workspace / "integration-plan.json").write_text(
                json.dumps(plan),
                encoding="utf-8",
            )
            (workspace / "pixi.lock").write_text("fake\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ApprovalError,
                "must not contain an AI-generated pixi.lock",
            ):
                prepare_approval_bundle(
                    CANDIDATE,
                    workspace,
                    root / "approval",
                    _options(),
                    lock_materializer=FakeLockMaterializer(),
                    created_at=CREATED_AT,
                )

    def test_mutable_repository_revision_is_rejected(self) -> None:
        payload = read_json(EXAMPLES / "model-spec.json")
        payload["repository"] = {
            "url": payload["repository"]["url"],
            "commit": "main",
        }
        with self.assertRaises(ContractValidationError):
            ModelSpec.from_payload(payload)

    def test_invented_secret_names_must_use_env_syntax(self) -> None:
        payload = read_json(EXAMPLES / "execution-manifest.json")
        payload["permissions"]["secret_names"] = ["openai-api-key"]
        with self.assertRaisesRegex(
            ContractValidationError,
            "environment variable syntax",
        ):
            validate_execution_manifest(payload)

    def test_over_budget_manifest_cannot_be_approved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ApprovalError, "worst-case approved cost"):
                prepare_bundle(root, max_budget_usd=0.01)

    def test_altered_bundle_cannot_execute_after_grant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = prepare_bundle(root)
            approval = grant_bundle(bundle, root / "execution-approval.json")
            (bundle / "integrations/example_cell_model.py").write_text(
                "raise RuntimeError('mutated')\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ApprovalError, "checksum changed"):
                execute_approved_bundle(
                    bundle,
                    approval,
                    root / "run",
                    FakeHost(),
                )

    def test_executor_does_not_exceed_approved_retry_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = prepare_bundle(root, max_attempts=2)
            approval = grant_bundle(bundle, root / "execution-approval.json")
            host = FakeHost(fail_step_prefix="scfm-eval")
            with self.assertRaisesRegex(ExecutionError, "failed after 2"):
                execute_approved_bundle(
                    bundle,
                    approval,
                    root / "run",
                    host,
                )
            evaluate_calls = [
                command
                for _, command in host.commands
                if command[:2] == ("scfm-eval", "run")
            ]
            self.assertEqual(len(evaluate_calls), 2)

    def test_unreviewed_and_rejected_runs_cannot_publish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, run_dir = complete_unreviewed_run(root)
            with self.assertRaisesRegex(ValueError, "No accepted"):
                create_report_bundle(
                    [root],
                    root / "official-unreviewed",
                    accepted_only=True,
                )

            record_review(
                run_dir,
                ReviewOptions(
                    decision_id="example-rejected",
                    decision="rejected",
                    identity="scientist",
                    rationale="Does not meet scientific bar.",
                ),
                decided_at="2026-07-18T03:00:00Z",
            )
            with self.assertRaisesRegex(ValueError, "No accepted"):
                create_report_bundle(
                    [root],
                    root / "official-rejected",
                    accepted_only=True,
                )

    def test_verify_rejects_extra_unreviewed_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = prepare_bundle(root)
            (bundle / "sneaky.sh").write_text("echo hi\n", encoding="utf-8")
            with self.assertRaisesRegex(ApprovalError, "unexpected file set"):
                verify_approval_bundle(bundle)

    def test_execution_manifest_rejects_reordered_steps(self) -> None:
        payload = read_json(EXAMPLES / "execution-manifest.json")
        payload["steps"] = list(reversed(payload["steps"]))
        with self.assertRaisesRegex(
            ContractValidationError,
            "approved execution order",
        ):
            ExecutionManifest.from_payload(payload)


if __name__ == "__main__":
    unittest.main()
