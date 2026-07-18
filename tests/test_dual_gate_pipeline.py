"""Synthetic candidate through planner, both human gates, and accepted report."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scfm_cancer_eval.onboarding import (
    FakeHost,
    ReviewOptions,
    build_execution_approval,
    execute_approved_bundle,
    load_integration_plan,
    load_model_candidate,
    load_model_spec,
    plan_candidate,
    prepare_approval_bundle,
    record_review,
    verify_approval_bundle,
    write_execution_approval,
)
from scfm_cancer_eval.reporting import create_report_bundle, discover_results
from scfm_cancer_eval.utils.results_json import write_results_json
from pipeline_fixtures import CANDIDATE, results_payload
from test_ai_planner import FakeProvider, _ready_proposal
from test_approval import CREATED_AT, FakeLockMaterializer, _options


class DualGatePipelineTests(unittest.TestCase):
    def test_synthetic_model_passes_both_human_gates(self) -> None:
        candidate = load_model_candidate(CANDIDATE)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # 1) Proposal-only planner (no install, lock, or execution).
            workspace = root / "planning" / candidate.candidate_id
            plan_outcome = plan_candidate(
                candidate,
                FakeProvider(_ready_proposal()),
                workspace,
                created_at="2026-07-17T22:00:00Z",
            )
            self.assertEqual(plan_outcome.status, "ready")
            self.assertFalse((workspace / "pixi.lock").exists())
            self.assertFalse(
                (workspace / "execution-manifest.json").exists()
            )
            model_spec = load_model_spec(plan_outcome.model_spec_path)
            integration_plan = load_integration_plan(
                plan_outcome.integration_plan_path
            )
            self.assertEqual(
                model_spec.to_dict()["candidate"]["fingerprint"],
                candidate.fingerprint,
            )
            self.assertEqual(
                integration_plan.to_dict()["model_spec_fingerprint"],
                model_spec.fingerprint,
            )

            # 2) Deterministic approval materialization + pre-run grant.
            bundle = root / "approvals" / candidate.candidate_id / "attempt-1"
            prepare_approval_bundle(
                CANDIDATE,
                workspace,
                bundle,
                _options(manifest_id="synthetic-attempt-1"),
                lock_materializer=FakeLockMaterializer(),
                created_at=CREATED_AT,
            )
            verified = verify_approval_bundle(bundle)
            self.assertTrue((bundle / "pixi.lock").is_file())
            self.assertTrue((bundle / "execution-manifest.json").is_file())
            approval = write_execution_approval(
                root / "execution-approvals" / "synthetic-attempt-1.json",
                build_execution_approval(
                    approval_id="synthetic-attempt-1-approval",
                    approved_at="2026-07-18T01:00:00Z",
                    manifest_fingerprint=verified.manifest.fingerprint,
                    bundle_path=(
                        f"approvals/{candidate.candidate_id}/attempt-1"
                    ),
                    identity="ci-reviewer",
                    method="github_pr",
                    pull_request_url=(
                        "https://github.com/example/scFM_eval/pull/20"
                    ),
                    merge_commit="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                ),
            )

            # 3) Bounded fake execution (no GPU / SSH).
            run_dir = root / "runs" / "synthetic-attempt-1"
            execution = execute_approved_bundle(
                bundle,
                approval,
                run_dir,
                FakeHost(),
                now="2026-07-18T02:00:00Z",
            )
            self.assertEqual(execution.status, "completed_unreviewed")
            write_results_json(
                str(run_dir / "output" / "results.json"),
                results_payload(verified.manifest.document_id, nmi=0.91),
            )

            draft = discover_results([root / "runs"])
            self.assertEqual(draft.valid_count, 1)
            self.assertEqual(
                draft.runs[0].review_status,
                "completed_unreviewed",
            )
            with self.assertRaisesRegex(ValueError, "No accepted"):
                create_report_bundle(
                    [root / "runs"],
                    root / "official-too-early",
                    accepted_only=True,
                )

            # 4) Post-run scientific acceptance, then official report.
            review = record_review(
                run_dir,
                ReviewOptions(
                    decision_id="synthetic-attempt-1-accepted",
                    decision="accepted",
                    identity="ci-scientist",
                    rationale=(
                        "Synthetic dual-gate rollout metrics are acceptable."
                    ),
                    promote_baseline=False,
                ),
                decided_at="2026-07-18T03:00:00Z",
            )
            self.assertEqual(
                review.decision.to_dict()["decision"],
                "accepted",
            )

            official = create_report_bundle(
                [root / "runs"],
                root / "published",
                accepted_only=True,
                title="Synthetic accepted report",
            )
            self.assertEqual(official.discovery.valid_count, 1)
            self.assertEqual(
                official.discovery.runs[0].review_status,
                "accepted",
            )
            self.assertTrue(official.html_path.is_file())
            self.assertTrue(official.comparison.json_path.is_file())
            html = official.html_path.read_text(encoding="utf-8")
            self.assertIn("accepted", html)
            self.assertIn("review-filter", html)


if __name__ == "__main__":
    unittest.main()
