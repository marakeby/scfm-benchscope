from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scfm_cancer_eval import cli
from scfm_cancer_eval.onboarding import (
    FakeHost,
    ReviewError,
    ReviewOptions,
    build_execution_approval,
    execute_approved_bundle,
    prepare_approval_bundle,
    record_review,
    verify_approval_bundle,
    write_execution_approval,
)
from scfm_cancer_eval.reporting import create_report_bundle, discover_results
from scfm_cancer_eval.utils.results_json import write_results_json
from test_approval import (
    CANDIDATE_PATH,
    CREATED_AT,
    FakeLockMaterializer,
    _options,
    _planning_workspace,
)


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
                "aggregate": {"metrics": {"NMI": 0.8}},
                "folds": [],
                "artifacts": {},
                "status": "success",
                "errors": [],
            }
        ],
        "timing": {},
    }


def _completed_run(root: Path) -> Path:
    bundle = root / "approvals" / "example" / "attempt-1"
    prepare_approval_bundle(
        CANDIDATE_PATH,
        _planning_workspace(root),
        bundle,
        _options(),
        lock_materializer=FakeLockMaterializer(),
        created_at=CREATED_AT,
    )
    verified = verify_approval_bundle(bundle)
    approval = write_execution_approval(
        root / "execution-approval.json",
        build_execution_approval(
            approval_id="example-cell-model-attempt-1-approval",
            approved_at="2026-07-18T01:00:00Z",
            manifest_fingerprint=verified.manifest.fingerprint,
            bundle_path="approvals/example/attempt-1",
            identity="example-reviewer",
            method="github_pr",
            pull_request_url="https://github.com/example/scFM_eval/pull/1",
            merge_commit="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ),
    )
    run_dir = root / "runs" / "attempt-1"
    execute_approved_bundle(
        bundle,
        approval,
        run_dir,
        FakeHost(),
        now="2026-07-18T02:00:00Z",
    )
    # Replace stub results with a validated envelope for review hashing.
    write_results_json(
        str(run_dir / "output" / "results.json"),
        _payload(verified.manifest.document_id),
    )
    return run_dir


class ReviewTests(unittest.TestCase):
    def test_accept_enables_accepted_only_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _completed_run(root)
            local = root / "local" / "results.json"
            local.parent.mkdir()
            write_results_json(str(local), _payload("local-run"))

            draft = discover_results([root])
            self.assertEqual(
                {summary.review_status for summary in draft.runs},
                {"completed_unreviewed", "local"},
            )

            with self.assertRaisesRegex(ValueError, "No accepted"):
                create_report_bundle(
                    [root],
                    root / "official-empty",
                    accepted_only=True,
                )

            outcome = record_review(
                run_dir,
                ReviewOptions(
                    decision_id="example-attempt-1-accepted",
                    decision="accepted",
                    identity="scientist",
                    rationale="Metrics look scientifically sound.",
                ),
                decided_at="2026-07-18T03:00:00Z",
            )
            self.assertTrue(outcome.decision_path.is_file())
            record = json.loads(
                (run_dir / "execution-record.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["review_status"], "accepted")

            official = create_report_bundle(
                [root],
                root / "official",
                accepted_only=True,
            )
            self.assertEqual(official.discovery.valid_count, 1)
            self.assertEqual(
                official.discovery.runs[0].review_status,
                "accepted",
            )
            payload = json.loads(
                official.comparison.json_path.read_text(encoding="utf-8")
            )
            self.assertEqual(payload["schema"]["version"], "1.1.0")
            self.assertEqual(
                payload["records"][0]["review_status"],
                "accepted",
            )

    def test_needs_tuning_writes_lineage_and_blocks_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _completed_run(root)
            outcome = record_review(
                run_dir,
                ReviewOptions(
                    decision_id="example-attempt-1-tuning",
                    decision="needs_tuning",
                    identity="scientist",
                    rationale="Need a better learning rate.",
                    tuning_changes=("lower learning rate",),
                    expected_improvement="Higher NMI on held-out cells",
                    max_additional_budget_usd=2.0,
                ),
                decided_at="2026-07-18T03:00:00Z",
            )
            self.assertIsNotNone(outcome.lineage_path)
            lineage = json.loads(outcome.lineage_path.read_text(encoding="utf-8"))
            self.assertTrue(lineage["requires_new_pre_run_approval"])
            with self.assertRaisesRegex(ValueError, "No accepted"):
                create_report_bundle(
                    [root],
                    root / "official",
                    accepted_only=True,
                )

    def test_rejects_second_decision_and_publication_for_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _completed_run(root)
            record_review(
                run_dir,
                ReviewOptions(
                    decision_id="example-attempt-1-rejected",
                    decision="rejected",
                    identity="scientist",
                    rationale="Artifacts do not support the claim.",
                ),
                decided_at="2026-07-18T03:00:00Z",
            )
            with self.assertRaisesRegex(ReviewError, "already exists"):
                record_review(
                    run_dir,
                    ReviewOptions(
                        decision_id="example-attempt-1-again",
                        decision="accepted",
                        identity="scientist",
                        rationale="Changed mind.",
                    ),
                )

            other_root = Path(tempfile.mkdtemp(dir=root))
            other = _completed_run(other_root)
            with self.assertRaisesRegex(ReviewError, "publication"):
                record_review(
                    other,
                    ReviewOptions(
                        decision_id="bad-publication",
                        decision="rejected",
                        identity="scientist",
                        rationale="no",
                        include_in_reports=True,
                    ),
                )

    def test_cli_review_and_accepted_only_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _completed_run(root)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = cli.main(
                    [
                        "review",
                        "decide",
                        str(run_dir),
                        "--decision-id",
                        "cli-accept",
                        "--decision",
                        "accepted",
                        "--identity",
                        "scientist",
                        "--rationale",
                        "Looks good.",
                    ]
                )
                report_code = cli.main(
                    [
                        "report",
                        str(root / "runs"),
                        "--accepted-only",
                        "--output",
                        str(root / "published"),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(report_code, 0)
            self.assertIn("accepted", stdout.getvalue())
            self.assertIn("accepted-only", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
