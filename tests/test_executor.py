from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scfm_cancer_eval import cli
from scfm_cancer_eval.onboarding import (
    ExecutionError,
    FakeHost,
    build_execution_approval,
    execute_approved_bundle,
    prepare_approval_bundle,
    verify_approval_bundle,
    write_execution_approval,
)
from test_approval import (
    CANDIDATE_PATH,
    CREATED_AT,
    FakeLockMaterializer,
    _options,
    _planning_workspace,
)


def _prepare_bundle(root: Path) -> Path:
    output = root / "approvals" / "example" / "attempt-1"
    prepare_approval_bundle(
        CANDIDATE_PATH,
        _planning_workspace(root),
        output,
        _options(),
        lock_materializer=FakeLockMaterializer(),
        created_at=CREATED_AT,
    )
    return output


def _grant(bundle: Path, path: Path) -> Path:
    verified = verify_approval_bundle(bundle)
    approval = build_execution_approval(
        approval_id="example-cell-model-attempt-1-approval",
        approved_at="2026-07-18T01:00:00Z",
        manifest_fingerprint=verified.manifest.fingerprint,
        bundle_path="approvals/example/attempt-1",
        identity="example-reviewer",
        method="github_pr",
        pull_request_url="https://github.com/example/scFM_eval/pull/1",
        merge_commit="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    return write_execution_approval(path, approval)


class ExecutorTests(unittest.TestCase):
    def test_fake_transport_runs_approved_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = _prepare_bundle(root)
            approval = _grant(bundle, root / "execution-approval.json")
            host = FakeHost()
            outcome = execute_approved_bundle(
                bundle,
                approval,
                root / "run",
                host,
                now="2026-07-18T02:00:00Z",
            )
            record = json.loads(outcome.record_path.read_text(encoding="utf-8"))

            self.assertEqual(outcome.status, "completed_unreviewed")
            self.assertEqual(record["status"], "completed_unreviewed")
            self.assertEqual(record["review_status"], "completed_unreviewed")
            self.assertEqual(outcome.attempts, 1)
            step_names = [step["step"] for step in record["attempts"][0]["steps"]]
            self.assertEqual(
                step_names,
                [
                    "checkout",
                    "create_environment",
                    "install",
                    "download_weights",
                    "smoke_test",
                    "evaluate",
                    "collect_results",
                ],
            )

    def test_rejects_missing_or_mismatched_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = _prepare_bundle(root)
            host = FakeHost()

            with self.assertRaises(Exception):
                execute_approved_bundle(
                    bundle,
                    root / "missing.json",
                    root / "run-missing",
                    host,
                )

            other = build_execution_approval(
                approval_id="other-approval",
                approved_at="2026-07-18T01:00:00Z",
                manifest_fingerprint="0" * 64,
                bundle_path="approvals/example/attempt-1",
                identity="example-reviewer",
                method="manual",
                pull_request_url="https://github.com/example/scFM_eval/pull/2",
                merge_commit="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            )
            approval_path = write_execution_approval(
                root / "bad-approval.json",
                other,
            )
            with self.assertRaisesRegex(ExecutionError, "fingerprint"):
                execute_approved_bundle(
                    bundle,
                    approval_path,
                    root / "run-bad",
                    host,
                )

    def test_rejects_altered_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = _prepare_bundle(root)
            approval = _grant(bundle, root / "execution-approval.json")
            (bundle / "integrations/example_cell_model.py").write_text(
                "# altered\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(Exception, "checksum changed"):
                execute_approved_bundle(
                    bundle,
                    approval,
                    root / "run",
                    FakeHost(),
                )

    def test_retries_only_retryable_failed_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = _prepare_bundle(root)
            approval = _grant(bundle, root / "execution-approval.json")
            host = FakeHost(fail_step_prefix="scfm-eval")

            with self.assertRaisesRegex(ExecutionError, "failed after 2"):
                execute_approved_bundle(
                    bundle,
                    approval,
                    root / "run",
                    host,
                )
            # create_environment uses pixi; failing evaluate retries twice.
            evaluate_calls = [
                command
                for _, command in host.commands
                if command[:2] == ("scfm-eval", "run")
            ]
            self.assertEqual(len(evaluate_calls), 2)

    def test_does_not_retry_non_retryable_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = _prepare_bundle(root)
            approval = _grant(bundle, root / "execution-approval.json")
            host = FakeHost(fail_step_prefix="git")

            with self.assertRaisesRegex(ExecutionError, "failed after 1"):
                execute_approved_bundle(
                    bundle,
                    approval,
                    root / "run",
                    host,
                )
            git_calls = [
                command
                for _, command in host.commands
                if command and command[0] == "git"
            ]
            self.assertEqual(len(git_calls), 1)

    def test_cli_grant_and_execute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = _prepare_bundle(root)
            approval = root / "execution-approval.json"
            run_dir = root / "run"
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                grant_code = cli.main(
                    [
                        "approval",
                        "grant",
                        str(bundle),
                        "--output",
                        str(approval),
                        "--approval-id",
                        "example-cell-model-attempt-1-approval",
                        "--identity",
                        "example-reviewer",
                        "--pr-url",
                        "https://github.com/example/scFM_eval/pull/1",
                        "--merge-commit",
                        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "--bundle-path",
                        "approvals/example/attempt-1",
                    ]
                )
                execute_code = cli.main(
                    [
                        "execute",
                        str(bundle),
                        "--approval",
                        str(approval),
                        "--output",
                        str(run_dir),
                        "--transport",
                        "fake",
                    ]
                )

            self.assertEqual(grant_code, 0)
            self.assertEqual(execute_code, 0)
            self.assertIn("completed_unreviewed", stdout.getvalue())
            self.assertIn("Scientific review", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
