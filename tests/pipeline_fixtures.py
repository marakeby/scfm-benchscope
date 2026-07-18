"""Shared fixtures for dual-gate and adversarial pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

from scfm_cancer_eval.onboarding import (
    FakeHost,
    ReviewOptions,
    build_execution_approval,
    execute_approved_bundle,
    prepare_approval_bundle,
    record_review,
    verify_approval_bundle,
    write_execution_approval,
)
from scfm_cancer_eval.utils.results_json import write_results_json
from test_approval import (
    CANDIDATE_PATH,
    CREATED_AT,
    FakeLockMaterializer,
    _options,
    _planning_workspace,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = CANDIDATE_PATH


def results_payload(run_id: str, *, nmi: float = 0.8) -> dict:
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
                "aggregate": {"metrics": {"NMI": nmi}},
                "folds": [],
                "artifacts": {},
                "status": "success",
                "errors": [],
            }
        ],
        "timing": {},
    }


def prepare_bundle(root: Path, **option_overrides) -> Path:
    bundle = root / "approvals" / "example" / "attempt-1"
    prepare_approval_bundle(
        CANDIDATE,
        _planning_workspace(root),
        bundle,
        _options(**option_overrides),
        lock_materializer=FakeLockMaterializer(),
        created_at=CREATED_AT,
    )
    return bundle


def grant_bundle(bundle: Path, approval_path: Path) -> Path:
    verified = verify_approval_bundle(bundle)
    return write_execution_approval(
        approval_path,
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


def complete_unreviewed_run(root: Path) -> tuple[Path, Path]:
    bundle = prepare_bundle(root)
    approval = grant_bundle(bundle, root / "execution-approval.json")
    run_dir = root / "runs" / "attempt-1"
    execute_approved_bundle(
        bundle,
        approval,
        run_dir,
        FakeHost(),
        now="2026-07-18T02:00:00Z",
    )
    verified = verify_approval_bundle(bundle)
    write_results_json(
        str(run_dir / "output" / "results.json"),
        results_payload(verified.manifest.document_id),
    )
    return bundle, run_dir


def accept_run(run_dir: Path, *, decision_id: str = "example-accepted") -> Path:
    outcome = record_review(
        run_dir,
        ReviewOptions(
            decision_id=decision_id,
            decision="accepted",
            identity="scientist",
            rationale="Synthetic dual-gate acceptance for CI.",
        ),
        decided_at="2026-07-18T03:00:00Z",
    )
    return outcome.decision_path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
