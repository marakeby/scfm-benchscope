"""Record post-run scientific decisions against completed executions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scfm_cancer_eval.onboarding.review_decision import (
    ReviewDecision,
    load_review_decision,
    validate_review_decision,
)

TUNING_LINEAGE_SCHEMA_NAME = "scfm_eval.tuning_lineage"
TUNING_LINEAGE_SCHEMA_VERSION = "1.0.0"


class ReviewError(ValueError):
    """Raised when a scientific review cannot be recorded."""


@dataclass(frozen=True)
class ReviewOptions:
    decision_id: str
    decision: str
    identity: str
    rationale: str
    method: str = "manual"
    include_in_reports: bool | None = None
    promote_baseline: bool = False
    tuning_changes: tuple[str, ...] = ()
    expected_improvement: str | None = None
    max_additional_budget_usd: float | None = None
    previous_run_id: str | None = None
    attempt: int | None = None


@dataclass(frozen=True)
class ReviewOutcome:
    run_dir: Path
    decision: ReviewDecision
    decision_path: Path
    lineage_path: Path | None


def record_review(
    run_dir: str | Path,
    options: ReviewOptions,
    *,
    decided_at: str | None = None,
) -> ReviewOutcome:
    """Bind one human decision to exact run and result fingerprints."""
    root = Path(run_dir)
    if not root.is_dir():
        raise ReviewError(f"Run directory does not exist: {root}")

    decision_path = root / "review-decision.json"
    if decision_path.exists():
        raise ReviewError(
            "review-decision.json already exists; create a new run for "
            "another decision"
        )

    record = _load_execution_record(root)
    results_path = _find_results_json(root)
    results_sha256 = _sha256_file(results_path)
    timestamp = decided_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00",
        "Z",
    )

    if record["status"] != "completed_unreviewed":
        raise ReviewError(
            "only completed_unreviewed executions may receive a scientific "
            f"review (found {record['status']!r})"
        )
    if record.get("review_status") not in {
        "completed_unreviewed",
        "not_applicable",
    }:
        raise ReviewError(
            "execution-record.json already has a scientific review status"
        )

    include_in_reports = options.include_in_reports
    if include_in_reports is None:
        include_in_reports = options.decision == "accepted"
    if options.decision != "accepted" and (
        include_in_reports or options.promote_baseline
    ):
        raise ReviewError(
            "only accepted decisions may set publication flags"
        )
    if options.decision == "needs_tuning":
        if not options.tuning_changes or not options.expected_improvement:
            raise ReviewError(
                "needs_tuning requires --change and --expected-improvement"
            )
        if options.max_additional_budget_usd is None:
            raise ReviewError(
                "needs_tuning requires --max-additional-budget-usd"
            )

    payload: dict[str, Any] = {
        "schema": {
            "name": "scfm_eval.review_decision",
            "version": "1.0.0",
        },
        "decision_id": options.decision_id,
        "decided_at": timestamp,
        "reviewer": {
            "identity": options.identity,
            "method": options.method,
        },
        "run": {
            "run_id": record["run_id"],
            "results_sha256": results_sha256,
            "manifest_fingerprint": record["manifest_fingerprint"],
            "attempt": options.attempt or _attempt_count(record),
            "previous_run_id": options.previous_run_id,
        },
        "decision": options.decision,
        "rationale": options.rationale,
        "tuning": None,
        "publication": {
            "include_in_reports": include_in_reports,
            "promote_baseline": options.promote_baseline,
        },
    }
    if options.decision == "needs_tuning":
        payload["tuning"] = {
            "changes": list(options.tuning_changes),
            "expected_improvement": options.expected_improvement,
            "max_additional_budget_usd": options.max_additional_budget_usd,
        }

    try:
        validate_review_decision(payload)
        decision = ReviewDecision.from_payload(payload)
    except ValueError as exc:
        raise ReviewError(str(exc)) from exc

    _write_json(decision_path, decision.to_dict())
    lineage_path = None
    if options.decision == "needs_tuning":
        lineage_path = root / "tuning-lineage.json"
        _write_json(
            lineage_path,
            {
                "schema": {
                    "name": TUNING_LINEAGE_SCHEMA_NAME,
                    "version": TUNING_LINEAGE_SCHEMA_VERSION,
                },
                "lineage_id": f"{options.decision_id}-lineage",
                "created_at": timestamp,
                "source_run_id": record["run_id"],
                "source_decision_fingerprint": decision.fingerprint,
                "changes": list(options.tuning_changes),
                "expected_improvement": options.expected_improvement,
                "max_additional_budget_usd": options.max_additional_budget_usd,
                "requires_new_pre_run_approval": True,
                "note": (
                    "Material dependency, command, source, weight, dataset, "
                    "or budget changes must open a new pre-run approval PR "
                    "before re-execution."
                ),
            },
        )

    record["review_status"] = options.decision
    record["review_decision_fingerprint"] = decision.fingerprint
    _write_json(root / "execution-record.json", record)

    return ReviewOutcome(
        run_dir=root,
        decision=decision,
        decision_path=decision_path,
        lineage_path=lineage_path,
    )


def load_run_review_status(results_path: str | Path) -> str:
    """Resolve publication status for one results.json location."""
    path = Path(results_path)
    decision = _nearby_review_decision(path)
    if decision is not None:
        payload = decision.to_dict()
        if (
            payload["decision"] == "accepted"
            and not payload["publication"]["include_in_reports"]
        ):
            return "accepted_unpublished"
        return str(payload["decision"])

    record_path = _nearby_execution_record(path)
    if record_path is not None:
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "completed_unreviewed"
        status = record.get("review_status")
        if isinstance(status, str) and status not in {
            "",
            "not_applicable",
        }:
            return status
        if record.get("status") == "completed_unreviewed":
            return "completed_unreviewed"
    return "local"


def is_publishable_review_status(status: str) -> bool:
    return status == "accepted"


def _load_execution_record(root: Path) -> dict[str, Any]:
    path = root / "execution-record.json"
    if not path.is_file():
        raise ReviewError("execution-record.json is required for scientific review")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewError(f"Invalid execution-record.json: {exc}") from exc
    for key in (
        "run_id",
        "status",
        "manifest_fingerprint",
        "review_status",
    ):
        if key not in record:
            raise ReviewError(f"execution-record.json missing {key}")
    return record


def _find_results_json(root: Path) -> Path:
    direct = root / "results.json"
    nested = root / "output" / "results.json"
    if direct.is_file():
        return direct
    if nested.is_file():
        return nested
    matches = sorted(root.rglob("results.json"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ReviewError("results.json was not found in the run directory")
    raise ReviewError("run directory contains more than one results.json")


def _attempt_count(record: dict[str, Any]) -> int:
    attempts = record.get("attempts")
    if isinstance(attempts, list) and attempts:
        return len(attempts)
    return 1


def _nearby_review_decision(results_path: Path) -> ReviewDecision | None:
    for candidate in (
        results_path.parent / "review-decision.json",
        results_path.parent.parent / "review-decision.json",
    ):
        if candidate.is_file():
            try:
                return load_review_decision(candidate)
            except ValueError:
                return None
    return None


def _nearby_execution_record(results_path: Path) -> Path | None:
    for candidate in (
        results_path.parent / "execution-record.json",
        results_path.parent.parent / "execution-record.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
