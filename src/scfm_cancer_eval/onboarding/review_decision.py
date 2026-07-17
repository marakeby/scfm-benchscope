"""Human scientific decision bound to exact run artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scfm_cancer_eval.onboarding._contract import (
    ImmutableContract,
    Validator,
    load_json,
    validate_envelope,
)

REVIEW_DECISION_SCHEMA_NAME = "scfm_eval.review_decision"
REVIEW_DECISION_SCHEMA_VERSION = "1.0.0"
REVIEW_DECISIONS = {"accepted", "needs_tuning", "rejected"}
REVIEW_METHODS = {"github_pr", "manual"}


def validate_review_decision(payload: Any) -> None:
    validator, root = validate_envelope(
        payload,
        contract_name="review decision",
        schema_name=REVIEW_DECISION_SCHEMA_NAME,
        schema_version=REVIEW_DECISION_SCHEMA_VERSION,
        id_key="decision_id",
        time_key="decided_at",
        allowed={
            "schema",
            "decision_id",
            "decided_at",
            "reviewer",
            "run",
            "decision",
            "rationale",
            "tuning",
            "publication",
        },
    )
    if root is None:
        return
    _reviewer(validator, root.get("reviewer"))
    _run(validator, root.get("run"))
    decision = validator.enum(
        root.get("decision"),
        REVIEW_DECISIONS,
        "$.decision",
    )
    validator.text(root.get("rationale"), "$.rationale")
    tuning = root.get("tuning")
    if tuning is not None:
        _tuning(validator, tuning)
    include_in_reports, promote_baseline = _publication(
        validator,
        root.get("publication"),
    )
    if decision == "needs_tuning" and tuning is None:
        validator.errors.append(
            "$.tuning is required when decision is 'needs_tuning'"
        )
    if decision != "needs_tuning" and tuning is not None:
        validator.errors.append(
            "$.tuning is only allowed when decision is 'needs_tuning'"
        )
    if decision != "accepted" and (
        include_in_reports is True or promote_baseline is True
    ):
        validator.errors.append(
            "$.publication must remain false until the run is accepted"
        )
    validator.finish()


def _reviewer(validator: Validator, value: Any) -> None:
    item = validator.object(
        value,
        "$.reviewer",
        allowed={"identity", "method"},
    )
    if item is None:
        return
    validator.text(item.get("identity"), "$.reviewer.identity")
    validator.enum(
        item.get("method"),
        REVIEW_METHODS,
        "$.reviewer.method",
    )


def _run(validator: Validator, value: Any) -> None:
    item = validator.object(
        value,
        "$.run",
        allowed={
            "run_id",
            "results_sha256",
            "manifest_fingerprint",
            "attempt",
            "previous_run_id",
        },
    )
    if item is None:
        return
    validator.identifier(item.get("run_id"), "$.run.run_id")
    validator.sha256(item.get("results_sha256"), "$.run.results_sha256")
    validator.sha256(
        item.get("manifest_fingerprint"),
        "$.run.manifest_fingerprint",
    )
    validator.number(
        item.get("attempt"),
        "$.run.attempt",
        minimum=1,
        integer=True,
    )
    if "previous_run_id" in item and item["previous_run_id"] is not None:
        validator.identifier(
            item["previous_run_id"],
            "$.run.previous_run_id",
        )


def _tuning(validator: Validator, value: Any) -> None:
    item = validator.object(
        value,
        "$.tuning",
        allowed={
            "changes",
            "expected_improvement",
            "max_additional_budget_usd",
        },
    )
    if item is None:
        return
    validator.string_list(
        item.get("changes"),
        "$.tuning.changes",
        non_empty=True,
    )
    validator.text(
        item.get("expected_improvement"),
        "$.tuning.expected_improvement",
    )
    validator.number(
        item.get("max_additional_budget_usd"),
        "$.tuning.max_additional_budget_usd",
    )


def _publication(
    validator: Validator,
    value: Any,
) -> tuple[bool | None, bool | None]:
    item = validator.object(
        value,
        "$.publication",
        allowed={"include_in_reports", "promote_baseline"},
    )
    if item is None:
        return None, None
    return (
        validator.boolean(
            item.get("include_in_reports"),
            "$.publication.include_in_reports",
        ),
        validator.boolean(
            item.get("promote_baseline"),
            "$.publication.promote_baseline",
        ),
    )


class ReviewDecision(ImmutableContract):
    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "ReviewDecision":
        validate_review_decision(payload)
        return cls._from_validated(
            payload,
            id_key="decision_id",
            time_key="decided_at",
        )


def load_review_decision(path: str | Path) -> ReviewDecision:
    return ReviewDecision.from_payload(load_json(path))
