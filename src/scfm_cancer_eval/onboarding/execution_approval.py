"""Human pre-run approval bound to one exact execution manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from scfm_cancer_eval.onboarding._contract import (
    ImmutableContract,
    load_json,
    validate_envelope,
)

EXECUTION_APPROVAL_SCHEMA_NAME = "scfm_eval.execution_approval"
EXECUTION_APPROVAL_SCHEMA_VERSION = "1.0.0"
APPROVAL_METHODS = {"github_pr", "manual"}


def validate_execution_approval(payload: Any) -> None:
    validator, root = validate_envelope(
        payload,
        contract_name="execution approval",
        schema_name=EXECUTION_APPROVAL_SCHEMA_NAME,
        schema_version=EXECUTION_APPROVAL_SCHEMA_VERSION,
        id_key="approval_id",
        time_key="approved_at",
        allowed={
            "schema",
            "approval_id",
            "approved_at",
            "manifest_fingerprint",
            "bundle_path",
            "approver",
            "pull_request",
            "status",
        },
    )
    if root is None:
        return
    validator.sha256(
        root.get("manifest_fingerprint"),
        "$.manifest_fingerprint",
    )
    bundle_path = validator.relative_path(
        root.get("bundle_path"),
        "$.bundle_path",
    )
    del bundle_path
    approver = validator.object(
        root.get("approver"),
        "$.approver",
        allowed={"identity", "method"},
    )
    if approver is not None:
        validator.text(approver.get("identity"), "$.approver.identity")
        validator.enum(
            approver.get("method"),
            APPROVAL_METHODS,
            "$.approver.method",
        )
    pull_request = validator.object(
        root.get("pull_request"),
        "$.pull_request",
        allowed={"url", "merge_commit"},
    )
    if pull_request is not None:
        url = validator.text(pull_request.get("url"), "$.pull_request.url")
        if url is not None:
            parts = urlsplit(url)
            if parts.scheme not in {"http", "https"} or not parts.netloc:
                validator.errors.append(
                    "$.pull_request.url must be an http(s) URL"
                )
        validator.commit(
            pull_request.get("merge_commit"),
            "$.pull_request.merge_commit",
        )
    validator.enum(root.get("status"), {"approved"}, "$.status")
    validator.finish()


class ExecutionApproval(ImmutableContract):
    """Immutable record that one manifest fingerprint was approved to run."""

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "ExecutionApproval":
        validate_execution_approval(payload)
        return cls._from_validated(
            payload,
            id_key="approval_id",
            time_key="approved_at",
        )

    @property
    def manifest_fingerprint(self) -> str:
        return str(self.to_dict()["manifest_fingerprint"])


def load_execution_approval(path: str | Path) -> ExecutionApproval:
    return ExecutionApproval.from_payload(load_json(path))


def build_execution_approval(
    *,
    approval_id: str,
    approved_at: str,
    manifest_fingerprint: str,
    bundle_path: str,
    identity: str,
    method: str,
    pull_request_url: str,
    merge_commit: str,
) -> ExecutionApproval:
    return ExecutionApproval.from_payload(
        {
            "schema": {
                "name": EXECUTION_APPROVAL_SCHEMA_NAME,
                "version": EXECUTION_APPROVAL_SCHEMA_VERSION,
            },
            "approval_id": approval_id,
            "approved_at": approved_at,
            "manifest_fingerprint": manifest_fingerprint,
            "bundle_path": bundle_path,
            "approver": {
                "identity": identity,
                "method": method,
            },
            "pull_request": {
                "url": pull_request_url,
                "merge_commit": merge_commit,
            },
            "status": "approved",
        }
    )


def write_execution_approval(
    path: str | Path,
    approval: ExecutionApproval,
) -> Path:
    output = Path(path)
    if output.exists():
        raise ValueError(f"Approval record already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(approval.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
