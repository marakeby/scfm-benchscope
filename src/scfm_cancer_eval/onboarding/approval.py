"""Deterministic lockfile materialization and pre-run approval bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from scfm_cancer_eval.onboarding.candidate import (
    ModelCandidate,
    load_model_candidate,
)
from scfm_cancer_eval.onboarding.execution_manifest import (
    EXECUTION_STEPS,
    ExecutionManifest,
    load_execution_manifest,
)
from scfm_cancer_eval.onboarding.integration_plan import (
    IntegrationPlan,
    load_integration_plan,
)
from scfm_cancer_eval.onboarding.model_spec import (
    ModelSpec,
    load_model_spec,
)
from scfm_cancer_eval.onboarding.planning import validate_planning_chain

APPROVAL_REQUEST_SCHEMA_NAME = "scfm_eval.approval_request"
APPROVAL_REQUEST_SCHEMA_VERSION = "1.0.0"
MAX_LOCK_BYTES = 50_000_000


class ApprovalError(ValueError):
    """Raised when a proposal cannot become a valid approval bundle."""


class LockMaterializer(Protocol):
    def materialize(self, pixi_toml: Path, lock_path: Path) -> None:
        """Resolve a lockfile without installing the environment."""


class PixiLockMaterializer:
    """Use Pixi's lock-only operation; no environment is installed."""

    def materialize(self, pixi_toml: Path, lock_path: Path) -> None:
        completed = subprocess.run(
            [
                "pixi",
                "lock",
                "--manifest-path",
                str(pixi_toml),
                "--no-install",
            ],
            cwd=pixi_toml.parent,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ApprovalError(f"pixi lock failed: {detail}")
        if not lock_path.is_file():
            raise ApprovalError("pixi lock did not create pixi.lock")


@dataclass(frozen=True)
class ApprovalOptions:
    manifest_id: str
    gpu_type: str
    gpu_count: int
    disk_gb: float
    max_runtime_minutes: float
    hourly_rate_usd: float
    max_budget_usd: float
    max_attempts: int = 1
    retryable_steps: tuple[str, ...] = (
        "download_weights",
        "evaluate",
    )
    secret_names: tuple[str, ...] = ()
    additional_network_hosts: tuple[str, ...] = ()
    experiment_path: str | None = None


@dataclass(frozen=True)
class ApprovalBundle:
    root: Path
    candidate: ModelCandidate
    model_spec: ModelSpec
    integration_plan: IntegrationPlan
    manifest: ExecutionManifest
    request_path: Path


def prepare_approval_bundle(
    candidate_path: str | Path,
    planning_workspace: str | Path,
    output_dir: str | Path,
    options: ApprovalOptions,
    *,
    lock_materializer: LockMaterializer | None = None,
    created_at: str | None = None,
) -> ApprovalBundle:
    """Materialize one immutable bundle for review in a pull request."""
    source = Path(planning_workspace)
    output = Path(output_dir)
    if output.exists():
        raise ApprovalError(f"Approval output already exists: {output}")

    candidate = load_model_candidate(candidate_path)
    model_spec = load_model_spec(source / "model-spec.json")
    integration_plan = load_integration_plan(
        source / "integration-plan.json"
    )
    _validate_proposal_links(candidate, model_spec, integration_plan)
    _verify_generated_files(source, integration_plan)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            dir=output.parent,
        )
    )
    try:
        _copy_generated_files(source, staging, integration_plan)
        _copy_planner_evidence(source, staging)
        _write_json(
            staging / "proposed-integration-plan.json",
            integration_plan.to_dict(),
        )
        materializer = lock_materializer or PixiLockMaterializer()
        lock_path = staging / "pixi.lock"
        materializer.materialize(staging / "pixi.toml", lock_path)
        _validate_lockfile(lock_path)

        finalized_plan = _finalize_plan(integration_plan, lock_path)
        manifest = _build_manifest(
            model_spec,
            finalized_plan,
            options,
            created_at=created_at,
        )
        validate_planning_chain(
            candidate,
            model_spec,
            finalized_plan,
            manifest,
        )

        _write_json(staging / "candidate.json", candidate.to_dict())
        _write_json(staging / "model-spec.json", model_spec.to_dict())
        _write_json(
            staging / "integration-plan.json",
            finalized_plan.to_dict(),
        )
        _write_json(staging / "execution-manifest.json", manifest.to_dict())
        _write_json(
            staging / "approval-request.json",
            _approval_request(
                candidate,
                model_spec,
                integration_plan,
                finalized_plan,
                manifest,
                _planner_evidence_hashes(staging),
                created_at=created_at,
            ),
        )
        verify_approval_bundle(staging)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return verify_approval_bundle(output)


def verify_approval_bundle(root: str | Path) -> ApprovalBundle:
    """Verify fingerprints, generated files, and budget-bound manifest."""
    bundle_root = Path(root)
    candidate = load_model_candidate(bundle_root / "candidate.json")
    model_spec = load_model_spec(bundle_root / "model-spec.json")
    integration_plan = load_integration_plan(
        bundle_root / "integration-plan.json"
    )
    proposed_plan = load_integration_plan(
        bundle_root / "proposed-integration-plan.json"
    )
    manifest = load_execution_manifest(
        bundle_root / "execution-manifest.json"
    )
    _validate_proposal_links(candidate, model_spec, proposed_plan)
    lock_path = bundle_root / "pixi.lock"
    _validate_lockfile(lock_path)
    expected_plan = _finalize_plan(proposed_plan, lock_path)
    if integration_plan.fingerprint != expected_plan.fingerprint:
        raise ApprovalError(
            "final integration plan is not the locked proposal"
        )
    validate_planning_chain(
        candidate,
        model_spec,
        integration_plan,
        manifest,
    )
    _verify_generated_files(bundle_root, integration_plan)

    request_path = bundle_root / "approval-request.json"
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApprovalError(f"Invalid approval-request.json: {exc}") from exc
    expected_request = _approval_request(
        candidate,
        model_spec,
        proposed_plan,
        integration_plan,
        manifest,
        _planner_evidence_hashes(bundle_root),
        created_at=manifest.timestamp,
    )
    if request != expected_request:
        raise ApprovalError(
            "approval request does not exactly match the execution manifest"
        )
    _verify_bundle_file_set(bundle_root, integration_plan)

    return ApprovalBundle(
        root=bundle_root,
        candidate=candidate,
        model_spec=model_spec,
        integration_plan=integration_plan,
        manifest=manifest,
        request_path=request_path,
    )


def _validate_proposal_links(
    candidate: ModelCandidate,
    model_spec: ModelSpec,
    integration_plan: IntegrationPlan,
) -> None:
    model_payload = model_spec.to_dict()
    plan_payload = integration_plan.to_dict()
    if model_payload["candidate"] != {
        "candidate_id": candidate.candidate_id,
        "fingerprint": candidate.fingerprint,
    }:
        raise ApprovalError("model spec does not match candidate")
    if (
        plan_payload["candidate_fingerprint"] != candidate.fingerprint
        or plan_payload["model_spec_fingerprint"] != model_spec.fingerprint
    ):
        raise ApprovalError("integration plan references do not match")
    if plan_payload["unresolved_fields"]:
        raise ApprovalError(
            "integration plan still has unresolved fields"
        )
    if any(
        item["path"] == "pixi.lock"
        for item in plan_payload["generated_files"]
    ):
        raise ApprovalError(
            "proposal must not contain an AI-generated pixi.lock"
        )


def _generated_items(plan: IntegrationPlan) -> list[dict]:
    return list(plan.to_dict()["generated_files"])


def _verify_generated_files(root: Path, plan: IntegrationPlan) -> None:
    generated_paths = {item["path"] for item in _generated_items(plan)}
    if "pixi.toml" not in generated_paths:
        raise ApprovalError("integration plan must generate pixi.toml")
    for item in _generated_items(plan):
        path = root / item["path"]
        if (
            path.is_symlink()
            or not path.is_file()
            or not _is_within(path, root)
        ):
            raise ApprovalError(
                f"generated file is missing or not regular: {item['path']}"
            )
        if _sha256_file(path) != item["sha256"]:
            raise ApprovalError(
                f"generated file checksum changed: {item['path']}"
            )


def _copy_generated_files(
    source: Path,
    destination: Path,
    plan: IntegrationPlan,
) -> None:
    for item in _generated_items(plan):
        source_path = source / item["path"]
        destination_path = destination / item["path"]
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination_path)


def _copy_planner_evidence(source: Path, destination: Path) -> None:
    for name in ("proposal.json", "planning-status.json"):
        source_path = source / name
        if (
            source_path.is_symlink()
            or not source_path.is_file()
            or not _is_within(source_path, source)
        ):
            raise ApprovalError(f"planner evidence is missing: {name}")
        destination_path = destination / name
        shutil.copyfile(source_path, destination_path)


def _verify_bundle_file_set(root: Path, plan: IntegrationPlan) -> None:
    control_files = {
        "candidate.json",
        "model-spec.json",
        "proposed-integration-plan.json",
        "integration-plan.json",
        "execution-manifest.json",
        "approval-request.json",
        "proposal.json",
        "planning-status.json",
    }
    expected = control_files | {
        item["path"] for item in _generated_items(plan)
    }
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ApprovalError(f"approval bundle contains symlink: {path}")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != expected:
        extras = sorted(actual - expected)
        missing = sorted(expected - actual)
        detail = extras[0] if extras else missing[0]
        raise ApprovalError(f"approval bundle has an unexpected file set: {detail}")


def _validate_lockfile(lock_path: Path) -> None:
    if lock_path.is_symlink() or not lock_path.is_file():
        raise ApprovalError("pixi.lock must be a regular file")
    size = lock_path.stat().st_size
    if size == 0:
        raise ApprovalError("pixi.lock must not be empty")
    if size > MAX_LOCK_BYTES:
        raise ApprovalError("pixi.lock exceeds the size limit")


def _finalize_plan(
    plan: IntegrationPlan,
    lock_path: Path,
) -> IntegrationPlan:
    payload = plan.to_dict()
    generated = [
        item
        for item in payload["generated_files"]
        if item["path"] != "pixi.lock"
    ]
    generated.append(
        {
            "path": "pixi.lock",
            "purpose": "Deterministically resolved Pixi lockfile",
            "sha256": _sha256_file(lock_path),
        }
    )
    payload["generated_files"] = sorted(
        generated,
        key=lambda item: item["path"],
    )
    return IntegrationPlan.from_payload(payload)


def _build_manifest(
    model_spec: ModelSpec,
    plan: IntegrationPlan,
    options: ApprovalOptions,
    *,
    created_at: str | None,
) -> ExecutionManifest:
    model = model_spec.to_dict()
    plan_payload = plan.to_dict()
    timestamp = created_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00",
        "Z",
    )
    experiment_path = options.experiment_path or _single_experiment(
        plan_payload["generated_files"]
    )
    hosts = {
        urlsplit(model["repository"]["url"]).hostname,
        *(
            urlsplit(weight["url"]).hostname
            for weight in model["weights"]
        ),
        *options.additional_network_hosts,
    }
    hosts.discard(None)
    lock = next(
        item
        for item in plan_payload["generated_files"]
        if item["path"] == "pixi.lock"
    )
    payload = {
        "schema": {
            "name": "scfm_eval.execution_manifest",
            "version": "1.0.0",
        },
        "manifest_id": options.manifest_id,
        "created_at": timestamp,
        "model_spec_fingerprint": model_spec.fingerprint,
        "integration_plan_fingerprint": plan.fingerprint,
        "repository": model["repository"],
        "weights": model["weights"],
        "generated_files": plan_payload["generated_files"],
        "environment": {
            "name": plan_payload["environment"]["name"],
            "lock_path": "pixi.lock",
            "lock_sha256": lock["sha256"],
        },
        "evaluation": {
            "experiment_path": experiment_path,
            "adapter_module": model["adapter"]["module"],
            "adapter_class": model["adapter"]["class"],
            "output_subdir": (
                f"{model['model']['model_id']}/{options.manifest_id}"
            ),
            "tasks": model["tasks"],
        },
        "steps": EXECUTION_STEPS,
        "resources": {
            "gpu_type": options.gpu_type,
            "gpu_count": options.gpu_count,
            "disk_gb": options.disk_gb,
            "max_runtime_minutes": options.max_runtime_minutes,
            "hourly_rate_usd": options.hourly_rate_usd,
            "max_budget_usd": options.max_budget_usd,
        },
        "permissions": {
            "network_hosts": sorted(str(host) for host in hosts),
            "secret_names": list(options.secret_names),
            "dataset_read_only": True,
        },
        "retry_policy": {
            "max_attempts": options.max_attempts,
            "retryable_steps": list(options.retryable_steps),
        },
        "expected_outputs": [
            "results.json",
            "resolved_config.yaml",
            "execution.log",
        ],
    }
    try:
        return ExecutionManifest.from_payload(payload)
    except ValueError as exc:
        raise ApprovalError(f"Invalid execution options: {exc}") from exc


def _single_experiment(generated: list[dict]) -> str:
    experiments = [
        item["path"]
        for item in generated
        if item["path"].startswith("experiments/")
        and item["path"].endswith((".yaml", ".yml"))
    ]
    if len(experiments) != 1:
        raise ApprovalError(
            "select experiment_path when the plan does not contain exactly "
            "one experiment YAML"
        )
    return experiments[0]


def _approval_request(
    candidate: ModelCandidate,
    model_spec: ModelSpec,
    proposed_plan: IntegrationPlan,
    plan: IntegrationPlan,
    manifest: ExecutionManifest,
    planner_evidence: dict[str, str],
    *,
    created_at: str | None,
) -> dict:
    timestamp = created_at or manifest.timestamp
    return {
        "schema": {
            "name": APPROVAL_REQUEST_SCHEMA_NAME,
            "version": APPROVAL_REQUEST_SCHEMA_VERSION,
        },
        "request_id": manifest.document_id,
        "created_at": timestamp,
        "status": "pending_human_review",
        "candidate_fingerprint": candidate.fingerprint,
        "model_spec_fingerprint": model_spec.fingerprint,
        "proposed_integration_plan_fingerprint": proposed_plan.fingerprint,
        "integration_plan_fingerprint": plan.fingerprint,
        "manifest_fingerprint": manifest.fingerprint,
        "planner_evidence": planner_evidence,
        "budget": manifest.to_dict()["resources"],
        "review_checks": [
            "sources_and_licenses",
            "generated_code_and_dependencies",
            "datasets_tasks_and_permissions",
            "runtime_retries_and_budget",
        ],
    }


def _planner_evidence_hashes(root: Path) -> dict[str, str]:
    evidence: dict[str, str] = {}
    for name in ("proposal.json", "planning-status.json"):
        path = root / name
        if path.is_symlink() or not path.is_file() or not _is_within(path, root):
            raise ApprovalError(f"planner evidence is missing: {name}")
        evidence[name] = _sha256_file(path)
    return evidence


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
