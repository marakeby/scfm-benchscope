"""Public facade for planning, execution, and review contracts."""

from __future__ import annotations

import json
from importlib.resources import files

from scfm_cancer_eval.onboarding._contract import ContractValidationError
from scfm_cancer_eval.onboarding.candidate import ModelCandidate
from scfm_cancer_eval.onboarding.execution_manifest import (
    EXECUTION_MANIFEST_SCHEMA_NAME,
    EXECUTION_MANIFEST_SCHEMA_VERSION,
    ExecutionManifest,
    load_execution_manifest,
    validate_execution_manifest,
)
from scfm_cancer_eval.onboarding.integration_plan import (
    INTEGRATION_PLAN_SCHEMA_NAME,
    INTEGRATION_PLAN_SCHEMA_VERSION,
    IntegrationPlan,
    load_integration_plan,
    validate_integration_plan,
)
from scfm_cancer_eval.onboarding.model_spec import (
    MODEL_SPEC_SCHEMA_NAME,
    MODEL_SPEC_SCHEMA_VERSION,
    ModelSpec,
    load_model_spec,
    validate_model_spec,
)
from scfm_cancer_eval.onboarding.review_decision import (
    REVIEW_DECISION_SCHEMA_NAME,
    REVIEW_DECISION_SCHEMA_VERSION,
    ReviewDecision,
    load_review_decision,
    validate_review_decision,
)


def planning_schema(name: str) -> dict:
    """Load one packaged planning-contract JSON Schema by short name."""
    filenames = {
        "model_spec": "model-spec-v1.0.0.json",
        "integration_plan": "integration-plan-v1.0.0.json",
        "execution_manifest": "execution-manifest-v1.0.0.json",
        "execution_approval": "execution-approval-v1.0.0.json",
        "review_decision": "review-decision-v1.0.0.json",
    }
    try:
        filename = filenames[name]
    except KeyError as exc:
        raise ValueError(f"Unknown planning schema: {name}") from exc
    schema_text = (
        files("scfm_cancer_eval")
        .joinpath(f"schemas/{filename}")
        .read_text(encoding="utf-8")
    )
    return json.loads(schema_text)


def validate_planning_chain(
    candidate: ModelCandidate,
    model_spec: ModelSpec,
    integration_plan: IntegrationPlan,
    manifest: ExecutionManifest,
) -> None:
    """Ensure an execution manifest exactly follows its proposed inputs."""
    errors: list[str] = []
    model_payload = model_spec.to_dict()
    plan_payload = integration_plan.to_dict()
    manifest_payload = manifest.to_dict()

    links = [
        (
            model_payload["candidate"]["candidate_id"],
            candidate.candidate_id,
            "$.model_spec.candidate.candidate_id",
        ),
        (
            model_payload["candidate"]["fingerprint"],
            candidate.fingerprint,
            "$.model_spec.candidate.fingerprint",
        ),
        (
            plan_payload["candidate_fingerprint"],
            candidate.fingerprint,
            "$.integration_plan.candidate_fingerprint",
        ),
        (
            plan_payload["model_spec_fingerprint"],
            model_spec.fingerprint,
            "$.integration_plan.model_spec_fingerprint",
        ),
        (
            manifest_payload["model_spec_fingerprint"],
            model_spec.fingerprint,
            "$.manifest.model_spec_fingerprint",
        ),
        (
            manifest_payload["integration_plan_fingerprint"],
            integration_plan.fingerprint,
            "$.manifest.integration_plan_fingerprint",
        ),
    ]
    for actual, expected, path in links:
        if actual != expected:
            errors.append(f"{path} does not match its referenced document")

    if plan_payload["unresolved_fields"]:
        errors.append(
            "$.integration_plan.unresolved_fields must be empty before "
            "an execution manifest can be approved"
        )
    for field in ("repository", "weights"):
        if manifest_payload[field] != model_payload[field]:
            errors.append(
                f"$.manifest.{field} does not match the model spec"
            )
    if manifest_payload["generated_files"] != plan_payload["generated_files"]:
        errors.append(
            "$.manifest.generated_files does not match the integration plan"
        )
    if (
        manifest_payload["environment"]["name"]
        != plan_payload["environment"]["name"]
    ):
        errors.append(
            "$.manifest.environment.name does not match the integration plan"
        )

    adapter = model_payload["adapter"]
    evaluation = manifest_payload["evaluation"]
    if (
        evaluation["adapter_module"] != adapter["module"]
        or evaluation["adapter_class"] != adapter["class"]
    ):
        errors.append(
            "$.manifest.evaluation adapter does not match the model spec"
        )
    if evaluation["tasks"] != model_payload["tasks"]:
        errors.append(
            "$.manifest.evaluation.tasks does not match the model spec"
        )

    lock_path = manifest_payload["environment"]["lock_path"]
    lock_sha256 = manifest_payload["environment"]["lock_sha256"]
    generated_hashes = {
        item["path"]: item["sha256"]
        for item in manifest_payload["generated_files"]
    }
    if generated_hashes.get(lock_path) != lock_sha256:
        errors.append(
            "$.manifest.environment lock does not match generated_files"
        )

    if errors:
        raise ContractValidationError("planning chain", errors)


__all__ = [
    "ContractValidationError",
    "EXECUTION_MANIFEST_SCHEMA_NAME",
    "EXECUTION_MANIFEST_SCHEMA_VERSION",
    "ExecutionManifest",
    "INTEGRATION_PLAN_SCHEMA_NAME",
    "INTEGRATION_PLAN_SCHEMA_VERSION",
    "IntegrationPlan",
    "MODEL_SPEC_SCHEMA_NAME",
    "MODEL_SPEC_SCHEMA_VERSION",
    "ModelSpec",
    "REVIEW_DECISION_SCHEMA_NAME",
    "REVIEW_DECISION_SCHEMA_VERSION",
    "ReviewDecision",
    "load_execution_manifest",
    "load_integration_plan",
    "load_model_spec",
    "load_review_decision",
    "planning_schema",
    "validate_execution_manifest",
    "validate_integration_plan",
    "validate_model_spec",
    "validate_planning_chain",
    "validate_review_decision",
]
