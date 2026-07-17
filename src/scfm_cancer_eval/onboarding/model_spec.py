"""Planner-enriched, immutable model specification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scfm_cancer_eval.onboarding._contract import (
    ImmutableContract,
    load_json,
    validate_envelope,
)
from scfm_cancer_eval.onboarding._planning_fields import (
    LICENSE_STATUSES,
    PLATFORMS,
    adapter,
    reference,
    repository,
    weights,
)

MODEL_SPEC_SCHEMA_NAME = "scfm_eval.model_spec"
MODEL_SPEC_SCHEMA_VERSION = "1.0.0"


def validate_model_spec(payload: Any) -> None:
    validator, root = validate_envelope(
        payload,
        contract_name="model spec",
        schema_name=MODEL_SPEC_SCHEMA_NAME,
        schema_version=MODEL_SPEC_SCHEMA_VERSION,
        id_key="model_spec_id",
        time_key="created_at",
        allowed={
            "schema",
            "model_spec_id",
            "created_at",
            "candidate",
            "model",
            "license",
            "repository",
            "weights",
            "adapter",
            "tasks",
            "constraints",
        },
    )
    if root is None:
        return
    reference(
        validator,
        root.get("candidate"),
        "$.candidate",
        id_key="candidate_id",
    )
    model = validator.object(
        root.get("model"),
        "$.model",
        allowed={"model_id", "name", "description"},
    )
    if model is not None:
        validator.identifier(model.get("model_id"), "$.model.model_id")
        validator.text(model.get("name"), "$.model.name")
        validator.text(model.get("description"), "$.model.description")
    license_info = validator.object(
        root.get("license"),
        "$.license",
        allowed={"expression", "status", "notes"},
    )
    if license_info is not None:
        validator.text(license_info.get("expression"), "$.license.expression")
        validator.enum(
            license_info.get("status"),
            LICENSE_STATUSES,
            "$.license.status",
        )
        validator.text(
            license_info.get("notes"),
            "$.license.notes",
            allow_empty=True,
        )
    repository(validator, root.get("repository"), "$.repository")
    weights(validator, root.get("weights"), "$.weights")
    adapter(validator, root.get("adapter"), "$.adapter")
    validator.string_list(
        root.get("tasks"),
        "$.tasks",
        non_empty=True,
        identifiers=True,
    )
    constraints = validator.object(
        root.get("constraints"),
        "$.constraints",
        allowed={
            "requires_gpu",
            "min_gpu_memory_gb",
            "platforms",
            "notes",
        },
    )
    if constraints is not None:
        validator.boolean(
            constraints.get("requires_gpu"),
            "$.constraints.requires_gpu",
        )
        validator.number(
            constraints.get("min_gpu_memory_gb"),
            "$.constraints.min_gpu_memory_gb",
        )
        validator.string_list(
            constraints.get("platforms"),
            "$.constraints.platforms",
            non_empty=True,
            allowed=PLATFORMS,
        )
        validator.string_list(
            constraints.get("notes"),
            "$.constraints.notes",
        )
    validator.finish()


class ModelSpec(ImmutableContract):
    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ModelSpec":
        validate_model_spec(payload)
        return cls._from_validated(
            payload,
            id_key="model_spec_id",
            time_key="created_at",
        )


def load_model_spec(path: str | Path) -> ModelSpec:
    return ModelSpec.from_payload(load_json(path))
