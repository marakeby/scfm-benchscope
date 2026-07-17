"""Proposal-only AI integration plan contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scfm_cancer_eval.onboarding._contract import (
    ImmutableContract,
    Validator,
    load_json,
    validate_envelope,
)
from scfm_cancer_eval.onboarding._planning_fields import (
    environment,
    estimated_resources,
    generated_files,
)

INTEGRATION_PLAN_SCHEMA_NAME = "scfm_eval.integration_plan"
INTEGRATION_PLAN_SCHEMA_VERSION = "1.0.0"
SMOKE_TEST_KINDS = {"import", "adapter", "tiny_evaluation"}


def validate_integration_plan(payload: Any) -> None:
    validator, root = validate_envelope(
        payload,
        contract_name="integration plan",
        schema_name=INTEGRATION_PLAN_SCHEMA_NAME,
        schema_version=INTEGRATION_PLAN_SCHEMA_VERSION,
        id_key="integration_plan_id",
        time_key="created_at",
        allowed={
            "schema",
            "integration_plan_id",
            "created_at",
            "candidate_fingerprint",
            "model_spec_fingerprint",
            "planner",
            "environment",
            "installation",
            "generated_files",
            "smoke_tests",
            "resources",
            "assumptions",
            "risks",
            "unresolved_fields",
        },
    )
    if root is None:
        return
    validator.sha256(
        root.get("candidate_fingerprint"),
        "$.candidate_fingerprint",
    )
    validator.sha256(
        root.get("model_spec_fingerprint"),
        "$.model_spec_fingerprint",
    )
    planner = validator.object(
        root.get("planner"),
        "$.planner",
        allowed={"agent", "backend", "run_id"},
    )
    if planner is not None:
        validator.text(planner.get("agent"), "$.planner.agent")
        validator.text(planner.get("backend"), "$.planner.backend")
        validator.identifier(planner.get("run_id"), "$.planner.run_id")
    environment(validator, root.get("environment"), "$.environment")
    installation = validator.object(
        root.get("installation"),
        "$.installation",
        allowed={"method", "package_path", "editable", "no_deps"},
    )
    if installation is not None:
        validator.enum(
            installation.get("method"),
            {"pixi"},
            "$.installation.method",
        )
        validator.relative_path(
            installation.get("package_path"),
            "$.installation.package_path",
        )
        validator.boolean(
            installation.get("editable"),
            "$.installation.editable",
        )
        validator.boolean(
            installation.get("no_deps"),
            "$.installation.no_deps",
        )
    generated_files(
        validator,
        root.get("generated_files"),
        "$.generated_files",
    )
    _smoke_tests(validator, root.get("smoke_tests"))
    estimated_resources(validator, root.get("resources"), "$.resources")
    for key in ("assumptions", "risks", "unresolved_fields"):
        validator.string_list(root.get(key), f"$.{key}")
    validator.finish()


def _smoke_tests(validator: Validator, value: Any) -> None:
    if not isinstance(value, list):
        validator.errors.append("$.smoke_tests must be an array")
        return
    if not value:
        validator.errors.append("$.smoke_tests must not be empty")
    for index, value_item in enumerate(value):
        path = f"$.smoke_tests[{index}]"
        item = validator.object(
            value_item,
            path,
            allowed={"test_id", "kind", "timeout_minutes"},
        )
        if item is None:
            continue
        validator.identifier(item.get("test_id"), f"{path}.test_id")
        validator.enum(item.get("kind"), SMOKE_TEST_KINDS, f"{path}.kind")
        validator.number(
            item.get("timeout_minutes"),
            f"{path}.timeout_minutes",
            minimum=1,
            integer=True,
        )


class IntegrationPlan(ImmutableContract):
    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "IntegrationPlan":
        validate_integration_plan(payload)
        return cls._from_validated(
            payload,
            id_key="integration_plan_id",
            time_key="created_at",
        )


def load_integration_plan(path: str | Path) -> IntegrationPlan:
    return IntegrationPlan.from_payload(load_json(path))
