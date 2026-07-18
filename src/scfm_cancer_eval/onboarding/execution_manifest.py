"""Immutable, bounded input proposed for human-approved execution."""

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
    generated_files,
    repository,
    weights,
)

EXECUTION_MANIFEST_SCHEMA_NAME = "scfm_eval.execution_manifest"
EXECUTION_MANIFEST_SCHEMA_VERSION = "1.0.0"
EXECUTION_STEPS = [
    "checkout",
    "create_environment",
    "install",
    "download_weights",
    "smoke_test",
    "evaluate",
    "collect_results",
]
RETRYABLE_STEPS = set(EXECUTION_STEPS) - {"collect_results"}


def validate_execution_manifest(payload: Any) -> None:
    validator, root = validate_envelope(
        payload,
        contract_name="execution manifest",
        schema_name=EXECUTION_MANIFEST_SCHEMA_NAME,
        schema_version=EXECUTION_MANIFEST_SCHEMA_VERSION,
        id_key="manifest_id",
        time_key="created_at",
        allowed={
            "schema",
            "manifest_id",
            "created_at",
            "model_spec_fingerprint",
            "integration_plan_fingerprint",
            "repository",
            "weights",
            "generated_files",
            "environment",
            "evaluation",
            "steps",
            "resources",
            "permissions",
            "retry_policy",
            "expected_outputs",
        },
    )
    if root is None:
        return
    validator.sha256(
        root.get("model_spec_fingerprint"),
        "$.model_spec_fingerprint",
    )
    validator.sha256(
        root.get("integration_plan_fingerprint"),
        "$.integration_plan_fingerprint",
    )
    repository(validator, root.get("repository"), "$.repository")
    weights(validator, root.get("weights"), "$.weights")
    generated_files(
        validator,
        root.get("generated_files"),
        "$.generated_files",
    )
    _environment(validator, root.get("environment"))
    _evaluation(validator, root.get("evaluation"))

    steps = validator.string_list(
        root.get("steps"),
        "$.steps",
        non_empty=True,
    )
    if steps is not None and steps != EXECUTION_STEPS:
        validator.errors.append(
            "$.steps must use the approved execution order: "
            + ", ".join(EXECUTION_STEPS)
        )

    resource_values = _resources(validator, root.get("resources"))
    _permissions(validator, root.get("permissions"))
    max_attempts = _retry_policy(
        validator,
        root.get("retry_policy"),
    )
    expected_outputs = validator.string_list(
        root.get("expected_outputs"),
        "$.expected_outputs",
        non_empty=True,
    )
    if expected_outputs is not None:
        for index, output_path in enumerate(expected_outputs):
            validator.relative_path(
                output_path,
                f"$.expected_outputs[{index}]",
            )
    _validate_worst_case_budget(
        validator,
        resource_values,
        max_attempts,
    )
    validator.finish()


def _environment(validator: Validator, value: Any) -> None:
    item = validator.object(
        value,
        "$.environment",
        allowed={"name", "lock_path", "lock_sha256"},
    )
    if item is None:
        return
    validator.identifier(item.get("name"), "$.environment.name")
    validator.relative_path(item.get("lock_path"), "$.environment.lock_path")
    validator.sha256(item.get("lock_sha256"), "$.environment.lock_sha256")


def _evaluation(validator: Validator, value: Any) -> None:
    item = validator.object(
        value,
        "$.evaluation",
        allowed={
            "experiment_path",
            "adapter_module",
            "adapter_class",
            "output_subdir",
            "tasks",
        },
    )
    if item is None:
        return
    validator.relative_path(
        item.get("experiment_path"),
        "$.evaluation.experiment_path",
    )
    validator.text(
        item.get("adapter_module"),
        "$.evaluation.adapter_module",
    )
    validator.text(
        item.get("adapter_class"),
        "$.evaluation.adapter_class",
    )
    validator.relative_path(
        item.get("output_subdir"),
        "$.evaluation.output_subdir",
    )
    validator.string_list(
        item.get("tasks"),
        "$.evaluation.tasks",
        non_empty=True,
        identifiers=True,
    )


def _resources(
    validator: Validator,
    value: Any,
) -> dict[str, float | int | None]:
    item = validator.object(
        value,
        "$.resources",
        allowed={
            "gpu_type",
            "gpu_count",
            "disk_gb",
            "max_runtime_minutes",
            "hourly_rate_usd",
            "max_budget_usd",
        },
    )
    values: dict[str, float | int | None] = {}
    if item is None:
        return values
    validator.text(item.get("gpu_type"), "$.resources.gpu_type")
    values["gpu_count"] = validator.number(
        item.get("gpu_count"),
        "$.resources.gpu_count",
        minimum=1,
        integer=True,
    )
    for key in ("disk_gb", "max_runtime_minutes"):
        values[key] = validator.number(
            item.get(key),
            f"$.resources.{key}",
            minimum=0.01,
        )
    for key in ("hourly_rate_usd", "max_budget_usd"):
        values[key] = validator.number(
            item.get(key),
            f"$.resources.{key}",
            minimum=0,
        )
    return values


def _permissions(validator: Validator, value: Any) -> None:
    item = validator.object(
        value,
        "$.permissions",
        allowed={"network_hosts", "secret_names", "dataset_read_only"},
    )
    if item is None:
        return
    validator.hostname_list(
        item.get("network_hosts"),
        "$.permissions.network_hosts",
    )
    validator.secret_list(
        item.get("secret_names"),
        "$.permissions.secret_names",
    )
    read_only = validator.boolean(
        item.get("dataset_read_only"),
        "$.permissions.dataset_read_only",
    )
    if read_only is False:
        validator.errors.append(
            "$.permissions.dataset_read_only must be true"
        )


def _retry_policy(validator: Validator, value: Any) -> int | None:
    item = validator.object(
        value,
        "$.retry_policy",
        allowed={"max_attempts", "retryable_steps"},
    )
    if item is None:
        return None
    attempts = validator.number(
        item.get("max_attempts"),
        "$.retry_policy.max_attempts",
        minimum=1,
        maximum=3,
        integer=True,
    )
    validator.string_list(
        item.get("retryable_steps"),
        "$.retry_policy.retryable_steps",
        allowed=RETRYABLE_STEPS,
    )
    return attempts if isinstance(attempts, int) else None


def _validate_worst_case_budget(
    validator: Validator,
    resources: Mapping[str, float | int | None],
    max_attempts: int | None,
) -> None:
    values = [
        resources.get("gpu_count"),
        resources.get("max_runtime_minutes"),
        resources.get("hourly_rate_usd"),
        resources.get("max_budget_usd"),
        max_attempts,
    ]
    if any(value is None for value in values):
        return
    gpu_count, runtime, hourly_rate, max_budget, attempts = values
    worst_case = (
        float(gpu_count)
        * float(runtime)
        / 60
        * float(hourly_rate)
        * int(attempts)
    )
    if worst_case > float(max_budget) + 1e-9:
        validator.errors.append(
            "$.resources.max_budget_usd is below the worst-case approved "
            f"cost of {worst_case:.2f}"
        )


class ExecutionManifest(ImmutableContract):
    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "ExecutionManifest":
        validate_execution_manifest(payload)
        return cls._from_validated(
            payload,
            id_key="manifest_id",
            time_key="created_at",
        )


def load_execution_manifest(path: str | Path) -> ExecutionManifest:
    return ExecutionManifest.from_payload(load_json(path))
