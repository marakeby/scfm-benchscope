"""Reusable field validators for planning documents."""

from __future__ import annotations

from typing import Any

from scfm_cancer_eval.onboarding._contract import Validator

LICENSE_STATUSES = {"compatible", "review_required", "incompatible"}
WEIGHT_ACCESS = {"public", "gated", "private"}
PLATFORMS = {"linux-64"}


def reference(
    validator: Validator,
    value: Any,
    path: str,
    *,
    id_key: str,
) -> None:
    item = validator.object(
        value,
        path,
        allowed={id_key, "fingerprint"},
    )
    if item is None:
        return
    validator.identifier(item.get(id_key), f"{path}.{id_key}")
    validator.sha256(item.get("fingerprint"), f"{path}.fingerprint")


def repository(validator: Validator, value: Any, path: str) -> None:
    item = validator.object(
        value,
        path,
        allowed={"url", "commit"},
    )
    if item is None:
        return
    validator.public_url(item.get("url"), f"{path}.url")
    validator.commit(item.get("commit"), f"{path}.commit")


def weights(validator: Validator, value: Any, path: str) -> None:
    if not isinstance(value, list):
        validator.errors.append(f"{path} must be an array")
        return
    if not value:
        validator.errors.append(f"{path} must not be empty")
    ids: set[str] = set()
    for index, value_item in enumerate(value):
        item_path = f"{path}[{index}]"
        item = validator.object(
            value_item,
            item_path,
            allowed={
                "artifact_id",
                "url",
                "sha256",
                "filename",
                "access",
            },
        )
        if item is None:
            continue
        artifact_id = validator.identifier(
            item.get("artifact_id"),
            f"{item_path}.artifact_id",
        )
        if artifact_id is not None:
            if artifact_id in ids:
                validator.errors.append(
                    f"{item_path}.artifact_id duplicates {artifact_id!r}"
                )
            ids.add(artifact_id)
        validator.public_url(item.get("url"), f"{item_path}.url")
        validator.sha256(item.get("sha256"), f"{item_path}.sha256")
        validator.relative_path(item.get("filename"), f"{item_path}.filename")
        validator.enum(
            item.get("access"),
            WEIGHT_ACCESS,
            f"{item_path}.access",
        )


def adapter(validator: Validator, value: Any, path: str) -> None:
    item = validator.object(
        value,
        path,
        allowed={"module", "class", "output_key"},
    )
    if item is None:
        return
    validator.text(item.get("module"), f"{path}.module")
    validator.text(item.get("class"), f"{path}.class")
    validator.text(item.get("output_key"), f"{path}.output_key")


def environment(validator: Validator, value: Any, path: str) -> None:
    item = validator.object(
        value,
        path,
        allowed={
            "name",
            "platform",
            "python",
            "conda_dependencies",
            "pypi_dependencies",
        },
    )
    if item is None:
        return
    validator.identifier(item.get("name"), f"{path}.name")
    validator.enum(item.get("platform"), PLATFORMS, f"{path}.platform")
    validator.text(item.get("python"), f"{path}.python")
    validator.string_map(
        item.get("conda_dependencies"),
        f"{path}.conda_dependencies",
    )
    validator.string_map(
        item.get("pypi_dependencies"),
        f"{path}.pypi_dependencies",
    )


def generated_files(validator: Validator, value: Any, path: str) -> None:
    if not isinstance(value, list):
        validator.errors.append(f"{path} must be an array")
        return
    if not value:
        validator.errors.append(f"{path} must not be empty")
    seen: set[str] = set()
    for index, value_item in enumerate(value):
        item_path = f"{path}[{index}]"
        item = validator.object(
            value_item,
            item_path,
            allowed={"path", "purpose", "sha256"},
        )
        if item is None:
            continue
        file_path = validator.relative_path(
            item.get("path"),
            f"{item_path}.path",
        )
        if file_path is not None:
            if file_path in seen:
                validator.errors.append(
                    f"{item_path}.path duplicates {file_path!r}"
                )
            seen.add(file_path)
        validator.text(item.get("purpose"), f"{item_path}.purpose")
        validator.sha256(item.get("sha256"), f"{item_path}.sha256")


def estimated_resources(
    validator: Validator,
    value: Any,
    path: str,
) -> None:
    item = validator.object(
        value,
        path,
        allowed={
            "gpu_type",
            "gpu_count",
            "gpu_memory_gb",
            "disk_gb",
            "runtime_minutes",
            "estimated_cost_usd",
        },
    )
    if item is None:
        return
    validator.text(item.get("gpu_type"), f"{path}.gpu_type")
    validator.number(
        item.get("gpu_count"),
        f"{path}.gpu_count",
        minimum=1,
        integer=True,
    )
    for key in (
        "gpu_memory_gb",
        "disk_gb",
        "runtime_minutes",
        "estimated_cost_usd",
    ):
        validator.number(item.get(key), f"{path}.{key}")
