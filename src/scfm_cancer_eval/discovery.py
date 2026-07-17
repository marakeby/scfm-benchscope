"""Small bridge from the existing catalog agent to model candidates."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

from scfm_cancer_eval.onboarding import (
    CandidateValidationError,
    ModelCandidate,
)


@dataclass(frozen=True)
class CandidateExport:
    written: tuple[Path, ...]
    existing: tuple[Path, ...]
    errors: tuple[str, ...]


def safe_json_for_html(value: Any) -> str:
    """Serialize untrusted agent text safely inside an HTML script block."""
    return (
        json.dumps(value, ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _public_https(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    hostname = parsed.hostname.rstrip(".").lower()
    if (
        hostname == "localhost"
        or hostname.endswith(".localhost")
        or hostname.endswith(".local")
    ):
        return None
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        return None
    return value


def _candidate_id(model_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")
    if normalized:
        return normalized[:128].rstrip("-")
    suffix = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:12]
    return f"model-{suffix}"


def _confidence(model: dict[str, Any], source_count: int) -> float:
    reported = model.get("confidence")
    if (
        not isinstance(reported, bool)
        and isinstance(reported, (int, float))
        and 0 <= reported <= 1
    ):
        return float(reported)
    return min(0.95, 0.35 + 0.2 * source_count)


def catalog_model_to_candidate(
    model: dict[str, Any],
    *,
    agent: str,
    discovered_at: str,
) -> ModelCandidate:
    """Translate one legacy catalog row without inventing missing evidence."""
    name = str(model.get("model_name") or "").strip()
    if not name:
        raise CandidateValidationError(["$.model.name must be provided"])

    paper_url = _public_https(model.get("paper_url"))
    repository_url = _public_https(model.get("github_url"))
    weights_url = _public_https(model.get("weights_url"))
    source_count = sum(
        value is not None
        for value in (paper_url, repository_url, weights_url)
    )

    paper: dict[str, Any] | None = None
    if paper_url is not None:
        paper = {"url": paper_url}
        title = str(model.get("paper_title") or "").strip()
        if title:
            paper["title"] = title

    repository = (
        {"url": repository_url}
        if repository_url is not None
        else None
    )
    weights = []
    if weights_url is not None:
        weights.append(
            {
                "url": weights_url,
                "kind": "checkpoint",
                "access": "unknown",
                "notes": (
                    "Discovery link; the integration planner must resolve "
                    "exact files and checksums."
                ),
            }
        )

    unresolved = ["evaluation_tasks", "license_compatibility"]
    unresolved.append(
        "immutable_repository_revision"
        if repository is not None
        else "repository_url"
    )
    unresolved.append(
        "weight_file_checksums" if weights else "weights_url"
    )
    if paper is None:
        unresolved.append("paper_url")

    discovery: dict[str, Any] = {
        "agent": agent,
        "source_type": "scheduled_search",
        "confidence": _confidence(model, source_count),
    }
    if paper_url is not None:
        discovery["source_url"] = paper_url

    model_data: dict[str, Any] = {"name": name}
    summary = str(model.get("description") or "").strip()
    if summary:
        model_data["summary"] = summary

    payload = {
        "schema": {
            "name": "scfm_eval.model_candidate",
            "version": "1.0.0",
        },
        "candidate_id": _candidate_id(name),
        "discovered_at": discovered_at,
        "discovery": discovery,
        "model": model_data,
        "sources": {
            "paper": paper,
            "repository": repository,
            "weights": weights,
        },
        "suggested_tasks": [],
        "unresolved_fields": unresolved,
        "notes": (
            "Automatically published discovery evidence. Catalog "
            "classifications remain in docs/models.json."
        ),
    }
    return ModelCandidate.from_payload(payload)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
            json.dump(payload, temp_file, indent=2, ensure_ascii=False)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def export_candidate_records(
    models: Iterable[dict[str, Any]],
    output_root: str | Path,
    *,
    agent: str,
    discovered_at: str | None = None,
) -> CandidateExport:
    """Write new immutable candidate records and retain prior discoveries."""
    timestamp = discovered_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00",
        "Z",
    )
    date = timestamp[:10]
    written: list[Path] = []
    existing: list[Path] = []
    errors: list[str] = []

    for model in models:
        name = str(model.get("model_name") or "<unnamed>")
        try:
            candidate = catalog_model_to_candidate(
                model,
                agent=agent,
                discovered_at=timestamp,
            )
        except (CandidateValidationError, TypeError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
            continue
        path = (
            Path(output_root)
            / date
            / f"{candidate.candidate_id}-{candidate.fingerprint[:12]}.json"
        )
        if path.exists():
            existing.append(path)
            continue
        _write_json_atomic(path, candidate.to_dict())
        written.append(path)

    return CandidateExport(
        written=tuple(written),
        existing=tuple(existing),
        errors=tuple(errors),
    )
