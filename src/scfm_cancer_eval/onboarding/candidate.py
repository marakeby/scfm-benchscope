"""Versioned input contract for model candidates from discovery agents."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

MODEL_CANDIDATE_SCHEMA_NAME = "scfm_eval.model_candidate"
MODEL_CANDIDATE_SCHEMA_VERSION = "1.0.0"

_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_SOURCE_TYPES = {
    "scheduled_search",
    "manual",
    "citation",
    "repository",
    "other",
}
_WEIGHT_KINDS = {
    "checkpoint",
    "tokenizer",
    "vocabulary",
    "config",
    "archive",
    "other",
}
_ACCESS_TYPES = {"public", "gated", "private", "unknown"}


class CandidateValidationError(ValueError):
    """Raised when discovery output violates the candidate contract."""

    def __init__(self, errors: list[str]):
        self.errors = tuple(errors)
        super().__init__("Invalid model candidate: " + "; ".join(errors))


def model_candidate_schema() -> dict[str, Any]:
    """Return the packaged JSON Schema for discovery-agent producers."""
    schema_text = (
        files("scfm_cancer_eval")
        .joinpath("schemas/model-candidate-v1.0.0.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(schema_text)


def _mapping(
    value: Any,
    path: str,
    errors: list[str],
) -> Mapping[str, Any] | None:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object")
        return None
    return value


def _known_fields(
    value: Mapping[str, Any],
    allowed: set[str],
    path: str,
    errors: list[str],
) -> None:
    for key in sorted(set(value) - allowed):
        errors.append(f"{path}.{key} is not supported")


def _text(
    value: Any,
    path: str,
    errors: list[str],
    *,
    required: bool = True,
) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or (required and not value.strip()):
        qualifier = "non-empty " if required else ""
        errors.append(f"{path} must be a {qualifier}string")
        return None
    return value


def _public_https_url(value: Any, path: str, errors: list[str]) -> str | None:
    text = _text(value, path, errors)
    if text is None:
        return None
    parsed = urlsplit(text)
    if parsed.scheme != "https" or not parsed.hostname:
        errors.append(f"{path} must be an absolute HTTPS URL")
        return text
    if parsed.username is not None or parsed.password is not None:
        errors.append(f"{path} must not contain credentials")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".local"):
        errors.append(f"{path} must use a public hostname")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        errors.append(f"{path} must not target a private or reserved address")
    return text


def _timestamp(value: Any, path: str, errors: list[str]) -> None:
    text = _text(value, path, errors)
    if text is None:
        return
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"{path} must be an RFC 3339 timestamp")
        return
    if parsed.tzinfo is None:
        errors.append(f"{path} must include a timezone")


def _unique_text_list(
    value: Any,
    path: str,
    errors: list[str],
    *,
    id_format: bool = False,
) -> None:
    if not isinstance(value, list):
        errors.append(f"{path} must be an array")
        return
    seen: set[str] = set()
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        text = _text(item, item_path, errors)
        if text is None:
            continue
        if id_format and not _ID_PATTERN.fullmatch(text):
            errors.append(f"{item_path} must use lowercase identifier syntax")
        if text in seen:
            errors.append(f"{item_path} duplicates {text!r}")
        seen.add(text)


def _validate_discovery(value: Any, errors: list[str]) -> None:
    discovery = _mapping(value, "$.discovery", errors)
    if discovery is None:
        return
    _known_fields(
        discovery,
        {"agent", "source_type", "source_url", "query", "confidence"},
        "$.discovery",
        errors,
    )
    _text(discovery.get("agent"), "$.discovery.agent", errors)
    source_type = discovery.get("source_type")
    if not isinstance(source_type, str) or source_type not in _SOURCE_TYPES:
        errors.append(
            "$.discovery.source_type must be one of "
            + ", ".join(sorted(_SOURCE_TYPES))
        )
    if "source_url" in discovery:
        _public_https_url(
            discovery.get("source_url"),
            "$.discovery.source_url",
            errors,
        )
    if "query" in discovery:
        _text(
            discovery.get("query"),
            "$.discovery.query",
            errors,
            required=False,
        )
    confidence = discovery.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        errors.append("$.discovery.confidence must be a number from 0 to 1")


def _validate_model(value: Any, errors: list[str]) -> None:
    model = _mapping(value, "$.model", errors)
    if model is None:
        return
    _known_fields(model, {"name", "summary"}, "$.model", errors)
    _text(model.get("name"), "$.model.name", errors)
    if "summary" in model:
        _text(model.get("summary"), "$.model.summary", errors, required=False)


def _validate_paper(value: Any, errors: list[str]) -> bool:
    if value is None:
        return False
    paper = _mapping(value, "$.sources.paper", errors)
    if paper is None:
        return False
    _known_fields(
        paper,
        {"url", "title", "identifier"},
        "$.sources.paper",
        errors,
    )
    _public_https_url(paper.get("url"), "$.sources.paper.url", errors)
    for key in ("title", "identifier"):
        if key in paper:
            _text(
                paper.get(key),
                f"$.sources.paper.{key}",
                errors,
                required=False,
            )
    return True


def _validate_repository(value: Any, errors: list[str]) -> bool:
    if value is None:
        return False
    repository = _mapping(value, "$.sources.repository", errors)
    if repository is None:
        return False
    _known_fields(
        repository,
        {"url", "revision_hint"},
        "$.sources.repository",
        errors,
    )
    _public_https_url(
        repository.get("url"),
        "$.sources.repository.url",
        errors,
    )
    if "revision_hint" in repository:
        _text(
            repository.get("revision_hint"),
            "$.sources.repository.revision_hint",
            errors,
            required=False,
        )
    return True


def _validate_weights(value: Any, errors: list[str]) -> bool:
    if not isinstance(value, list):
        errors.append("$.sources.weights must be an array")
        return False
    urls: set[str] = set()
    for index, item in enumerate(value):
        path = f"$.sources.weights[{index}]"
        weight = _mapping(item, path, errors)
        if weight is None:
            continue
        _known_fields(weight, {"url", "kind", "access", "notes"}, path, errors)
        url = _public_https_url(weight.get("url"), f"{path}.url", errors)
        if url is not None:
            if url in urls:
                errors.append(f"{path}.url duplicates {url!r}")
            urls.add(url)
        kind = weight.get("kind")
        if not isinstance(kind, str) or kind not in _WEIGHT_KINDS:
            errors.append(
                f"{path}.kind must be one of " + ", ".join(sorted(_WEIGHT_KINDS))
            )
        access = weight.get("access")
        if not isinstance(access, str) or access not in _ACCESS_TYPES:
            errors.append(
                f"{path}.access must be one of "
                + ", ".join(sorted(_ACCESS_TYPES))
            )
        if "notes" in weight:
            _text(
                weight.get("notes"),
                f"{path}.notes",
                errors,
                required=False,
            )
    return bool(value)


def _validate_sources(value: Any, errors: list[str]) -> None:
    sources = _mapping(value, "$.sources", errors)
    if sources is None:
        return
    _known_fields(
        sources,
        {"paper", "repository", "weights"},
        "$.sources",
        errors,
    )
    paper_present = _validate_paper(sources.get("paper"), errors)
    repository_present = _validate_repository(
        sources.get("repository"),
        errors,
    )
    weights_present = _validate_weights(sources.get("weights"), errors)
    if not (paper_present or repository_present or weights_present):
        errors.append(
            "$.sources must include a paper, repository, or weight source"
        )


def validate_model_candidate(payload: Any) -> None:
    """Validate discovery evidence without performing network access."""
    errors: list[str] = []
    root = _mapping(payload, "$", errors)
    if root is None:
        raise CandidateValidationError(errors)
    try:
        json.dumps(root, allow_nan=False)
    except (TypeError, ValueError) as exc:
        errors.append(f"$ must contain only finite JSON values: {exc}")

    _known_fields(
        root,
        {
            "schema",
            "candidate_id",
            "discovered_at",
            "discovery",
            "model",
            "sources",
            "suggested_tasks",
            "unresolved_fields",
            "notes",
        },
        "$",
        errors,
    )

    schema = _mapping(root.get("schema"), "$.schema", errors)
    if schema is not None:
        _known_fields(schema, {"name", "version"}, "$.schema", errors)
        if schema.get("name") != MODEL_CANDIDATE_SCHEMA_NAME:
            errors.append(
                f"$.schema.name must be {MODEL_CANDIDATE_SCHEMA_NAME!r}"
            )
        if schema.get("version") != MODEL_CANDIDATE_SCHEMA_VERSION:
            errors.append(
                f"$.schema.version must be {MODEL_CANDIDATE_SCHEMA_VERSION!r}"
            )

    candidate_id = _text(root.get("candidate_id"), "$.candidate_id", errors)
    if candidate_id is not None:
        if len(candidate_id) > 128:
            errors.append("$.candidate_id must be at most 128 characters")
        if not _ID_PATTERN.fullmatch(candidate_id):
            errors.append(
                "$.candidate_id must use lowercase identifier syntax"
            )

    _timestamp(root.get("discovered_at"), "$.discovered_at", errors)
    _validate_discovery(root.get("discovery"), errors)
    _validate_model(root.get("model"), errors)
    _validate_sources(root.get("sources"), errors)
    _unique_text_list(
        root.get("suggested_tasks"),
        "$.suggested_tasks",
        errors,
        id_format=True,
    )
    _unique_text_list(
        root.get("unresolved_fields"),
        "$.unresolved_fields",
        errors,
    )
    if "notes" in root:
        _text(root.get("notes"), "$.notes", errors, required=False)

    if errors:
        raise CandidateValidationError(errors)


@dataclass(frozen=True)
class ModelCandidate:
    """Immutable, validated discovery evidence."""

    candidate_id: str
    discovered_at: str
    model_name: str
    _canonical_json: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ModelCandidate":
        validate_model_candidate(payload)
        canonical_json = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return cls(
            candidate_id=str(payload["candidate_id"]),
            discovered_at=str(payload["discovered_at"]),
            model_name=str(payload["model"]["name"]),
            _canonical_json=canonical_json,
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self._canonical_json.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_json)


def load_model_candidate(path: str | Path) -> ModelCandidate:
    """Read and validate one model candidate JSON file."""
    with Path(path).open(encoding="utf-8") as candidate_file:
        payload = json.load(candidate_file)
    return ModelCandidate.from_payload(payload)
