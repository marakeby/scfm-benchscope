"""Shared primitives for small, immutable onboarding contracts."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import urlsplit

IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
COMMIT_PATTERN = re.compile(r"^[a-f0-9]{40}$")
SECRET_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


class ContractValidationError(ValueError):
    """Raised when a planning or review document is invalid."""

    def __init__(self, contract_name: str, errors: list[str]):
        self.contract_name = contract_name
        self.errors = tuple(errors)
        super().__init__(
            f"Invalid {contract_name}: " + "; ".join(errors)
        )


class Validator:
    """Collect readable validation errors without failing at the first field."""

    def __init__(self, contract_name: str):
        self.contract_name = contract_name
        self.errors: list[str] = []

    def object(
        self,
        value: Any,
        path: str,
        *,
        allowed: set[str] | None = None,
    ) -> Mapping[str, Any] | None:
        if not isinstance(value, dict):
            self.errors.append(f"{path} must be an object")
            return None
        if allowed is not None:
            for key in sorted(set(value) - allowed):
                self.errors.append(f"{path}.{key} is not supported")
        return value

    def text(
        self,
        value: Any,
        path: str,
        *,
        allow_empty: bool = False,
    ) -> str | None:
        if not isinstance(value, str) or (
            not allow_empty and not value.strip()
        ):
            qualifier = "" if allow_empty else "non-empty "
            self.errors.append(f"{path} must be a {qualifier}string")
            return None
        return value

    def identifier(self, value: Any, path: str) -> str | None:
        text = self.text(value, path)
        if text is not None and not IDENTIFIER_PATTERN.fullmatch(text):
            self.errors.append(
                f"{path} must use lowercase identifier syntax"
            )
        return text

    def enum(
        self,
        value: Any,
        allowed: set[str],
        path: str,
    ) -> str | None:
        if not isinstance(value, str) or value not in allowed:
            self.errors.append(
                f"{path} must be one of " + ", ".join(sorted(allowed))
            )
            return None
        return value

    def boolean(self, value: Any, path: str) -> bool | None:
        if not isinstance(value, bool):
            self.errors.append(f"{path} must be a boolean")
            return None
        return value

    def number(
        self,
        value: Any,
        path: str,
        *,
        minimum: float = 0,
        maximum: float | None = None,
        integer: bool = False,
    ) -> float | int | None:
        valid_type = isinstance(value, int) if integer else isinstance(
            value, (int, float)
        )
        if isinstance(value, bool) or not valid_type:
            kind = "integer" if integer else "number"
            self.errors.append(f"{path} must be a {kind}")
            return None
        if value < minimum:
            self.errors.append(f"{path} must be at least {minimum}")
        if maximum is not None and value > maximum:
            self.errors.append(f"{path} must be at most {maximum}")
        return value

    def timestamp(self, value: Any, path: str) -> str | None:
        text = self.text(value, path)
        if text is None:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            self.errors.append(f"{path} must be an RFC 3339 timestamp")
            return text
        if parsed.tzinfo is None:
            self.errors.append(f"{path} must include a timezone")
        return text

    def sha256(self, value: Any, path: str) -> str | None:
        text = self.text(value, path)
        if text is not None and not SHA256_PATTERN.fullmatch(text):
            self.errors.append(
                f"{path} must be a lowercase SHA-256 digest"
            )
        return text

    def commit(self, value: Any, path: str) -> str | None:
        text = self.text(value, path)
        if text is not None and not COMMIT_PATTERN.fullmatch(text):
            self.errors.append(
                f"{path} must be a full lowercase 40-character Git commit"
            )
        return text

    def public_url(self, value: Any, path: str) -> str | None:
        text = self.text(value, path)
        if text is None:
            return None
        parsed = urlsplit(text)
        if parsed.scheme != "https" or not parsed.hostname:
            self.errors.append(f"{path} must be an absolute HTTPS URL")
            return text
        if parsed.username is not None or parsed.password is not None:
            self.errors.append(f"{path} must not contain credentials")
        hostname = parsed.hostname.rstrip(".").lower()
        if (
            hostname == "localhost"
            or hostname.endswith(".localhost")
            or hostname.endswith(".local")
        ):
            self.errors.append(f"{path} must use a public hostname")
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            self.errors.append(
                f"{path} must not target a private or reserved address"
            )
        return text

    def relative_path(self, value: Any, path: str) -> str | None:
        text = self.text(value, path)
        if text is None:
            return None
        parsed = PurePosixPath(text)
        if parsed.is_absolute() or ".." in parsed.parts or text in {".", ""}:
            self.errors.append(
                f"{path} must be a safe relative POSIX path"
            )
        return text

    def string_list(
        self,
        value: Any,
        path: str,
        *,
        non_empty: bool = False,
        identifiers: bool = False,
        allowed: set[str] | None = None,
    ) -> list[str] | None:
        if not isinstance(value, list):
            self.errors.append(f"{path} must be an array")
            return None
        if non_empty and not value:
            self.errors.append(f"{path} must not be empty")
        seen: set[str] = set()
        result: list[str] = []
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            text = (
                self.identifier(item, item_path)
                if identifiers
                else self.text(item, item_path)
            )
            if text is None:
                continue
            if allowed is not None and text not in allowed:
                self.errors.append(
                    f"{item_path} must be one of "
                    + ", ".join(sorted(allowed))
                )
            if text in seen:
                self.errors.append(f"{item_path} duplicates {text!r}")
            seen.add(text)
            result.append(text)
        return result

    def string_map(
        self,
        value: Any,
        path: str,
    ) -> Mapping[str, str] | None:
        mapping = self.object(value, path)
        if mapping is None:
            return None
        for key, item in mapping.items():
            if not isinstance(key, str) or not key.strip():
                self.errors.append(
                    f"{path} keys must be non-empty strings"
                )
            self.text(item, f"{path}.{key}")
        return mapping

    def hostname_list(self, value: Any, path: str) -> list[str] | None:
        hosts = self.string_list(value, path)
        if hosts is None:
            return None
        for index, host in enumerate(hosts):
            if (
                "://" in host
                or "/" in host
                or "@" in host
                or host == "localhost"
                or host.endswith(".local")
            ):
                self.errors.append(
                    f"{path}[{index}] must be a public hostname"
                )
                continue
            try:
                address = ipaddress.ip_address(host)
            except ValueError:
                address = None
            if address is not None and not address.is_global:
                self.errors.append(
                    f"{path}[{index}] must not be a private or reserved address"
                )
        return hosts

    def secret_list(self, value: Any, path: str) -> list[str] | None:
        secrets = self.string_list(value, path)
        if secrets is None:
            return None
        for index, secret in enumerate(secrets):
            if not SECRET_PATTERN.fullmatch(secret):
                self.errors.append(
                    f"{path}[{index}] must use environment variable syntax"
                )
        return secrets

    def finish(self) -> None:
        if self.errors:
            raise ContractValidationError(
                self.contract_name,
                self.errors,
            )


def validate_envelope(
    payload: Any,
    *,
    contract_name: str,
    schema_name: str,
    schema_version: str,
    id_key: str,
    time_key: str,
    allowed: set[str],
) -> tuple[Validator, Mapping[str, Any] | None]:
    validator = Validator(contract_name)
    root = validator.object(payload, "$", allowed=allowed)
    if root is None:
        validator.finish()
        return validator, None
    try:
        json.dumps(root, allow_nan=False)
    except (TypeError, ValueError) as exc:
        validator.errors.append(
            f"$ must contain only finite JSON values: {exc}"
        )

    schema = validator.object(
        root.get("schema"),
        "$.schema",
        allowed={"name", "version"},
    )
    if schema is not None:
        if schema.get("name") != schema_name:
            validator.errors.append(
                f"$.schema.name must be {schema_name!r}"
            )
        if schema.get("version") != schema_version:
            validator.errors.append(
                f"$.schema.version must be {schema_version!r}"
            )
    validator.identifier(root.get(id_key), f"$.{id_key}")
    validator.timestamp(root.get(time_key), f"$.{time_key}")
    return validator, root


@dataclass(frozen=True)
class ImmutableContract:
    """Canonical JSON document with stable identity and fingerprint."""

    document_id: str
    timestamp: str
    _canonical_json: str

    @classmethod
    def _from_validated(
        cls,
        payload: Mapping[str, Any],
        *,
        id_key: str,
        time_key: str,
    ) -> "ImmutableContract":
        canonical = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return cls(
            document_id=str(payload[id_key]),
            timestamp=str(payload[time_key]),
            _canonical_json=canonical,
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self._canonical_json.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_json)


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as input_file:
        return json.load(input_file)
