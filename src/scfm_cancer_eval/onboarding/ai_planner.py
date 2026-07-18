"""Provider-neutral, proposal-only AI integration planner."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from scfm_cancer_eval.onboarding.candidate import ModelCandidate
from scfm_cancer_eval.onboarding.integration_plan import IntegrationPlan
from scfm_cancer_eval.onboarding.model_spec import ModelSpec
from scfm_cancer_eval.onboarding.providers import PlannerProvider

MAX_GENERATED_FILES = 20
MAX_FILE_BYTES = 500_000
MAX_TOTAL_BYTES = 1_000_000
_RESERVED_FILES = {
    "proposal.json",
    "planning-status.json",
    "model-spec.json",
    "integration-plan.json",
}


class PlannerError(ValueError):
    """Raised when an AI proposal cannot become a safe planning workspace."""


@dataclass(frozen=True)
class PlannerOutcome:
    status: str
    provider: str
    model: str
    workspace: Path
    proposal_path: Path
    issues: tuple[str, ...]
    model_spec_path: Path | None = None
    integration_plan_path: Path | None = None
    generated_files: tuple[Path, ...] = ()


@dataclass(frozen=True)
class _GeneratedFile:
    path: str
    purpose: str
    content: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


def build_planner_prompt(candidate: ModelCandidate) -> str:
    """Build a provider-independent research request."""
    candidate_json = json.dumps(candidate.to_dict(), indent=2)
    return f"""Research this single-cell model candidate and propose an integration.

Candidate evidence:
{candidate_json}

This is proposal-only work. Do not claim that you ran installation, model code,
tests, or evaluation. Use web research to inspect the paper, repository,
dependency files, model-loading code, and weight documentation.

Never invent a Git commit, weight checksum, license, dependency, or hardware
measurement. If any required value cannot be verified, return status
"needs_input", list the missing facts in unresolved_fields, and leave
model_spec and integration_plan null.

When all required facts are verified, return status "ready" and:
1. model_spec: a complete scfm_eval.model_spec v1.0.0 object.
2. integration_plan: a complete scfm_eval.integration_plan v1.0.0 object.
3. files: generated text files with path, purpose, and content.

The files must include:
- pixi.toml with a dedicated environment
- integrations/<adapter>.py implementing output_key and fit_transform(loader)
- experiments/<model>.yaml selecting that adapter

Use structured Pixi dependencies and smoke tests. Do not include shell scripts,
secrets, API keys, a pixi.lock fabricated by the model, or an execution
manifest. The framework will replace timestamps, fingerprints, planner
identity, and generated-file hashes deterministically.

Return only one JSON object with this shape:
{{
  "status": "ready" or "needs_input",
  "unresolved_fields": ["..."],
  "research_notes": ["..."],
  "model_spec": {{...}} or null,
  "integration_plan": {{...}} or null,
  "files": [
    {{"path": "pixi.toml", "purpose": "...", "content": "..."}}
  ]
}}
"""


def plan_candidate(
    candidate: ModelCandidate,
    provider: PlannerProvider,
    output_dir: str | Path,
    *,
    created_at: str | None = None,
) -> PlannerOutcome:
    """Generate a reviewable workspace without executing generated code."""
    output = Path(output_dir)
    if output.exists():
        raise PlannerError(f"Planner output already exists: {output}")

    timestamp = created_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00",
        "Z",
    )
    proposal = provider.generate(build_planner_prompt(candidate))
    if not isinstance(proposal, Mapping):
        raise PlannerError("Planner provider must return a JSON object")
    proposal_copy = json.loads(
        json.dumps(proposal, allow_nan=False, ensure_ascii=False)
    )

    status = proposal_copy.get("status")
    if status not in {"ready", "needs_input"}:
        raise PlannerError("Proposal status must be 'ready' or 'needs_input'")
    unresolved = _string_list(
        proposal_copy.get("unresolved_fields"),
        "unresolved_fields",
    )
    notes = _string_list(
        proposal_copy.get("research_notes"),
        "research_notes",
    )

    if status == "needs_input":
        if not unresolved:
            raise PlannerError(
                "needs_input proposals must name unresolved fields"
            )
        return _write_outcome(
            output,
            proposal_copy,
            provider,
            status=status,
            issues=tuple(unresolved + notes),
        )

    if unresolved:
        raise PlannerError("ready proposals cannot have unresolved fields")
    generated = _generated_files(proposal_copy.get("files"))
    model_spec = _model_spec(
        proposal_copy.get("model_spec"),
        candidate,
        timestamp,
    )
    integration_plan = _integration_plan(
        proposal_copy.get("integration_plan"),
        candidate,
        model_spec,
        generated,
        provider,
        timestamp,
        proposal_copy,
    )
    _check_required_files(model_spec, generated)
    return _write_outcome(
        output,
        proposal_copy,
        provider,
        status="ready",
        issues=tuple(notes),
        model_spec=model_spec,
        integration_plan=integration_plan,
        generated=generated,
    )


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise PlannerError(f"{name} must be an array of non-empty strings")
    if len(set(value)) != len(value):
        raise PlannerError(f"{name} must not contain duplicates")
    return list(value)


def _generated_files(value: Any) -> tuple[_GeneratedFile, ...]:
    if not isinstance(value, list) or not value:
        raise PlannerError("ready proposals must include generated files")
    if len(value) > MAX_GENERATED_FILES:
        raise PlannerError(
            f"proposal exceeds the {MAX_GENERATED_FILES}-file limit"
        )

    generated: list[_GeneratedFile] = []
    seen: set[str] = set()
    total_bytes = 0
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise PlannerError(f"files[{index}] must be an object")
        if set(item) != {"path", "purpose", "content"}:
            raise PlannerError(
                f"files[{index}] must contain path, purpose, and content"
            )
        path = item["path"]
        purpose = item["purpose"]
        content = item["content"]
        if not all(
            isinstance(field, str) and field.strip()
            for field in (path, purpose, content)
        ):
            raise PlannerError(
                f"files[{index}] fields must be non-empty strings"
            )
        parsed = PurePosixPath(path)
        if (
            parsed.is_absolute()
            or ".." in parsed.parts
            or path in {".", ""}
            or "\\" in path
            or "\x00" in content
        ):
            raise PlannerError(f"files[{index}] has an unsafe path or content")
        if path in _RESERVED_FILES or not (
            path == "pixi.toml"
            or path.startswith("integrations/")
            or path.startswith("experiments/")
            or path.startswith("tests/")
        ):
            raise PlannerError(
                f"files[{index}] is outside the allowed workspace paths"
            )
        if path in seen:
            raise PlannerError(f"generated file path is duplicated: {path}")
        size = len(content.encode("utf-8"))
        if size > MAX_FILE_BYTES:
            raise PlannerError(f"generated file is too large: {path}")
        total_bytes += size
        seen.add(path)
        generated.append(_GeneratedFile(path, purpose, content))
    if total_bytes > MAX_TOTAL_BYTES:
        raise PlannerError("generated files exceed the total size limit")
    return tuple(sorted(generated, key=lambda item: item.path))


def _model_spec(
    value: Any,
    candidate: ModelCandidate,
    timestamp: str,
) -> ModelSpec:
    if not isinstance(value, dict):
        raise PlannerError("ready proposal must contain model_spec")
    payload = json.loads(json.dumps(value))
    payload["schema"] = {
        "name": "scfm_eval.model_spec",
        "version": "1.0.0",
    }
    payload["created_at"] = timestamp
    payload["candidate"] = {
        "candidate_id": candidate.candidate_id,
        "fingerprint": candidate.fingerprint,
    }
    try:
        return ModelSpec.from_payload(payload)
    except ValueError as exc:
        raise PlannerError(f"Invalid proposed model_spec: {exc}") from exc


def _integration_plan(
    value: Any,
    candidate: ModelCandidate,
    model_spec: ModelSpec,
    generated: tuple[_GeneratedFile, ...],
    provider: PlannerProvider,
    timestamp: str,
    proposal: Mapping[str, Any],
) -> IntegrationPlan:
    if not isinstance(value, dict):
        raise PlannerError("ready proposal must contain integration_plan")
    payload = json.loads(json.dumps(value))
    proposal_digest = hashlib.sha256(
        json.dumps(proposal, sort_keys=True).encode("utf-8")
    ).hexdigest()
    payload["schema"] = {
        "name": "scfm_eval.integration_plan",
        "version": "1.0.0",
    }
    payload["created_at"] = timestamp
    payload["candidate_fingerprint"] = candidate.fingerprint
    payload["model_spec_fingerprint"] = model_spec.fingerprint
    payload["planner"] = {
        "agent": provider.name,
        "backend": provider.model,
        "run_id": f"{candidate.candidate_id}-{proposal_digest[:12]}",
    }
    payload["generated_files"] = [
        {
            "path": item.path,
            "purpose": item.purpose,
            "sha256": item.sha256,
        }
        for item in generated
    ]
    payload["unresolved_fields"] = []
    try:
        return IntegrationPlan.from_payload(payload)
    except ValueError as exc:
        raise PlannerError(
            f"Invalid proposed integration_plan: {exc}"
        ) from exc


def _check_required_files(
    model_spec: ModelSpec,
    generated: tuple[_GeneratedFile, ...],
) -> None:
    paths = {item.path for item in generated}
    payload = model_spec.to_dict()
    adapter_path = payload["adapter"]["module"].replace(".", "/") + ".py"
    if "pixi.toml" not in paths:
        raise PlannerError("ready proposal must generate pixi.toml")
    if adapter_path not in paths:
        raise PlannerError(
            f"ready proposal must generate adapter file {adapter_path}"
        )
    if not any(
        path.startswith("experiments/")
        and path.endswith((".yaml", ".yml"))
        for path in paths
    ):
        raise PlannerError(
            "ready proposal must generate an experiment YAML file"
        )


def _write_outcome(
    output: Path,
    proposal: Mapping[str, Any],
    provider: PlannerProvider,
    *,
    status: str,
    issues: tuple[str, ...],
    model_spec: ModelSpec | None = None,
    integration_plan: IntegrationPlan | None = None,
    generated: tuple[_GeneratedFile, ...] = (),
) -> PlannerOutcome:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            dir=output.parent,
        )
    )
    try:
        _write_json(staging / "proposal.json", proposal)
        status_payload = {
            "status": status,
            "provider": provider.name,
            "model": provider.model,
            "issues": list(issues),
            "model_spec_fingerprint": (
                model_spec.fingerprint if model_spec else None
            ),
            "integration_plan_fingerprint": (
                integration_plan.fingerprint if integration_plan else None
            ),
        }
        _write_json(staging / "planning-status.json", status_payload)
        if model_spec is not None and integration_plan is not None:
            _write_json(staging / "model-spec.json", model_spec.to_dict())
            _write_json(
                staging / "integration-plan.json",
                integration_plan.to_dict(),
            )
            for item in generated:
                path = staging / item.path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(item.content, encoding="utf-8")
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return PlannerOutcome(
        status=status,
        provider=provider.name,
        model=provider.model,
        workspace=output,
        proposal_path=output / "proposal.json",
        issues=issues,
        model_spec_path=(
            output / "model-spec.json" if model_spec is not None else None
        ),
        integration_plan_path=(
            output / "integration-plan.json"
            if integration_plan is not None
            else None
        ),
        generated_files=tuple(output / item.path for item in generated),
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
