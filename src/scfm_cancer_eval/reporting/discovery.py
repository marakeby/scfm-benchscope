"""Find and normalize validated ``results.json`` records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from scfm_cancer_eval.contracts import RunResult


@dataclass(frozen=True)
class DiscoveryIssue:
    path: Path
    message: str


class ResultDiscoveryError(ValueError):
    """Raised in strict mode when any requested result cannot be loaded."""

    def __init__(self, issues: Iterable[DiscoveryIssue]):
        self.issues = tuple(issues)
        details = "; ".join(
            f"{issue.path}: {issue.message}" for issue in self.issues
        )
        super().__init__(f"Result discovery failed: {details}")


@dataclass(frozen=True)
class RunSummary:
    """Stable run metadata used to group and compare evaluations."""

    result: RunResult
    model_id: str
    dataset_path: str | None
    task_id: str | None
    started_at: str | None
    finished_at: str | None
    review_status: str = "local"

    @property
    def run_id(self) -> str:
        return self.result.run_id

    @property
    def status(self) -> str:
        return self.result.status

    @classmethod
    def from_result(cls, result: RunResult) -> "RunSummary":
        from scfm_cancer_eval.onboarding.review import load_run_review_status

        payload = result.payload
        inputs = _mapping(payload.get("inputs"))
        embedding = _mapping(inputs.get("embedding"))
        dataset = _mapping(inputs.get("dataset"))
        task = _mapping(inputs.get("task"))
        run = _mapping(payload.get("run"))

        model_id = str(
            embedding.get("method")
            or embedding.get("model_id")
            or "unknown"
        )
        dataset_path = _optional_text(
            dataset.get("path") or dataset.get("dataset_id")
        )
        task_id = _optional_text(
            task.get("task_id")
            or task.get("name")
            or task.get("label_key")
            or dataset.get("label_key")
        )
        return cls(
            result=result,
            model_id=model_id,
            dataset_path=dataset_path,
            task_id=task_id,
            started_at=_optional_text(run.get("started_at")),
            finished_at=_optional_text(run.get("finished_at")),
            review_status=load_run_review_status(result.results_path),
        )


@dataclass(frozen=True)
class DiscoveryResult:
    runs: tuple[RunSummary, ...]
    issues: tuple[DiscoveryIssue, ...]

    @property
    def valid_count(self) -> int:
        return len(self.runs)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _candidate_paths(roots: Iterable[str | Path]) -> tuple[list[Path], list[DiscoveryIssue]]:
    candidates: set[Path] = set()
    issues: list[DiscoveryIssue] = []
    for root_value in roots:
        root = Path(root_value).expanduser()
        if not root.exists():
            issues.append(DiscoveryIssue(root, "path does not exist"))
        elif root.is_file():
            candidates.add(root.resolve())
        else:
            candidates.update(path.resolve() for path in root.rglob("results.json"))
    return sorted(candidates), issues


def discover_results(
    roots: Iterable[str | Path],
    *,
    strict: bool = False,
    accepted_only: bool = False,
) -> DiscoveryResult:
    """Recursively load validated result files from files or directories."""
    from scfm_cancer_eval.onboarding.review import is_publishable_review_status

    candidates, issues = _candidate_paths(roots)
    runs: list[RunSummary] = []

    for path in candidates:
        try:
            summary = RunSummary.from_result(RunResult.from_path(path))
        except Exception as exc:
            issues.append(DiscoveryIssue(path, f"{type(exc).__name__}: {exc}"))
            continue
        if accepted_only and not is_publishable_review_status(
            summary.review_status
        ):
            continue
        runs.append(summary)

    runs.sort(
        key=lambda summary: (
            summary.run_id,
            str(summary.result.results_path),
        )
    )
    issues.sort(key=lambda issue: str(issue.path))
    if strict and issues:
        raise ResultDiscoveryError(issues)
    return DiscoveryResult(runs=tuple(runs), issues=tuple(issues))
