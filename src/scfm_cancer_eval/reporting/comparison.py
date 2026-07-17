"""Normalize discovered evaluations and write portable comparison exports."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from scfm_cancer_eval.reporting.discovery import DiscoveryResult, RunSummary

COMPARISON_SCHEMA_NAME = "scfm_eval.comparison"
COMPARISON_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class ComparisonRecord:
    run_id: str
    run_status: str
    model_id: str
    dataset_path: str | None
    task_id: str | None
    evaluation_kind: str
    evaluation_variant: str
    split: str
    evaluation_status: str
    started_at: str | None
    finished_at: str | None
    results_path: str
    metrics: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonArtifacts:
    json_path: Path
    csv_path: Path
    record_count: int


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _record_for_evaluation(
    summary: RunSummary,
    evaluation: Mapping[str, Any] | None,
) -> ComparisonRecord:
    if evaluation is None:
        kind = "run"
        variant = ""
        split = ""
        status = summary.status
        metrics: Mapping[str, Any] = {}
    else:
        aggregate = _mapping(evaluation.get("aggregate"))
        kind = str(evaluation.get("kind") or "unknown")
        variant = str(evaluation.get("variant") or "")
        split = str(evaluation.get("split") or "")
        status = str(evaluation.get("status") or summary.status)
        metrics = _mapping(aggregate.get("metrics"))

    return ComparisonRecord(
        run_id=summary.run_id,
        run_status=summary.status,
        model_id=summary.model_id,
        dataset_path=summary.dataset_path,
        task_id=summary.task_id,
        evaluation_kind=kind,
        evaluation_variant=variant,
        split=split,
        evaluation_status=status,
        started_at=summary.started_at,
        finished_at=summary.finished_at,
        results_path=str(summary.result.results_path),
        metrics=dict(metrics),
    )


def build_comparison_records(
    discovery: DiscoveryResult,
) -> tuple[ComparisonRecord, ...]:
    records: list[ComparisonRecord] = []
    for summary in discovery.runs:
        evaluations = summary.result.evaluations
        if evaluations:
            records.extend(
                _record_for_evaluation(summary, evaluation)
                for evaluation in evaluations
            )
        else:
            records.append(_record_for_evaluation(summary, None))
    records.sort(
        key=lambda record: (
            record.dataset_path or "",
            record.task_id or "",
            record.evaluation_kind,
            record.evaluation_variant,
            record.model_id,
            record.run_id,
            record.results_path,
        )
    )
    return tuple(records)


def build_comparison_payload(
    discovery: DiscoveryResult,
    records: tuple[ComparisonRecord, ...] | None = None,
) -> dict[str, Any]:
    resolved_records = records or build_comparison_records(discovery)
    return {
        "schema": {
            "name": COMPARISON_SCHEMA_NAME,
            "version": COMPARISON_SCHEMA_VERSION,
        },
        "summary": {
            "run_count": discovery.valid_count,
            "record_count": len(resolved_records),
            "issue_count": len(discovery.issues),
        },
        "issues": [
            {"path": str(issue.path), "message": issue.message}
            for issue in discovery.issues
        ],
        "records": [record.to_json() for record in resolved_records],
    }


def _csv_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def comparison_csv_text(records: tuple[ComparisonRecord, ...]) -> str:
    fixed_columns = [
        "run_id",
        "run_status",
        "model_id",
        "dataset_path",
        "task_id",
        "evaluation_kind",
        "evaluation_variant",
        "split",
        "evaluation_status",
        "started_at",
        "finished_at",
        "results_path",
    ]
    metric_names = sorted(
        {str(name) for record in records for name in record.metrics}
    )
    metric_columns = [f"metric__{name}" for name in metric_names]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fixed_columns + metric_columns)
    writer.writeheader()
    for record in records:
        row = {
            key: _csv_value(value)
            for key, value in record.to_json().items()
            if key != "metrics"
        }
        row.update(
            {
                f"metric__{name}": _csv_value(record.metrics.get(name))
                for name in metric_names
            }
        )
        writer.writerow(row)
    return output.getvalue()


def _atomic_write_text(path: Path, text: str) -> None:
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
            temp_file.write(text)
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


def write_comparison_exports(
    discovery: DiscoveryResult,
    output_dir: str | Path,
) -> ComparisonArtifacts:
    output_root = Path(output_dir)
    records = build_comparison_records(discovery)
    payload = build_comparison_payload(discovery, records)
    json_path = output_root / "comparison.json"
    csv_path = output_root / "comparison.csv"
    _atomic_write_text(
        json_path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write_text(csv_path, comparison_csv_text(records))
    return ComparisonArtifacts(
        json_path=json_path,
        csv_path=csv_path,
        record_count=len(records),
    )
