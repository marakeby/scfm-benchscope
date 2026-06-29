from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

Json = Dict[str, Any]


def _utc_now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_git_info(repo_root: str) -> Json:
    """Return commit + dirty flag, or {} if git is unavailable."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return {"commit": commit, "dirty": bool(status)}
    except Exception:
        return {}


def _coerce_cv_fold_index(fold_id: Any, fallback_index: int) -> int:
    """Turn fold labels (0, 'fold_1', numpy int, …) into a stable int."""
    if isinstance(fold_id, bool):
        return fallback_index
    if isinstance(fold_id, int):
        return fold_id
    try:
        import numpy as np  # type: ignore

        if isinstance(fold_id, np.integer):
            return int(fold_id)
    except Exception:
        pass
    text = str(fold_id).strip()
    if text.isdigit():
        return int(text)
    trailing_digits = re.search(r"(\d+)$", text)
    if trailing_digits:
        return int(trailing_digits.group(1))
    return fallback_index


def _maybe_float(x: Any) -> Any:
    """Unwrap numpy/pandas scalars for JSON."""
    try:
        if hasattr(x, "item"):
            return x.item()
    except Exception:
        pass
    return x


def _metrics_from_mapping(m: Any) -> Json:
    return {str(k): _maybe_float(v) for k, v in dict(m).items()}


@dataclass
class FoldResult:
    fold: int
    metrics: Json
    seed: Optional[int] = None
    fold_label: Optional[str] = None
    artifacts: Json = field(default_factory=dict)


@dataclass
class EvaluationResult:
    kind: str
    variant: str
    split: str
    target: Json
    aggregate_metrics: Json
    folds: List[FoldResult] = field(default_factory=list)
    primary: Optional[Json] = None
    artifacts: Json = field(default_factory=dict)
    status: str = "success"
    errors: List[Json] = field(default_factory=list)
    aggregate_std: Optional[Json] = None
    aggregate_median: Optional[Json] = None
    notes: Json = field(default_factory=dict)

    def to_json(self) -> Json:
        data = asdict(self)
        std = data.pop("aggregate_std", None)
        median = data.pop("aggregate_median", None)
        notes = data.pop("notes") or {}

        aggregate: Json = {"metrics": data.pop("aggregate_metrics")}
        if std:
            aggregate["metrics_std"] = std
        if median:
            aggregate["metrics_median"] = median
        data["aggregate"] = aggregate
        if notes:
            data["notes"] = notes
        return data


EMBEDDING_BOOTSTRAP_RUNS_CSV = "embedding_bootstrap_runs.csv"
EMBEDDING_BOOTSTRAP_META_JSON = "embedding_bootstrap_meta.json"


def parse_embedding_bootstrap_evaluation(
    save_dir: str,
    *,
    embedding_key: Optional[str] = None,
) -> Optional[EvaluationResult]:
    """One embedding evaluation from bootstrap CSV + optional meta JSON, if files exist."""
    csv_path = os.path.join(save_dir, EMBEDDING_BOOTSTRAP_RUNS_CSV)
    meta_path = os.path.join(save_dir, EMBEDDING_BOOTSTRAP_META_JSON)
    if not os.path.isfile(csv_path):
        return None

    try:
        import pandas as pd  # type: ignore
    except Exception:
        return None

    notes: Json = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                notes = json.load(f)
        except Exception:
            notes = {}

    resolved_key = embedding_key or notes.get("embedding_key")
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if "run" not in df.columns:
        df = df.reset_index().rename(columns={"index": "run"})

    metric_cols = [
        c for c in df.columns if c != "run" and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not metric_cols:
        return None

    folds: List[FoldResult] = []
    for idx, row in df.iterrows():
        run_id = int(row["run"]) if "run" in df.columns else int(idx)
        row_metrics = {str(c): _maybe_float(row[c]) for c in metric_cols}
        folds.append(FoldResult(fold=run_id, metrics=row_metrics))

    mean = df[metric_cols].mean(numeric_only=True)
    std = df[metric_cols].std(numeric_only=True, ddof=1)
    median = df[metric_cols].median(numeric_only=True)

    agg_mean = _metrics_from_mapping(mean)
    agg_std = {str(k): _maybe_float(v) for k, v in std.items() if pd.notna(v)}
    agg_med = _metrics_from_mapping(median)

    artifact_tables = [
        {"name": "embedding_bootstrap_runs", "path": EMBEDDING_BOOTSTRAP_RUNS_CSV},
    ]
    artifact_files = []
    if os.path.isfile(meta_path):
        artifact_files.append(
            {"name": "embedding_bootstrap_meta", "path": EMBEDDING_BOOTSTRAP_META_JSON}
        )

    return EvaluationResult(
        kind="embedding",
        variant="subsample_repeated",
        split="all",
        target={"embedding_key": resolved_key} if resolved_key else {},
        aggregate_metrics=agg_mean,
        aggregate_std=agg_std or None,
        aggregate_median=agg_med or None,
        folds=folds,
        artifacts={"tables": artifact_tables, "files": artifact_files},
        notes=notes,
    )


def _run_dir_artifact_files(save_dir: str) -> List[Json]:
    """Standard optional files next to results.json."""
    out: List[Json] = []
    for name, rel in (
        ("resolved_config", "resolved_config.yaml"),
        ("run_summary", "run_summary.json"),
        ("metrics_legacy", "metrics.json"),
    ):
        if os.path.exists(os.path.join(save_dir, rel)):
            out.append({"name": name, "path": rel})
    return out


def _parse_cv_metrics_csv(path: str) -> Optional[EvaluationResult]:
    """Parse one ``*_cv_metrics.csv`` with columns fold, Metrics, <model>."""
    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(path)
    except Exception:
        return None

    if "fold" not in df.columns or "Metrics" not in df.columns:
        return None

    model_cols = [c for c in df.columns if c not in ("fold", "Metrics")]
    if not model_cols:
        return None
    model_col = model_cols[0]
    fname = os.path.basename(path)

    try:
        folds: List[FoldResult] = []
        for i, (fold_id, chunk) in enumerate(df.groupby("fold")):
            series = chunk.set_index("Metrics")[model_col]
            metrics = _metrics_from_mapping(series.to_dict())
            label = fold_id if isinstance(fold_id, str) else None
            folds.append(
                FoldResult(
                    fold=_coerce_cv_fold_index(fold_id, i),
                    metrics=metrics,
                    fold_label=label,
                )
            )

        agg_series = df.groupby("Metrics")[model_col].mean()
        aggregate = _metrics_from_mapping(agg_series.to_dict())

        return EvaluationResult(
            kind="classification",
            variant="base",
            split="cv",
            target={},
            aggregate_metrics=aggregate,
            folds=folds,
            artifacts={"tables": [{"name": "cv_metrics", "path": os.path.join("cv", fname)}]},
        )
    except Exception:
        return None


def _parse_classification_metrics(save_dir: str) -> List[EvaluationResult]:
    """Fold + aggregate metrics from classification outputs; never raises."""
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return []

    cv_dir = os.path.join(save_dir, "cv")
    if os.path.isdir(cv_dir):
        for fname in sorted(os.listdir(cv_dir)):
            if not fname.endswith("_cv_metrics.csv"):
                continue
            evaluation = _parse_cv_metrics_csv(os.path.join(cv_dir, fname))
            if evaluation is not None:
                return [evaluation]

    out: List[EvaluationResult] = []
    for fname in sorted(os.listdir(save_dir)):
        if not fname.startswith("cls_metrics_") or not fname.endswith(".csv"):
            continue
        path = os.path.join(save_dir, fname)
        try:
            df = pd.read_csv(path, index_col=0)
        except Exception:
            continue
        if df.shape[1] < 1:
            continue
        col = df.columns[0]
        metrics = _metrics_from_mapping(df[col].to_dict())
        out.append(
            EvaluationResult(
                kind="classification",
                variant="base",
                split="test",
                target={},
                aggregate_metrics=metrics,
                folds=[],
                artifacts={"tables": [{"name": fname.replace(".csv", ""), "path": fname}]},
            )
        )
    return out


def build_results_json(
    *,
    run_id: str,
    save_dir: str,
    repo_root: str,
    config_path: Optional[str],
    resolved_config_path: Optional[str],
    resolved_config: Optional[Json],
    dataset_cfg: Optional[Json],
    embedding_cfg: Optional[Json],
    task_cfg: Optional[Json],
    embedding_key: Optional[str],
    embedding_metrics: Optional[Json],
    rng_seed: int,
    timing: Optional[Json] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
    status: str = "success",
    errors: Optional[List[Json]] = None,
) -> Json:
    """Full results.json envelope (schema v1.1.0). Omits unknown fields gracefully."""
    errors = errors or []

    results: Json = {
        "schema": {"name": "scfm_eval.results", "version": "1.1.0"},
        "run": {
            "run_id": run_id,
            "started_at": started_at or _utc_now_rfc3339(),
            "finished_at": finished_at,
            "status": status,
            "errors": errors,
        },
        "provenance": {
            "scfm_eval_version": os.environ.get("SCFM_EVAL_VERSION") or "unknown",
            "git": _safe_git_info(repo_root),
            "python": {
                "version": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "env": {"pixi_env": os.environ.get("PIXI_ENVIRONMENT")},
        },
        "inputs": {
            "config": {
                "config_path": config_path,
                "resolved_config_path": resolved_config_path,
            },
            "dataset": dataset_cfg or {},
            "embedding": embedding_cfg or {},
            "task": task_cfg or {},
            "rng": {"seed": int(rng_seed)},
        },
        "artifacts": {
            "run_dir": save_dir,
            "files": _run_dir_artifact_files(save_dir),
        },
        "evaluations": [],
        "timing": timing or {},
    }

    evaluations: List[Json] = []

    if isinstance(embedding_metrics, dict):
        evaluations.append(
            EvaluationResult(
                kind="embedding",
                variant="base",
                split="all",
                target={"embedding_key": embedding_key} if embedding_key else {},
                aggregate_metrics=_metrics_from_mapping(embedding_metrics),
                folds=[],
            ).to_json()
        )

    bootstrap = parse_embedding_bootstrap_evaluation(save_dir, embedding_key=embedding_key)
    if bootstrap is not None:
        evaluations.append(bootstrap.to_json())

    for clf in _parse_classification_metrics(save_dir):
        if embedding_key and "embedding_key" not in clf.target:
            clf.target["embedding_key"] = embedding_key
        evaluations.append(clf.to_json())

    results["evaluations"] = evaluations

    if resolved_config is not None:
        try:
            results["inputs"]["config"]["resolved_config"] = resolved_config
        except Exception:
            pass

    if results["run"]["finished_at"] is None:
        results["run"]["finished_at"] = finished_at or _utc_now_rfc3339()

    return results


def write_results_json(path: str, payload: Json) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_maybe_float)


def read_run_inputs_for_results_json(
    run_dir: str,
    *,
    repo_root: str,
) -> Dict[str, Any]:
    """Load kwargs for ``build_results_json`` from a finished run directory."""
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None  # type: ignore

    with open(os.path.join(run_dir, "run_summary.json")) as f:
        summary = json.load(f)

    resolved: Optional[Json] = None
    rc_path = os.path.join(run_dir, "resolved_config.yaml")
    if yaml is not None and os.path.isfile(rc_path):
        try:
            with open(rc_path) as fy:
                resolved = yaml.safe_load(fy)
        except Exception:
            resolved = None
    if not isinstance(resolved, dict):
        resolved = None

    metrics_path = os.path.join(run_dir, "metrics.json")
    metrics_legacy: Json = {}
    if os.path.isfile(metrics_path):
        try:
            with open(metrics_path) as f:
                metrics_legacy = json.load(f)
        except Exception:
            metrics_legacy = {}

    embedding_metrics = None
    if isinstance(metrics_legacy, dict):
        embedding_metrics = metrics_legacy.get("embedding_metrics")

    return dict(
        run_id=str(summary.get("run_id", "")),
        save_dir=run_dir,
        repo_root=repo_root,
        config_path=summary.get("config_path"),
        resolved_config_path="resolved_config.yaml" if os.path.isfile(rc_path) else None,
        resolved_config=resolved,
        dataset_cfg=resolved.get("dataset") if resolved else None,
        embedding_cfg=resolved.get("embedding") if resolved else None,
        task_cfg=resolved.get("task") if resolved else None,
        embedding_key=summary.get("embedding_key"),
        embedding_metrics=embedding_metrics,
        rng_seed=int(summary.get("random_seed", 42)),
        started_at=summary.get("started_at"),
        finished_at=summary.get("finished_at"),
    )


def merge_embedding_bootstrap_into_results_json(
    run_dir: str,
    *,
    embedding_key: Optional[str] = None,
) -> bool:
    """Refresh results.json: replace prior subsample_repeated embedding eval with CSV-derived one."""
    results_path = os.path.join(run_dir, "results.json")
    if not os.path.isfile(results_path):
        return False

    with open(results_path) as f:
        payload = json.load(f)

    evaluations = payload.get("evaluations") or []
    payload["evaluations"] = [
        e
        for e in evaluations
        if not (
            isinstance(e, dict)
            and e.get("kind") == "embedding"
            and e.get("variant") == "subsample_repeated"
        )
    ]

    updated = parse_embedding_bootstrap_evaluation(run_dir, embedding_key=embedding_key)
    if updated is not None:
        payload["evaluations"].append(updated.to_json())

    files = payload.setdefault("artifacts", {}).setdefault("files", [])
    known_names = {entry.get("name") for entry in files if isinstance(entry, dict)}
    for name, relpath in (
        ("embedding_bootstrap_runs", EMBEDDING_BOOTSTRAP_RUNS_CSV),
        ("embedding_bootstrap_meta", EMBEDDING_BOOTSTRAP_META_JSON),
    ):
        if name in known_names:
            continue
        if os.path.isfile(os.path.join(run_dir, relpath)):
            files.append({"name": name, "path": relpath})

    write_results_json(results_path, payload)
    return True
