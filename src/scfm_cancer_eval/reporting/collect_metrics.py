"""Aggregate embedding and classification metric CSVs for dashboard exports."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scfm_cancer_eval.reporting.display_names import (
    EXPERIMENT_NAME_MAP,
    MODEL_NAME_MAP,
    map_groups,
)
from scfm_cancer_eval.reporting._io import atomic_write_text

CV_METRICS_FILENAMES: dict[str, list[str]] = {
    "vote": ["vote_cv_metrics.csv", "votecv_metrics.csv"],
    "avg": ["avg_cv_metrics.csv", "avgcv_metrics.csv"],
    "MIL": ["mil_cv_metrics.csv", "milcv_metrics.csv"],
}

_META_COLS = frozenset({"Metrics", "fold"})


def _log(message: str) -> None:
    """Print a progress line immediately (important on slow mounts)."""
    print(message, flush=True)


def _progress(label: str, index: int, total: int, *, every: int = 10) -> None:
    """Emit progress every ``every`` items and on the final item."""
    if total <= 0:
        return
    if index == 1 or index == total or index % every == 0:
        _log(f"  [{index}/{total}] {label}")


@dataclass(frozen=True)
class MetricsExport:
    csv_path: Path
    json_path: Path
    row_count: int


@dataclass(frozen=True)
class CollectMetricsBundle:
    embedding: MetricsExport | None
    classification: MetricsExport | None


def path_has_folder_substring(path: Path, needle: str) -> bool:
    nl = needle.lower()
    return any(nl in part.lower() for part in path.parts)


def write_table_exports(
    df: pd.DataFrame,
    output_stem: Path,
    *,
    schema_name: str,
    schema_version: str = "1.0.0",
) -> MetricsExport:
    """Write CSV + JSON records next to each other from one DataFrame."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    # Avoid Path.with_suffix: stems like ``embedding.metrics`` would lose ``.metrics``.
    csv_path = Path(f"{output_stem}.csv")
    json_path = Path(f"{output_stem}.json")
    df.to_csv(csv_path, index=False)
    payload = {
        "schema": {"name": schema_name, "version": schema_version},
        "row_count": int(len(df)),
        "records": json.loads(df.to_json(orient="records")),
    }
    atomic_write_text(json_path, json.dumps(payload, indent=2) + "\n")
    return MetricsExport(csv_path=csv_path, json_path=json_path, row_count=len(df))


def find_embedding_metrics_files(root: Path, folder_substring: str) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Folder does not exist: {root}")
    _log(
        f"Scanning for embedding_metrics.csv under {root} "
        f"(folder substring {folder_substring!r})…"
    )
    started = time.perf_counter()
    out: list[Path] = []
    seen = 0
    for path in root.rglob("embedding_metrics.csv"):
        seen += 1
        if path_has_folder_substring(path, folder_substring):
            out.append(path)
        if seen == 1 or seen % 25 == 0:
            _log(
                f"  scanned {seen} embedding_metrics.csv candidate(s); "
                f"kept {len(out)} so far"
            )
    elapsed = time.perf_counter() - started
    _log(
        f"Found {len(out)} matching embedding metrics file(s) "
        f"({seen} candidate(s) in {elapsed:.1f}s)."
    )
    return sorted(out)


def load_embedding_run_row(csv_path: Path) -> dict[str, object] | None:
    run_summary_path = csv_path.parent / "run_summary.json"
    try:
        with open(run_summary_path, encoding="utf-8") as f:
            run_summary = json.load(f)
        run_id = run_summary["run_id"]
        exp = Path(run_summary["config_path"]).stem
        model = str(run_id).replace(f"_{exp}", "")
    except OSError as exc:
        print(f"Cannot load run_summary.json, skipping {csv_path}: {exc}", file=sys.stderr)
        return None
    except (KeyError, json.JSONDecodeError) as exc:
        print(f"Invalid run_summary.json, skipping {csv_path}: {exc}", file=sys.stderr)
        return None

    try:
        df = pd.read_csv(csv_path, index_col=0)
    except Exception as exc:
        print(f"Cannot read {csv_path}: {exc}", file=sys.stderr)
        return None

    if df.empty:
        print(f"Empty metrics file, skipping {csv_path}", file=sys.stderr)
        return None

    row: dict[str, object] = {
        "run_id": run_id,
        "model": model,
        "exp": exp,
        "exp_path": str(csv_path.parent.resolve()),
        "embedding_key": ",".join(str(c) for c in df.columns),
        "metrics_path": str(csv_path.resolve()),
    }

    if df.shape[1] == 1:
        series = df.iloc[:, 0]
        for metric_name, val in series.items():
            row[str(metric_name)] = val
    else:
        for col in df.columns:
            series = df[col]
            for metric_name, val in series.items():
                row[f"{col}::{metric_name}"] = val
    return row


def collect_embedding_metrics(
    root: Path,
    *,
    folder_substring: str = "cell_type",
) -> pd.DataFrame:
    files = find_embedding_metrics_files(root, folder_substring)
    if not files:
        return pd.DataFrame()

    _log(f"Loading {len(files)} embedding metrics file(s)…")
    rows: list[dict[str, object]] = []
    for index, csv_path in enumerate(files, start=1):
        _progress(f"read {csv_path.name}", index, len(files))
        rec = load_embedding_run_row(csv_path)
        if rec is not None:
            rows.append(rec)
    _log(f"Loaded {len(rows)}/{len(files)} embedding run(s).")
    if not rows:
        return pd.DataFrame()

    combined = pd.DataFrame(rows)
    combined["group"] = combined["model"].map(map_groups)
    combined["model_display"] = combined["model"].map(MODEL_NAME_MAP)
    combined["exp_display"] = combined["exp"].map(EXPERIMENT_NAME_MAP)

    meta_cols = [
        "run_id",
        "model",
        "model_display",
        "group",
        "exp",
        "exp_path",
        "exp_display",
        "embedding_key",
        "metrics_path",
    ]
    meta_present = [c for c in meta_cols if c in combined.columns]
    metric_cols = [c for c in combined.columns if c not in meta_present]
    return combined[meta_present + sorted(metric_cols, key=str)]


def find_cv_metrics_files(
    root_folder: Path,
    filenames: list[str],
    *,
    include_arxiv: bool,
    strategy: str | None = None,
) -> list[Path]:
    if not root_folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {root_folder}")

    label = strategy or ",".join(filenames)
    _log(f"Scanning for {label} CV metrics under {root_folder}…")
    started = time.perf_counter()
    by_run_dir: dict[Path, Path] = {}
    seen = 0
    for priority, filename in enumerate(filenames):
        for path in root_folder.rglob(filename):
            seen += 1
            if not include_arxiv and "arxiv" in path.as_posix().lower():
                continue
            run_dir = path.parent.parent
            existing = by_run_dir.get(run_dir)
            if existing is None:
                by_run_dir[run_dir] = path
            else:
                existing_priority = filenames.index(existing.name)
                if priority < existing_priority:
                    by_run_dir[run_dir] = path
            if seen == 1 or seen % 25 == 0:
                _log(
                    f"  scanned {seen} {label} candidate(s); "
                    f"kept {len(by_run_dir)} run(s) so far"
                )
    elapsed = time.perf_counter() - started
    _log(
        f"Found {len(by_run_dir)} {label} metrics file(s) "
        f"({seen} candidate(s) in {elapsed:.1f}s)."
    )
    return sorted(by_run_dir.values())


def resolve_score_column(df: pd.DataFrame, preferred: str) -> str | None:
    if preferred in df.columns:
        return preferred
    score_cols = [c for c in df.columns if c not in _META_COLS]
    if len(score_cols) == 1:
        return score_cols[0]
    if not score_cols:
        return None
    print(
        f"Ambiguous classifier columns {score_cols!r}; using {score_cols[0]!r}",
        file=sys.stderr,
    )
    return score_cols[0]


def _load_classification_run_frame(
    matched_file: Path,
    strategy: str,
    score_col: str,
) -> pd.DataFrame | None:
    df = pd.read_csv(matched_file)
    resolved_col = resolve_score_column(df, score_col)
    if resolved_col is None:
        print(
            f"Skipping {matched_file}: no classifier score column "
            f"(have {list(df.columns)})",
            file=sys.stderr,
        )
        return None
    if resolved_col != score_col:
        print(
            f"Using classifier column {resolved_col!r} for {matched_file} "
            f"(preferred {score_col!r} not found).",
            file=sys.stderr,
        )
    df = df.pivot(index="Metrics", columns="fold", values=resolved_col).T
    run_summary_path = matched_file.parent.parent / "run_summary.json"
    try:
        with open(run_summary_path, encoding="utf-8") as f:
            run_summary = json.load(f)
        model = run_summary["run_id"]
        exp = Path(run_summary["config_path"]).stem
        model = model.replace(f"_{exp}", "")
    except OSError as exc:
        print(
            f"Cannot load summary file, skipping: {run_summary_path} ({exc})",
            file=sys.stderr,
        )
        return None
    except (KeyError, json.JSONDecodeError) as exc:
        print(
            f"Invalid run_summary.json, skipping: {run_summary_path} ({exc})",
            file=sys.stderr,
        )
        return None

    df["model"] = model
    df["exp"] = exp
    df["exp_path"] = str(matched_file.parent.parent.resolve())
    df["strategy"] = strategy
    df["classifier"] = resolved_col
    df["metrics_path"] = str(matched_file.resolve())
    return df


def collect_classification_metrics(
    root: Path,
    *,
    score_col: str = "randomforest",
    keep_luad_cancer_stage: bool = False,
    include_arxiv: bool = False,
) -> pd.DataFrame:
    vote_files = find_cv_metrics_files(
        root,
        CV_METRICS_FILENAMES["vote"],
        include_arxiv=include_arxiv,
        strategy="vote",
    )
    avg_files = find_cv_metrics_files(
        root,
        CV_METRICS_FILENAMES["avg"],
        include_arxiv=include_arxiv,
        strategy="avg",
    )
    mil_files = find_cv_metrics_files(
        root,
        CV_METRICS_FILENAMES["MIL"],
        include_arxiv=include_arxiv,
        strategy="MIL",
    )

    frames: list[pd.DataFrame] = []
    for files, strategy in (
        (vote_files, "vote"),
        (mil_files, "MIL"),
        (avg_files, "avg"),
    ):
        if not files:
            _log(f"No {strategy} metrics files to load.")
            continue
        _log(f"Loading {len(files)} {strategy} metrics file(s)…")
        loaded = 0
        for index, matched in enumerate(files, start=1):
            _progress(f"read {matched.name} ({strategy})", index, len(files))
            frame = _load_classification_run_frame(matched, strategy, score_col)
            if frame is not None:
                frames.append(frame)
                loaded += 1
        _log(f"Loaded {loaded}/{len(files)} {strategy} run(s).")

    if not frames:
        return pd.DataFrame()

    _log(f"Combining {len(frames)} classification frame(s)…")
    combined = pd.concat(frames).reset_index().drop(columns=["fold"], errors="ignore")
    if not keep_luad_cancer_stage and "exp" in combined.columns:
        before = len(combined)
        combined = combined[combined["exp"] != "luad_cancer_stage"]
        dropped = before - len(combined)
        if dropped:
            _log(f"Dropped {dropped} luad_cancer_stage row(s).")

    combined["group"] = combined["model"].map(map_groups)
    combined["model_display"] = combined["model"].map(MODEL_NAME_MAP)
    combined["exp_display"] = combined["exp"].map(EXPERIMENT_NAME_MAP)
    return combined.reset_index(drop=True)


def create_collect_metrics_bundle(
    root: str | Path,
    output_dir: str | Path,
    *,
    kind: str = "all",
    folder_substring: str = "cell_type",
    score_col: str = "randomforest",
    keep_luad_cancer_stage: bool = False,
    include_arxiv: bool = False,
) -> CollectMetricsBundle:
    """
    Collect embedding and/or classification metrics under ``root``.

    Writes dashboard-ready CSV/JSON files into ``output_dir``:
    - ``embedding.metrics.csv`` / ``.json``
    - ``classification.metrics.csv`` / ``.json``
    """
    root_path = Path(root)
    out = Path(output_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {root_path}")

    started = time.perf_counter()
    _log(f"Collect metrics: root={root_path}")
    _log(f"Collect metrics: output={out}")
    _log(f"Collect metrics: kind={kind}")

    embedding_export: MetricsExport | None = None
    classification_export: MetricsExport | None = None
    wrote_any = False
    steps: list[str] = []
    if kind in {"all", "embedding"}:
        steps.append("embedding")
    if kind in {"all", "classification"}:
        steps.append("classification")

    for step_index, step in enumerate(steps, start=1):
        _log(f"[{step_index}/{len(steps)}] Collecting {step} metrics…")
        if step == "embedding":
            embedding_df = collect_embedding_metrics(
                root_path, folder_substring=folder_substring
            )
            if embedding_df.empty:
                print(
                    f"No embedding_metrics.csv files found under {root_path} "
                    f"with folder substring {folder_substring!r}.",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                _log(
                    f"Writing embedding exports "
                    f"({len(embedding_df)} row(s)) → {out / 'embedding.metrics'}.*"
                )
                embedding_export = write_table_exports(
                    embedding_df,
                    out / "embedding.metrics",
                    schema_name="scfm_eval.embedding_metrics",
                )
                wrote_any = True
        else:
            classification_df = collect_classification_metrics(
                root_path,
                score_col=score_col,
                keep_luad_cancer_stage=keep_luad_cancer_stage,
                include_arxiv=include_arxiv,
            )
            if classification_df.empty:
                print(
                    f"No classification metrics found under {root_path}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                _log(
                    f"Writing classification exports "
                    f"({len(classification_df)} row(s)) → "
                    f"{out / 'classification.metrics'}.*"
                )
                classification_export = write_table_exports(
                    classification_df,
                    out / "classification.metrics",
                    schema_name="scfm_eval.classification_metrics",
                )
                wrote_any = True

    if not wrote_any:
        raise ValueError(f"No metrics found under {root_path}")

    _log(f"Collect metrics finished in {time.perf_counter() - started:.1f}s.")
    return CollectMetricsBundle(
        embedding=embedding_export,
        classification=classification_export,
    )
