"""Check for existing experiment results."""

import logging
import time
from pathlib import Path

import pandas as pd
import yaml

from parallel_experiments.config import OUTPUT_PATH

logger = logging.getLogger(__name__)

# batch_bio_integration: expected n_metrics from all_results_raw.csv (same as summarizer).
# All runs except technical_repeats should have 11 metrics (batch + bio + annotation); technical_repeats only 2.
BATCH_BIO_N_METRICS_FULL = 11
BATCH_BIO_N_METRICS_TECHNICAL_REPEATS = 2
BATCH_BIO_SUBGROUP_TECHNICAL_REPEATS = "technical_repeats"

# Baseline methods run inside each main-method YAML; all must be complete for (subgroup, dataset)
# so that re-running the main YAML will re-run baselines and fill any missing evaluation outputs.
INTEGRATION_BASELINES_FOR_COMPLETENESS = ("harmony", "scanorama", "bbknn", "pca_qc")


def _find_dir_case_insensitive(parent: Path, target_name: str) -> Path | None:
    """Find a directory with case-insensitive matching.

    Args:
        parent: Parent directory to search in.
        target_name: Target directory name (case-insensitive).

    Returns:
        The actual path if found, None otherwise.
    """
    if not parent.exists():
        return None
    target_lower = target_name.lower()
    for item in parent.iterdir():
        if item.is_dir() and item.name.lower() == target_lower:
            return item
    return None


def _has_valid_metrics(metrics_dir: Path) -> bool:
    """Return True if metrics_dir exists and contains at least one non-empty CSV.

    Uses rglob to also find CSVs in subdirectories (e.g. metrics/survival/*.csv).
    """
    if not metrics_dir.exists() or not metrics_dir.is_dir():
        return False
    try:
        csv_files = list(metrics_dir.rglob("*.csv"))
        if not csv_files:
            return False
        return any(
            f.stat().st_size > 0 for f in csv_files if f.is_file()
        )
    except Exception:
        return False


def _is_batch_bio_integration_complete(metrics_dir: Path) -> bool:
    """Return True if this run has the expected n_metrics (11 for full, 2 for technical_repeats).

    Uses the same notion of completeness as batch_bio_integration/summaries/all_results_raw.csv:
    all runs except technical_repeats should have 11 metrics (batch + bio + annotation); technical_repeats
    only get batch metrics (2). We collect metrics the same way as the summarizer and require
    collected count >= expected.
    """
    if not metrics_dir.exists() or not metrics_dir.is_dir():
        return False
    experiment_dir = metrics_dir.parent
    try:
        rel_parts = experiment_dir.relative_to(OUTPUT_PATH).parts
    except ValueError:
        return False
    # 3-level: task/method/dataset  -> len 3; 4-level: task/method/subgroup/dataset -> len 4
    if len(rel_parts) == 3:
        _task, method_name, dataset_name = rel_parts
        subgroup = None
    elif len(rel_parts) == 4:
        _task, method_name, subgroup, dataset_name = rel_parts
    else:
        return False

    expected = (
        BATCH_BIO_N_METRICS_TECHNICAL_REPEATS
        if subgroup == BATCH_BIO_SUBGROUP_TECHNICAL_REPEATS
        else BATCH_BIO_N_METRICS_FULL
    )

    try:
        from utils.metric_collector import MetricCollector

        collector = MetricCollector()
        metrics_dict = collector.collect_metrics_from_experiment(
            experiment_dir, dataset_name, "batch_bio_integration", method_name
        )
        n_collected = sum(1 for v in metrics_dict.values() if pd.notna(v))
        if n_collected < expected:
            return False
        # For full runs (non-technical_repeats), require annotation metrics to be present.
        # So re-running only re-executes annotation when batch/bio exist but annotation was invalid.
        if subgroup != BATCH_BIO_SUBGROUP_TECHNICAL_REPEATS:
            has_annotation = any(k.startswith("annotation_") for k in metrics_dict if pd.notna(metrics_dict.get(k)))
            if not has_annotation:
                return False
        return True
    except Exception as e:
        logger.debug("batch_bio_integration completeness check failed: %s", e)
        return False


def _are_batch_bio_baselines_complete(
    task_dir: Path, subgroup: str | None, dataset: str
) -> bool:
    """Return True if all integration baseline dirs for (subgroup, dataset) are complete.

    Baselines (harmony, scanorama, bbknn, pca_qc) are produced by the main-method run and have
    no YAMLs. So we only consider a main run "complete" if every baseline dir that exists for
    this (subgroup, dataset) also has the expected n_metrics; otherwise we re-run the main
    YAML to re-run baselines and fill gaps.
    """
    for baseline in INTEGRATION_BASELINES_FOR_COMPLETENESS:
        baseline_method_dir = _find_dir_case_insensitive(task_dir, baseline)
        if baseline_method_dir is None:
            continue
        if subgroup is None:
            baseline_metrics_dir = baseline_method_dir / dataset / "metrics"
        else:
            baseline_metrics_dir = baseline_method_dir / subgroup / dataset / "metrics"
        if not baseline_metrics_dir.exists():
            continue
        if not _is_batch_bio_integration_complete(baseline_metrics_dir):
            return False
    return True


def scan_completed_experiments() -> set[tuple[str, str, str | None, str]]:
    """Scan output directory once and return set of completed experiments.

    Supports both 3-level (task/method/dataset) and 4-level (task/method/subgroup/dataset) output.

    Returns:
        Set of (task, method, subgroup_or_none, dataset) tuples that have metrics files.
    """
    completed: set[tuple[str, str, str | None, str]] = set()

    logger.info("Scanning for completed experiments (this is fast)...")
    start_time = time.time()

    if not OUTPUT_PATH.exists():
        return completed

    for task_dir in OUTPUT_PATH.iterdir():
        if (
            not task_dir.is_dir()
            or task_dir.name.startswith(".")
            or task_dir.name == "summaries"
            or task_dir.name == "embeddings"
        ):
            continue

        task = task_dir.name

        for method_dir in task_dir.iterdir():
            if not method_dir.is_dir() or method_dir.name == "summaries":
                continue

            method = method_dir.name.lower()

            for item in method_dir.iterdir():
                if not item.is_dir() or item.name == "summaries":
                    continue

                metrics_dir = item / "metrics"
                if _has_valid_metrics(metrics_dir):
                    if task == "batch_bio_integration":
                        if not _is_batch_bio_integration_complete(metrics_dir):
                            continue
                        if not _are_batch_bio_baselines_complete(task_dir, None, item.name):
                            continue
                    # 3-level: task/method/dataset
                    completed.add((task, method, None, item.name))
                    continue

                # 4-level: task/method/subgroup/dataset
                subgroup = item.name
                for dataset_dir in item.iterdir():
                    if not dataset_dir.is_dir() or dataset_dir.name == "summaries":
                        continue
                    mdir = dataset_dir / "metrics"
                    if _has_valid_metrics(mdir):
                        if task == "batch_bio_integration":
                            if not _is_batch_bio_integration_complete(mdir):
                                continue
                            if not _are_batch_bio_baselines_complete(task_dir, subgroup, dataset_dir.name):
                                continue
                        completed.add((task, method, subgroup, dataset_dir.name))

    elapsed = time.time() - start_time
    logger.info(f"Found {len(completed)} completed experiments in {elapsed:.1f}s")

    return completed


def check_existing_results(
    yaml_path: Path,
    yaml_dir: Path,
    completed_set: set[tuple[str, str, str | None, str]] | None = None,
) -> bool:
    """Check if experiment results already exist.

    Supports 3-level (task/method/dataset.yaml) and 4-level (task/method/subgroup/dataset.yaml) paths.

    Args:
        yaml_path: Path to the YAML config file.
        yaml_dir: Base YAML directory.
        completed_set: Pre-computed set of (task, method, subgroup_or_none, dataset).
                      If provided, uses O(1) lookup. If None, falls back to filesystem check.

    Returns:
        True if metrics files exist, False otherwise.
    """
    rel_path = yaml_path.relative_to(yaml_dir)
    parts = rel_path.parts

    if len(parts) < 3:
        return False

    task = parts[0]
    method = parts[1].lower()
    # 4-level: task/method/subgroup/dataset.yaml -> subgroup = parts[2]
    subgroup: str | None = parts[2] if len(parts) == 4 else None

    dataset = yaml_path.stem
    try:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
            dataset_cfg = config.get("dataset", {}) if config else {}
            if dataset_cfg.get("dataset_name"):
                dataset = dataset_cfg["dataset_name"]
            # Use task_name from config if available (output dir uses config task_name,
            # which may differ from the YAML directory name, e.g. "batch_denoising" vs "batch_integration")
            if dataset_cfg.get("task_name"):
                task = dataset_cfg["task_name"]
    except Exception:
        pass

    key = (task, method, subgroup, dataset)

    if completed_set is not None:
        is_completed = key in completed_set
        if is_completed:
            task_dir = OUTPUT_PATH / task
            method_dir = _find_dir_case_insensitive(task_dir, method)
            if method_dir is None:
                logger.warning(f"Method dir not found for {task}/{method}/{dataset}")
                return False
            output_dir = method_dir / dataset if subgroup is None else method_dir / subgroup / dataset
            metrics_dir = output_dir / "metrics"
            if not _has_valid_metrics(metrics_dir):
                logger.warning(
                    f"Found {key} in completed set but metrics dir invalid: {metrics_dir}"
                )
                return False
            if task == "batch_bio_integration":
                if not _is_batch_bio_integration_complete(metrics_dir):
                    return False
                if not _are_batch_bio_baselines_complete(task_dir, subgroup, dataset):
                    return False
        return is_completed

    # Slow path: check filesystem
    task_dir = OUTPUT_PATH / task
    method_dir = _find_dir_case_insensitive(task_dir, method)
    if method_dir is None:
        return False

    output_dir = method_dir / dataset if subgroup is None else method_dir / subgroup / dataset
    metrics_dir = output_dir / "metrics"
    if _has_valid_metrics(metrics_dir):
        if task == "batch_bio_integration":
            if not _is_batch_bio_integration_complete(metrics_dir):
                return False
            if not _are_batch_bio_baselines_complete(task_dir, subgroup, dataset):
                return False
        logger.info(
            f"[SKIP REASON] {task}/{method}/{subgroup or ''}/{dataset}: metrics exist"
        )
        return True
    return False
