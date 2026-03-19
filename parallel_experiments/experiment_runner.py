"""Experiment execution logic."""

import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from parallel_experiments.config import EXPERIMENT_TIMEOUT_SECONDS, LOG_FILE
from parallel_experiments.gpu_manager import acquire_gpu, release_gpu
from parallel_experiments.models import ExperimentResult
from parallel_experiments.status_tracker import update_status_file

logger = logging.getLogger(__name__)


def run_single_experiment(
    yaml_path: Path,
    base_dir: Path,
    yaml_dir: Path,
    skip_existing: bool = True,
    dry_run: bool = False,
    gpu_list: list[int] | None = None,
    defer_viz_summary: bool = True,
) -> ExperimentResult:
    """Run a single experiment.

    Args:
        yaml_path: Path to the YAML config file.
        base_dir: Base directory of the project.
        yaml_dir: Base YAML directory.
        skip_existing: Skip if results already exist (default: True).
        dry_run: Don't actually run, just report what would be done.
        gpu_list: List of available GPU IDs (e.g., [0, 1]). If None, uses all GPUs.
        defer_viz_summary: If True, set SKIP_PLOTTING and SKIP_SUMMARIES so plotting
            and summarization are skipped per run and can be done once at the end.

    Returns:
        ExperimentResult with success/failure status.
    """
    rel_path = yaml_path.relative_to(yaml_dir)
    start_time = datetime.now()

    # NOTE: We no longer check skip_existing here - the filtering is done upfront in main()
    # This avoids race conditions and simplifies the logic

    # Dry run mode
    if dry_run:
        return ExperimentResult(
            yaml_path=yaml_path,
            success=True,
            dry_run=True,
        )

    # Extract method name from YAML path for memory checking
    method_name = rel_path.parts[1] if len(rel_path.parts) > 1 else "unknown"

    # Acquire GPU if GPU list is provided
    acquired_gpu = None
    cuda_visible_devices = None
    if gpu_list:
        acquired_gpu = acquire_gpu(gpu_list, method_name)
        if acquired_gpu is not None:
            cuda_visible_devices = str(acquired_gpu)
        else:
            # No GPU available: at capacity or not enough free memory
            return ExperimentResult(
                yaml_path=yaml_path,
                success=False,
                skipped=True,
                skipped_no_gpu=True,
                error_message="No GPU available (at capacity or insufficient free memory)",
            )

    # Log when experiment STARTS (print to console + append to log file)
    gpu_info = f" [GPU {acquired_gpu}]" if acquired_gpu is not None else ""
    start_msg = (
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO | ▶ STARTING: {rel_path}{gpu_info}"
    )
    print(start_msg, flush=True)
    # Also write to log file (append mode for multiprocess safety)
    with open(LOG_FILE, "a") as f:
        f.write(start_msg + "\n")
    # Update status file to track running experiments
    update_status_file("start", str(rel_path))

    # Run the experiment
    try:
        # Set up environment with CUDA_VISIBLE_DEVICES if specified
        env = os.environ.copy()
        if cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        if defer_viz_summary:
            env["SKIP_PLOTTING"] = "1"
            env["SKIP_SUMMARIES"] = "1"

        # Build subprocess.run arguments
        run_kwargs = {
            "args": [sys.executable, "run/run_exp.py", str(rel_path)],
            "cwd": base_dir,
            "capture_output": True,
            "text": True,
            "env": env,
        }
        # Only add timeout if it's set (None means no timeout)
        if EXPERIMENT_TIMEOUT_SECONDS is not None:
            run_kwargs["timeout"] = EXPERIMENT_TIMEOUT_SECONDS
        
        result = subprocess.run(**run_kwargs)

        duration = (datetime.now() - start_time).total_seconds()

        if result.returncode == 0:
            return ExperimentResult(
                yaml_path=yaml_path,
                success=True,
                duration_seconds=duration,
            )
        else:
            # Capture more of the error for debugging (last 2000 chars to see full traceback)
            error_msg = result.stderr[-2000:] if result.stderr else "Unknown error"

            # Check if it's a species mismatch (should be marked as skipped, not failed)
            is_species_mismatch = (
                "SpeciesMismatchError" in error_msg
                or "species mismatch" in error_msg.lower()
                or "non-human gene" in error_msg.lower()
                or "mouse gene ids" in error_msg.lower()
                or "requires human gene" in error_msg.lower()
            )

            if is_species_mismatch:
                return ExperimentResult(
                    yaml_path=yaml_path,
                    success=True,  # Mark as success since it's an expected skip
                    skipped=True,
                    skipped_species_mismatch=True,
                    error_message=error_msg,
                    duration_seconds=duration,
                )
            else:
                return ExperimentResult(
                    yaml_path=yaml_path,
                    success=False,
                    error_message=error_msg,
                    duration_seconds=duration,
                )

    except subprocess.TimeoutExpired:
        # This exception only occurs if timeout is set
        timeout_msg = f"Timeout after {EXPERIMENT_TIMEOUT_SECONDS} seconds" if EXPERIMENT_TIMEOUT_SECONDS else "Timeout (unexpected)"
        return ExperimentResult(
            yaml_path=yaml_path,
            success=False,
            error_message=timeout_msg,
            duration_seconds=EXPERIMENT_TIMEOUT_SECONDS or 0,
        )
    except Exception as e:
        return ExperimentResult(
            yaml_path=yaml_path,
            success=False,
            error_message=str(e),
        )
    finally:
        # Release GPU if we acquired one
        if acquired_gpu is not None:
            release_gpu(acquired_gpu)
        # Always update status file when experiment ends
        update_status_file("end", str(rel_path))


def run_experiment_wrapper(args: tuple) -> ExperimentResult:
    """Wrapper for ProcessPoolExecutor (unpacks tuple args).

    Args:
        args: Tuple of arguments for run_single_experiment.

    Returns:
        ExperimentResult from the experiment execution.
    """
    return run_single_experiment(*args)
