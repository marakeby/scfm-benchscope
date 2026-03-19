"""GPU management with file-based locking."""

import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

from parallel_experiments.config import (
    GPU_ACQUISITION_BASE_DELAY,
    GPU_ACQUISITION_MAX_DELAY,
    GPU_ACQUISITION_MAX_RETRIES,
    GPU_LOCK_FILE,
    MAX_WORKERS_PER_GPU,
    METHOD_MIN_MEMORY_GB,
)


def get_gpu_memory_free(gpu_id: int) -> float:
    """Get free GPU memory in GB using nvidia-smi.

    Args:
        gpu_id: GPU device ID.

    Returns:
        Free memory in GB, or 0.0 if unable to query.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,nounits,noheader",
                "--id",
                str(gpu_id),
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            memory_mb = float(result.stdout.strip())
            return memory_mb / 1024.0  # Convert MB to GB
    except Exception:
        pass
    return 0.0


def acquire_gpu(
    gpu_list: list[int],
    method_name: str,
    max_workers_per_gpu: int = MAX_WORKERS_PER_GPU,
    min_memory_gb_override: float | None = None,
) -> int | None:
    """Acquire a GPU with file-based locking.

    Ensures max workers per GPU. For memory-heavy methods (STATE, scGPT, Geneformer),
    checks free GPU memory before assigning; tries each GPU in turn and skips if
    none have enough memory (experiment is skipped and can be retried later).

    Args:
        gpu_list: List of available GPU IDs (e.g., [0, 1]).
        method_name: Name of the method/model (e.g., 'state', 'scgpt', 'geneformer').
        max_workers_per_gpu: Maximum workers per GPU (default: 2).
        min_memory_gb_override: If set, overrides per-method minimum (for testing).

    Returns:
        GPU ID if acquired, None if no GPU available (at capacity or insufficient memory).
    """
    if not gpu_list:
        return None

    # Per-method minimum free memory (GB). Methods not in dict are not checked.
    min_memory_gb = (
        min_memory_gb_override
        if min_memory_gb_override is not None
        else METHOD_MIN_MEMORY_GB.get(method_name.lower())
    )
    requires_memory_check = min_memory_gb is not None
    # Geneformer uses ~33+ GiB per run; allow only one worker per GPU to avoid OOM
    effective_max_workers = (
        1
        if (method_name.lower() == "geneformer")
        else max_workers_per_gpu
    )

    # Retry with exponential backoff
    for attempt in range(GPU_ACQUISITION_MAX_RETRIES):
        try:
            # Open file in read-write mode and keep lock for entire operation
            # This ensures atomicity: read-check-increment-write all under one lock
            with open(GPU_LOCK_FILE, "r+") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)

                # Read current GPU usage
                try:
                    f.seek(0)
                    content = f.read()
                    if content.strip():
                        gpu_usage = json.loads(content)
                    else:
                        gpu_usage = {str(gpu): 0 for gpu in gpu_list}
                except (json.JSONDecodeError, ValueError):
                    gpu_usage = {str(gpu): 0 for gpu in gpu_list}

                # Find available GPU (round-robin, first come first served)
                # Try each GPU in order, checking availability
                acquired_gpu = None
                for gpu_id in gpu_list:
                    gpu_str = str(gpu_id)
                    current_workers = gpu_usage.get(gpu_str, 0)

                    # Check if GPU has space (use method-specific max workers)
                    if current_workers >= effective_max_workers:
                        continue

                    # Check free memory for memory-heavy methods (STATE, scGPT, Geneformer)
                    if requires_memory_check:
                        free_memory = get_gpu_memory_free(gpu_id)
                        if free_memory < min_memory_gb:
                            continue  # Skip this GPU, not enough memory

                    # Acquire this GPU
                    gpu_usage[gpu_str] = current_workers + 1
                    acquired_gpu = gpu_id
                    break

                # Write back if we acquired a GPU (still holding the lock)
                if acquired_gpu is not None:
                    f.seek(0)
                    f.truncate()
                    json.dump(gpu_usage, f)
                    f.flush()
                    os.fsync(f.fileno())
                    return acquired_gpu

            # No GPU available, wait and retry with capped exponential backoff
            if attempt < GPU_ACQUISITION_MAX_RETRIES - 1:
                delay = min(
                    GPU_ACQUISITION_BASE_DELAY * (2 ** attempt),
                    GPU_ACQUISITION_MAX_DELAY,
                )
                time.sleep(delay)

        except FileNotFoundError:
            # File doesn't exist yet, create it with initial state
            try:
                initial_usage = {str(gpu): 0 for gpu in gpu_list}
                with open(GPU_LOCK_FILE, "w") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    json.dump(initial_usage, f)
                    f.flush()
                    os.fsync(f.fileno())
                # Retry on next iteration
                if attempt < GPU_ACQUISITION_MAX_RETRIES - 1:
                    delay = min(
                        GPU_ACQUISITION_BASE_DELAY * (2 ** attempt),
                        GPU_ACQUISITION_MAX_DELAY,
                    )
                    time.sleep(delay)
            except Exception:
                if attempt < GPU_ACQUISITION_MAX_RETRIES - 1:
                    delay = min(
                        GPU_ACQUISITION_BASE_DELAY * (2 ** attempt),
                        GPU_ACQUISITION_MAX_DELAY,
                    )
                    time.sleep(delay)
        except Exception:
            if attempt < GPU_ACQUISITION_MAX_RETRIES - 1:
                delay = min(
                    GPU_ACQUISITION_BASE_DELAY * (2 ** attempt),
                    GPU_ACQUISITION_MAX_DELAY,
                )
                time.sleep(delay)
            else:
                # Last attempt failed, return None
                return None

    return None


def release_gpu(gpu_id: int) -> None:
    """Release a GPU by decrementing its usage count.

    Args:
        gpu_id: GPU device ID to release.
    """
    if gpu_id is None:
        return

    for attempt in range(GPU_ACQUISITION_MAX_RETRIES):
        try:
            if GPU_LOCK_FILE.exists():
                with open(GPU_LOCK_FILE, "r+") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        gpu_usage = json.load(f)
                    except (json.JSONDecodeError, ValueError):
                        return  # File corrupted, skip

                    gpu_str = str(gpu_id)
                    current_workers = gpu_usage.get(gpu_str, 0)
                    if current_workers > 0:
                        gpu_usage[gpu_str] = current_workers - 1
                    else:
                        gpu_usage[gpu_str] = 0

                    # Write back
                    f.seek(0)
                    f.truncate()
                    json.dump(gpu_usage, f)
                    f.flush()
                    os.fsync(f.fileno())
                    return
            else:
                return
        except Exception:
            if attempt < GPU_ACQUISITION_MAX_RETRIES - 1:
                time.sleep(GPU_ACQUISITION_BASE_DELAY * (2**attempt))
            else:
                return


def initialize_gpu_lock_file(gpu_list: list[int]) -> None:
    """Initialize the GPU lock file with initial state.

    Args:
        gpu_list: List of available GPU IDs.
    """
    initial_usage = {str(gpu): 0 for gpu in gpu_list}
    GPU_LOCK_FILE.write_text(json.dumps(initial_usage))
