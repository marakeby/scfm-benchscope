"""Configuration constants for parallel experiment runner."""

from pathlib import Path

from setup_path import OUTPUT_PATH

# Allowed models to run (method folder names under yaml/<task>/)
ALLOWED_MODELS = [
    # Foundation models
    "scconcept",
    "scimilarity",
    "scgpt",
    "stack",
    "state",
    "geneformer",
    "scfoundation",
    "uce",
    "omnicell",
    # Integration baselines (first-class methods with their own YAMLs)
    "harmony",
    "bbknn",
    "scanorama",
    "pca_qc",
]

# Log and status file paths
LOG_FILE = OUTPUT_PATH / "parallel_run.log"
STATUS_FILE = OUTPUT_PATH / "parallel_run_status.txt"
GPU_LOCK_FILE = OUTPUT_PATH / "gpu_usage_lock.json"

# Ensure output directory exists
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# GPU management defaults
MAX_WORKERS_PER_GPU = 2
MIN_MEMORY_GB = 10.0  # Minimum free memory for STATE/scGPT models
MIN_MEMORY_GB_GENEFORMER = 35.0  # Geneformer is memory-heavy (~33+ GiB per run)
# Per-method minimum free GPU memory (GB). Methods not listed use no memory check.
METHOD_MIN_MEMORY_GB: dict[str, float] = {
    "state": MIN_MEMORY_GB,
    "scgpt": MIN_MEMORY_GB,
    "geneformer": MIN_MEMORY_GB_GENEFORMER,
}
GPU_ACQUISITION_MAX_RETRIES = 30
GPU_ACQUISITION_BASE_DELAY = 5.0  # Base delay in seconds for exponential backoff
GPU_ACQUISITION_MAX_DELAY = 60.0  # Cap per-retry wait (seconds)

# Experiment execution defaults
EXPERIMENT_TIMEOUT_SECONDS = None  # No timeout (experiments can run indefinitely)

# Temp file cleanup
TEMP_CLEANUP_AGE_HOURS = 24  # Clean up temp files older than 24 hours
