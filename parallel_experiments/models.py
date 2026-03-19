"""Data models for parallel experiment runner."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentResult:
    """Result of a single experiment run.

    Attributes:
        yaml_path: Path to the YAML config file.
        success: Whether the experiment completed successfully.
        skipped: Whether the experiment was skipped (e.g., already exists).
        dry_run: Whether this was a dry run (no actual execution).
        skipped_species_mismatch: Whether skipped due to species mismatch.
        skipped_no_gpu: Whether skipped because no GPU was available.
        error_message: Error message if the experiment failed.
        duration_seconds: Duration of the experiment in seconds.
    """

    yaml_path: Path
    success: bool
    skipped: bool = False
    dry_run: bool = False
    skipped_species_mismatch: bool = False
    skipped_no_gpu: bool = False
    error_message: str | None = None
    duration_seconds: float = 0.0
