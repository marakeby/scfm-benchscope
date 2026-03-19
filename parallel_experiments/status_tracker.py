"""Status file tracking for running experiments."""

import fcntl
from datetime import datetime
from pathlib import Path

from parallel_experiments.config import STATUS_FILE


def update_status_file(action: str, yaml_path: str) -> None:
    """Update the status file to track running experiments.

    Args:
        action: 'start' or 'end'.
        yaml_path: Relative path to the YAML file.
    """
    try:
        # Use file locking for multiprocess safety
        with open(STATUS_FILE, "a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.seek(0)
            lines = f.readlines()

            if action == "start":
                # Add this experiment to the list
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                lines.append(f"{timestamp} | {yaml_path}\n")
            elif action == "end":
                # Remove this experiment from the list
                lines = [l for l in lines if yaml_path not in l]

            # Rewrite the file
            f.seek(0)
            f.truncate()
            f.writelines(lines)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass  # Don't fail the experiment if status update fails


def initialize_status_file() -> None:
    """Initialize the status file at the start of a new session."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    STATUS_FILE.write_text(
        f"# Session started: {timestamp}\n# Currently running experiments:\n"
    )
