"""Temporary file and directory management."""

import atexit
import glob
import multiprocessing
import os
import shutil
import tempfile
import time
from pathlib import Path

from setup_path import TEMP_PATH


def configure_temp_directory() -> None:
    """Configure Python's tempfile module to use network storage instead of local /tmp.

    This prevents:
    1. Filling up the local /tmp filesystem (which is often small)
    2. Leaving large temporary h5ad files in /tmp
    3. pymp-* directories accumulating in the project root

    All temporary files will be created in TEMP_PATH (on network storage).
    """
    try:
        # Set Python's default temp directory to network storage
        tempfile.tempdir = str(TEMP_PATH)

        # Also set environment variables for subprocesses
        os.environ["TMPDIR"] = str(TEMP_PATH)
        os.environ["TEMP"] = str(TEMP_PATH)
        os.environ["TMP"] = str(TEMP_PATH)
        os.environ["MP_SHARED_TEMP_DIR"] = str(TEMP_PATH)

        # Set multiprocessing start method if not already set
        if hasattr(multiprocessing, "set_start_method"):
            try:
                multiprocessing.set_start_method("spawn", force=False)
            except RuntimeError:
                # Already set, ignore
                pass
    except Exception:
        # If configuration fails, continue anyway
        pass


def cleanup_temp_files(base_dir: Path | str | None = None) -> None:
    """Clean up temporary files and directories.

    Cleans:
    1. pymp-* directories in the specified base_dir (or cwd)
    2. Old temp files in TEMP_PATH (older than 24 hours)

    Args:
        base_dir: Base directory to search for pymp-* directories.
                  If None, uses current working directory.
    """
    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    # Clean pymp-* directories
    pymp_pattern = str(base_dir / "pymp-*")
    pymp_dirs = glob.glob(pymp_pattern)
    for pymp_dir in pymp_dirs:
        try:
            pymp_path = Path(pymp_dir)
            if pymp_path.is_dir() and pymp_path.name.startswith("pymp-"):
                shutil.rmtree(pymp_path, ignore_errors=True)
        except Exception:
            pass

    # Clean old temp files in TEMP_PATH (older than 24 hours)
    try:
        from parallel_experiments.config import TEMP_CLEANUP_AGE_HOURS

        cutoff_time = time.time() - (TEMP_CLEANUP_AGE_HOURS * 60 * 60)
        for temp_file in TEMP_PATH.iterdir():
            try:
                if temp_file.stat().st_mtime < cutoff_time:
                    if temp_file.is_file():
                        temp_file.unlink()
                    elif temp_file.is_dir():
                        shutil.rmtree(temp_file, ignore_errors=True)
            except Exception:
                pass
    except Exception:
        pass


# Configure temp directory on module import
configure_temp_directory()

# Register cleanup function to run on exit
atexit.register(cleanup_temp_files)
