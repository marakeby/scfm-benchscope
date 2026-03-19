"""Logging configuration for parallel experiment runner."""

import logging
from pathlib import Path

from parallel_experiments.config import LOG_FILE


def setup_logging() -> logging.Logger:
    """Set up logging to both console and file.

    Returns:
        Configured logger instance.
    """
    # Create formatter
    log_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Setup root logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # File handler - appends to file so you can see history
    file_handler = logging.FileHandler(LOG_FILE, mode="a")
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    return logger
