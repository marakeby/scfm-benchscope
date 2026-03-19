"""
Utility functions for experiment runner.
"""
import os
import glob
import random
import shutil
import tempfile
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml

from setup_path import TEMP_PATH


def _expand_env_in_config(obj: Any) -> Any:
    """Recursively expand ${VAR} and $VAR in string values using os.environ."""
    if isinstance(obj, dict):
        return {k: _expand_env_in_config(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_in_config(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj

# Module directory path
dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def configure_temp_directory() -> None:
    """
    Configure temporary directories for optimal performance.
    
    Strategy:
    - Use local /tmp for multiprocessing temp files (TMPDIR) to avoid NFS cleanup issues
      (pymp-* directories on NFS cause "Device or resource busy" errors)
    - Use TEMP_PATH (network storage) only for explicit large temp files (h5ad caches)
      that benefit from network storage persistence
    
    This prevents:
    1. OSError: [Errno 16] Device or resource busy errors from NFS multiprocessing cleanup
    2. pymp-* directories accumulating with stale NFS handles
    3. Large temp files filling up local /tmp
    """
    try:
        # Ensure TEMP_PATH exists for explicit large temp file operations
        TEMP_PATH.mkdir(parents=True, exist_ok=True)
        
        # Use LOCAL /tmp for multiprocessing to avoid NFS cleanup issues
        # Python's multiprocessing creates pymp-* directories in TMPDIR,
        # and NFS file locking causes "Device or resource busy" errors on cleanup
        local_tmp = '/tmp'
        os.environ['TMPDIR'] = local_tmp
        os.environ['TEMP'] = local_tmp
        os.environ['TMP'] = local_tmp
        
        # Don't override tempfile.tempdir - let it use system default (/tmp)
        # This ensures multiprocessing uses local storage
        # For explicit large temp files, use TEMP_PATH directly in code
        
        # Set multiprocessing start method to 'spawn' for cleaner process handling
        if hasattr(multiprocessing, 'set_start_method'):
            try:
                multiprocessing.set_start_method('spawn', force=False)
            except RuntimeError:
                # Already set, ignore
                pass
    except Exception:
        # If configuration fails, continue anyway
        pass


def cleanup_temp_files() -> None:
    """
    Clean up temporary files and directories.
    
    Cleans:
    1. pymp-* directories in the project root
    2. Old temp files in TEMP_PATH (older than 24 hours)
    """
    import time
    
    # Clean pymp-* directories in project root
    pymp_pattern = str(Path(dir_path) / 'pymp-*')
    pymp_dirs = glob.glob(pymp_pattern)
    for pymp_dir in pymp_dirs:
        try:
            pymp_path = Path(pymp_dir)
            if pymp_path.is_dir() and pymp_path.name.startswith('pymp-'):
                shutil.rmtree(pymp_path, ignore_errors=True)
        except Exception:
            pass
    
    # Clean old temp files in TEMP_PATH (older than 24 hours)
    try:
        cutoff_time = time.time() - (24 * 60 * 60)  # 24 hours ago
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



# List to store timing records
def _get_timing_log():
    """Return the global timing log list."""
    global _timing_log
    if '_timing_log' not in globals():
        _timing_log = []
    return _timing_log

_timing_log = _get_timing_log()


def timing(func):
    """
    Decorator to measure and store execution time of a function.
    Appends timing info to the global _timing_log list.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        _timing_log.append({
            'function': func.__name__,
            'time_seconds': elapsed
        })
        return res
    return wrapper


def set_random_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
        deterministic (bool): If True, sets PyTorch to deterministic mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


def get_configs(config_path: str) -> Tuple[Any, ...]:
    """
    Load experiment configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        Tuple containing run_id, data_config, qc_config, preproc_config, hvg_config, feat_config, evaluations_config.

    Raises:
        ValueError: If the file is empty or does not contain a valid YAML mapping.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    if config is None or not isinstance(config, dict):
        raise ValueError(
            f"Config file is empty or invalid (expected a YAML mapping): {config_path}. "
            "Ensure the file contains key-value pairs (e.g. run_id, dataset, embedding, evaluations)."
        )

    config = _expand_env_in_config(config)

    run_id = config.get("run_id", None)
    data_config = config.get('dataset', {})
    qc_config = config.get('qc', {})
    preproc_config = config.get('preprocessing', {})
    feat_config = config.get('embedding', {})
    hvg_config = config.get('hvg', {})
    evaluations_config = config.get('evaluations', [])
        
    return run_id, data_config, qc_config, preproc_config, hvg_config, feat_config, evaluations_config


def get_experiment_type(config_path: str) -> str:
    """
    Get the experiment type from a YAML config file.
    
    Args:
        config_path: Path to the YAML config file.
        
    Returns:
        Experiment type string. Options:
        - 'default': Standard single-method experiment (default)
        - 'synthetic_benchmark': Multi-method synthetic data benchmark
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    if config is None or not isinstance(config, dict):
        raise ValueError(
            f"Config file is empty or invalid: {config_path}. "
            "Cannot determine experiment_type."
        )
    config = _expand_env_in_config(config)
    return config.get("experiment_type", "default")


def get_embedding_key(feat_config: Dict[str, Any]) -> str:
    """
    Compute the expected embedding key from the feature config.
    
    Args:
        feat_config: Feature/embedding configuration dictionary.
    
    Returns:
        str: The embedding key (e.g., 'X_scimilarity' for single method,
             'X_concatenate_scconcept_scimilarity' for multiple methods with concatenate,
             'X_avg_scconcept_scimilarity' for multiple methods with average, etc.).
    """
    if 'methods' in feat_config and isinstance(feat_config['methods'], list):
        # Multiple methods: key depends on joining method
        joining_method = feat_config.get('embedding_joining_method', 'concatenate')
        method_names = []
        for method_config in feat_config['methods']:
            method_name = method_config['name']
            method_names.append(method_name.lower())
        return f'X_{joining_method}_' + '_'.join(method_names)
    else:
        # Single method
        method_name = feat_config['method']
        return f'X_{method_name.lower()}'
