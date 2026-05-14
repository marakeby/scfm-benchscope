import os
from os.path import join, realpath, dirname, normpath
from typing import AbstractSet

BASE_PATH = dirname(realpath(__file__))
# Experiment YAML fragments: bundled under the package by default. Override to a custom root
# (e.g. your own yaml/ tree) with SCFM_PARAMS_PATH — see README "Custom models".
_params_override = (os.environ.get("SCFM_PARAMS_PATH") or "").strip()
PARAMS_PATH = normpath(_params_override) if _params_override else join(BASE_PATH, "yaml")
RUN_PATH = join(BASE_PATH, 'run')

# When unset, paths are resolved at import time relative to the process working directory
# (not the package install location). Set SCFM_DATA_PATH / SCFM_OUTPUT_PATH / SCFM_MODELS_PATH
# for fixed locations (e.g. shared mounts on a VM).
_cwd_default = os.getcwd()
DATA_PATH = os.environ.get("SCFM_DATA_PATH", join(_cwd_default, "data"))
# Global run ledger (metrics_runs.csv); override if outputs live on another mount.
OUTPUT_PATH = os.environ.get("SCFM_OUTPUT_PATH", join(_cwd_default, "output"))

# Weights / checkpoints live under here. YAML can use paths relative to MODELS_PATH (see resolve_under_models_path).
MODELS_PATH = os.environ.get("SCFM_MODELS_PATH", join(_cwd_default, "models"))


def resolve_under_models_path(value: str) -> str:
    """
    If ``value`` is a non-empty relative path, return ``normpath(MODELS_PATH / value)``.
    Absolute paths and empty strings are returned unchanged (after normpath for absolutes).
    """
    stripped = value.strip()
    if not stripped:
        return value
    if os.path.isabs(stripped):
        return normpath(stripped)
    return normpath(join(MODELS_PATH, stripped))


def resolve_models_path_params(
    params: dict,
    keys: AbstractSet[str],
    *,
    recurse: bool = True,
) -> None:
    """
    Mutate ``params`` in place: for each key in ``keys``, string values are passed through
    :func:`resolve_under_models_path` (relative → under :data:`MODELS_PATH`).

    When ``recurse`` is True, nested dicts (e.g. ``continue_training``) are walked so the
    same key names can be resolved at any depth.
    """
    for k, v in list(params.items()):
        if recurse and isinstance(v, dict):
            resolve_models_path_params(v, keys, recurse=True)
        elif k in keys and isinstance(v, str):
            params[k] = resolve_under_models_path(v)
