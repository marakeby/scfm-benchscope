"""
Merge experiment YAML the same way as ``run_exp.get_configs`` (bases / dataset / model includes).

Used by ``validate_exp_constraints`` without importing the full experiment stack.
"""
from __future__ import annotations

import copy
import os
from typing import Any

import yaml

from scfm_cancer_eval.setup_path import PARAMS_PATH


def deep_merge_dicts(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge_dicts(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_base_path(base_path: str, cfg_path: str) -> str:
    if os.path.isabs(base_path):
        return base_path
    if base_path.startswith("."):
        return os.path.normpath(os.path.join(os.path.dirname(cfg_path), base_path))
    return os.path.normpath(os.path.join(PARAMS_PATH, base_path))


def load_merged_experiment_config(config_path: str) -> dict[str, Any]:
    """
    Load a fully merged experiment dict (same merge order as ``run_exp.get_configs``).

    Args:
        config_path: Absolute path, or path relative to ``PARAMS_PATH`` (``yaml/``).
    """
    if not os.path.isabs(config_path):
        config_path = os.path.normpath(os.path.join(PARAMS_PATH, config_path))

    def load_config_with_bases(cfg_path: str, visited: set[str] | None = None) -> dict[str, Any]:
        if visited is None:
            visited = set()
        cfg_path = os.path.abspath(cfg_path)
        if cfg_path in visited:
            raise ValueError(f"Cyclic YAML bases detected: {cfg_path}")
        visited.add(cfg_path)

        with open(cfg_path, "r", encoding="utf-8") as file:
            config: dict[str, Any] = yaml.safe_load(file) or {}

        dataset = config.get("dataset", None)
        model = config.get("model", None)
        classification = config.get("classification", None)
        datasets = config.pop("datasets", None)
        classifications = config.pop("classifications", None)
        models = config.pop("models", None)
        bases = config.pop("bases", None)
        if bases is None:
            bases = config.pop("defaults", None)

        def _as_list(x):
            if x is None:
                return []
            if isinstance(x, list):
                return x
            return [x]

        dataset_includes = None
        if isinstance(dataset, (str, list)):
            dataset_includes = config.pop("dataset", None)

        model_includes = None
        if isinstance(model, (str, list)):
            model_includes = config.pop("model", None)

        classification_includes = None
        if isinstance(classification, (str, list)):
            classification_includes = config.pop("classification", None)

        bases_list = (
            _as_list(dataset_includes)
            + _as_list(datasets)
            + _as_list(model_includes)
            + _as_list(models)
            + _as_list(classification_includes)
            + _as_list(classifications)
            + _as_list(bases)
        )

        merged: dict[str, Any] = {}
        for base in bases_list:
            base_cfg_path = resolve_base_path(str(base), cfg_path)
            merged = deep_merge_dicts(merged, load_config_with_bases(base_cfg_path, visited))
        merged = deep_merge_dicts(merged, config)
        visited.remove(cfg_path)
        return merged

    return load_config_with_bases(config_path, visited=set())
