"""
Validate merged experiment YAML against ``yaml/constraints/embedding_data_requirements.yaml``.

Usage::

    python -m scfm_cancer_eval.utils.validate_exp_constraints exp/geneformer/V1-10M-i2048/brca_cell_type.yaml
    python -m scfm_cancer_eval.utils.validate_exp_constraints exp/scvi/default/brca_cell_type.yaml

Exit code 0 if no errors (warnings printed to stderr). Exit code 1 on any error.

Optional: set environment variable ``SCFM_VALIDATE_EXP=1`` so ``run_exp.Experiment`` validates
after loading config (same rules).

Dataset-specific behavior is driven by ``dataset.matrix_in_x`` in merged YAML (see
``yaml/constraints/embedding_data_requirements.yaml``). Optional future step: validate
against the actual ``.h5ad`` (e.g. heuristics on ``adata.X``) with a separate flag.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import yaml

from scfm_cancer_eval.setup_path import BASE_PATH, PARAMS_PATH
from scfm_cancer_eval.utils.exp_yaml_merge import load_merged_experiment_config

DEFAULT_CONSTRAINTS = os.path.join(
    BASE_PATH, "yaml", "constraints", "embedding_data_requirements.yaml"
)


def _truthy_skip(d: dict[str, Any] | None) -> bool:
    if not d:
        return False
    return d.get("skip") is True


def _matrix_in_x(dataset: dict[str, Any]) -> str:
    """Declared semantics of the matrix in ``adata.X`` after load (see constraints YAML)."""
    v = dataset.get("matrix_in_x") or dataset.get("matrix_in_X")
    if v is None:
        return "unspecified"
    s = str(v).strip().lower()
    allowed = {"raw_counts", "log_normalized", "other_processed", "unspecified"}
    if s not in allowed:
        return "unspecified"
    return s


def _load_constraints(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def validate_merged_config(
    merged: dict[str, Any],
    *,
    constraints_path: str = DEFAULT_CONSTRAINTS,
) -> tuple[list[str], list[str]]:
    """
    Returns:
        (errors, warnings) — string messages suitable for printing.
    """
    spec = _load_constraints(constraints_path)
    methods_spec: dict[str, Any] = (spec.get("methods") or {})
    defaults = spec.get("defaults") or {}

    errors: list[str] = []
    warnings: list[str] = []

    embedding = merged.get("embedding") or {}
    method = embedding.get("method")
    if not method:
        errors.append("merged config missing embedding.method")
        return errors, warnings

    rules = methods_spec.get(method)
    if not rules or not isinstance(rules, dict):
        sev = (defaults.get("unknown_method_severity") or "warning").lower()
        msg = defaults.get("message") or f"No constraints entry for method {method!r}."
        if sev == "error":
            errors.append(msg)
        else:
            warnings.append(msg)
        return errors, warnings

    hvg = merged.get("hvg")
    pre = merged.get("preprocessing") or {}
    dataset = merged.get("dataset") or {}
    params = embedding.get("params") or {}
    mix = _matrix_in_x(dataset)

    if rules.get("require_hvg_skip"):
        if hvg and not _truthy_skip(hvg):
            errors.append(
                f"[{method}] require_hvg_skip: set ``hvg: {{skip: true}}`` or omit ``hvg`` "
                f"(empty hvg is skipped by run_exp). Got hvg={hvg!r}."
            )

    if rules.get("require_preprocessing_skip"):
        if not _truthy_skip(pre):
            errors.append(
                f"[{method}] require_preprocessing_skip: set ``preprocessing: {{skip: true}}``. "
                f"Got preprocessing={pre!r}."
            )

    forbid_rule = rules.get("forbid_preprocess_normalize_and_log1p")
    if forbid_rule:
        default_map = (
            (spec.get("defaults") or {}).get("forbid_preprocess_normalize_and_log1p")
            or {
                "raw_counts": "error",
                "log_normalized": "warning",
                "other_processed": "warning",
                "unspecified": "error",
            }
        )
        if forbid_rule is True:
            severity_map = default_map
        elif isinstance(forbid_rule, dict) and "when_matrix_in_x" in forbid_rule:
            severity_map = {**default_map, **(forbid_rule.get("when_matrix_in_x") or {})}
        else:
            severity_map = default_map

        if not _truthy_skip(pre) and pre.get("normalize") and pre.get("apply_log1p"):
            sev = str(severity_map.get(mix, "error")).lower()
            msg = (
                f"[{method}] Active Scanpy ``normalize`` + ``log1p`` preprocessing "
                f"(matrix_in_x={mix!r}): risk of double transform or wrong scale for this embedding. "
                f"Set ``preprocessing.skip: true`` or disable ``normalize`` / ``apply_log1p``."
            )
            if sev == "warning":
                warnings.append(msg)
            else:
                errors.append(msg)

    if rules.get("warn_pre_normalized_T_raw_load"):
        when = rules.get("warn_pre_normalized_T_raw_load_when_matrix_in_x")
        if when is None:
            when = ["raw_counts", "unspecified"]
        pn = str(params.get("pre_normalized", "")).strip().upper()
        load_raw = bool(dataset.get("load_raw"))
        if pn == "T" and _truthy_skip(pre) and load_raw and mix in {str(x).strip().lower() for x in when}:
            warnings.append(
                f"[{method}] pre_normalized=T with preprocessing skipped and dataset load_raw=true: "
                f"``adata.X`` may be raw counts while ``T`` expects log1p+TP10k-normalized values. "
                f"Consider ``pre_normalized: F`` (see ``yaml/models/scfoundation/override_pre_normalized_F.yaml``) "
                f"or run preprocessing before scFoundation."
            )

    if rules.get("warn_if_matrix_not_raw_counts"):
        if method == "geneformer" and mix in ("log_normalized", "other_processed"):
            warnings.append(
                f"[{method}] matrix_in_x={mix!r}: Geneformer is usually run from raw UMI counts in ``X``; "
                f"confirm your ``adata`` layout and ``load_raw`` / ``layer_name`` settings match intent."
            )

    if rules.get("error_if_matrix_in_x_not_raw_counts"):
        if method == "scVI" and mix in ("log_normalized", "other_processed"):
            errors.append(
                f"[{method}] matrix_in_x={mix!r}: scVI expects count-like data in the layer used for training; "
                f"set ``dataset.matrix_in_x: raw_counts`` if counts live in ``X`` (or point ``embedding.params.layer_name`` "
                f"at a counts layer and document it)."
            )

    return errors, warnings


def validate_config_file(
    exp_yaml_relative: str,
    *,
    constraints_path: str = DEFAULT_CONSTRAINTS,
) -> tuple[list[str], list[str]]:
    """Resolve path under ``PARAMS_PATH`` (``yaml/``). Strips a leading ``yaml/`` if present."""
    rel = exp_yaml_relative.strip()
    if not os.path.isabs(rel) and rel.startswith("yaml/"):
        rel = rel[len("yaml/") :]
    merged = load_merged_experiment_config(rel)
    return validate_merged_config(merged, constraints_path=constraints_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configs",
        nargs="+",
        help="Experiment YAML path(s) relative to the yaml/ directory (e.g. exp/geneformer/.../x.yaml), "
        "optionally prefixed with yaml/. Absolute paths are accepted.",
    )
    parser.add_argument(
        "--constraints",
        default=DEFAULT_CONSTRAINTS,
        help="Path to constraints YAML (default: embedding_data_requirements.yaml).",
    )
    args = parser.parse_args(argv)

    any_error = False
    for cfg in args.configs:
        errs, warns = validate_config_file(cfg, constraints_path=args.constraints)
        for w in warns:
            print(f"WARNING {cfg}: {w}", file=sys.stderr)
        for e in errs:
            print(f"ERROR {cfg}: {e}", file=sys.stderr)
        if errs:
            any_error = True
    return 1 if any_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
