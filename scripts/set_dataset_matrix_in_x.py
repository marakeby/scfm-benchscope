#!/usr/bin/env python3
"""
Set ``dataset.matrix_in_x`` on dataset YAML fragments under ``yaml/datasets/``.

Uses ``inspect_h5ad_metadata.infer_matrix_in_x`` (same effective ``X`` as ``H5ADLoader``).
Paths resolve under ``SCFM_DATA_PATH`` / ``setup_path.DATA_PATH``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import importlib.util

from scfm_cancer_eval.setup_path import DATA_PATH

_spec = importlib.util.spec_from_file_location(
    "inspect_h5ad_metadata", _REPO / "scripts" / "inspect_h5ad_metadata.py"
)
_im = importlib.util.module_from_spec(_spec)
assert _spec.loader
_spec.loader.exec_module(_im)
infer_matrix_in_x = _im.infer_matrix_in_x


def _iter_dataset_yaml_files() -> list[Path]:
    root = _REPO / "src" / "scfm_cancer_eval" / "yaml" / "datasets"
    out: list[Path] = []
    for p in sorted(root.rglob("*.yaml")):
        if ".ipynb_checkpoints" in p.parts:
            continue
        out.append(p)
    return out


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)


def _resolve_dataset_block(cfg: dict) -> dict | None:
    """Return inner ``dataset`` mapping if this file defines one with a ``path``."""
    ds = cfg.get("dataset")
    if isinstance(ds, dict) and ds.get("path"):
        return ds
    return None


def scan_dataset_matrix_in_x(*, write: bool = False) -> tuple[list[dict[str, Any]], int]:
    """
    Walk ``yaml/datasets/**/*.yaml``, infer ``matrix_in_x`` from H5AD (via ``infer_matrix_in_x``).

    If ``write`` is True, update each YAML file in place. Otherwise dry-run only.

    Returns ``(records, failure_count)``. Each record has ``kind`` in
    ``skip`` | ``ok`` | ``error`` and fields suitable for JSON / printing.
    """
    records: list[dict[str, Any]] = []
    failures = 0

    for ypath in _iter_dataset_yaml_files():
        rel_yaml = str(ypath.relative_to(_REPO))
        cfg = _load_yaml(ypath)
        ds = _resolve_dataset_block(cfg)
        if ds is None:
            records.append(
                {
                    "kind": "skip",
                    "yaml": rel_yaml,
                    "reason": "no dataset.path",
                }
            )
            continue

        rel = str(ds["path"]).strip()
        h5 = os.path.normpath(os.path.join(DATA_PATH, rel))
        load_raw = bool(ds.get("load_raw"))
        layer = str(ds.get("layer_name", "X") or "X")

        try:
            label, stats = infer_matrix_in_x(h5, load_raw=load_raw, layer_name=layer)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            records.append(
                {
                    "kind": "error",
                    "yaml": rel_yaml,
                    "h5ad": h5,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        old = ds.get("matrix_in_x")
        ds["matrix_in_x"] = label
        records.append(
            {
                "kind": "ok",
                "yaml": rel_yaml,
                "h5ad": h5,
                "matrix_in_x": label,
                "matrix_in_x_previous": old,
                "load_raw": load_raw,
                "layer_name": layer,
                "Xeff_min": stats["Xeff_min"],
                "Xeff_max": stats["Xeff_max"],
            }
        )
        if write:
            _save_yaml(ypath, cfg)

    return records, failures


def main() -> int:
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv
    rows, failures = scan_dataset_matrix_in_x(write=not dry_run)
    for row in rows:
        if row["kind"] == "skip":
            print(f"skip (no dataset.path): {row['yaml']}")
        elif row["kind"] == "error":
            print(f"ERROR {row['yaml']}: {row['h5ad']}: {row['error']}")
        else:
            print(
                f"{row['yaml']}: matrix_in_x={row['matrix_in_x']!r} (was {row['matrix_in_x_previous']!r}) "
                f"Xeff∈[{row['Xeff_min']:.4g},{row['Xeff_max']:.4g}] "
                f"load_raw={row['load_raw']} layer={row['layer_name']!r}"
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
