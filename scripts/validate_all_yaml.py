#!/usr/bin/env python3
"""
Validate YAML under ``yaml/`` for this repo.

1. **Parse** — ``yaml.safe_load`` every ``yaml/**/*.yaml``.
2. **Experiments** — merge each ``yaml/exp/**/*.yaml`` like ``run_exp`` / ``exp_yaml_merge``,
   then apply ``utils.validate_exp_constraints.validate_merged_config``.
3. **Optional** — ``--matrix-in-x-dry-run`` opens H5ADs and checks ``infer_matrix_in_x`` for
   dataset fragments (same as ``set_dataset_matrix_in_x.py --dry-run``; can take minutes).

Exit code ``1`` if any parse error, merge failure, constraint **error**, matrix scan failure,
or (with ``--warnings-as-errors``) any constraint warning.

Examples::

    python scripts/validate_all_yaml.py
    python scripts/validate_all_yaml.py --format json --matrix-in-x-dry-run
    python scripts/validate_all_yaml.py --constraints yaml/constraints/embedding_data_requirements.yaml
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]


def _load_scan_module():
    spec = importlib.util.spec_from_file_location(
        "set_dataset_matrix_in_x",
        REPO / "scripts" / "set_dataset_matrix_in_x.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def _parse_all_yaml(yaml_root: Path) -> tuple[int, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    n = 0
    for p in sorted(yaml_root.rglob("*.yaml")):
        if ".ipynb_checkpoints" in p.parts:
            continue
        n += 1
        rel = str(p.relative_to(REPO))
        try:
            yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": rel, "error": f"{type(exc).__name__}: {exc}"})
    return n, errors


def _validate_experiments(
    yaml_root: Path,
    *,
    constraints_path: str,
) -> tuple[
    int,
    list[dict[str, str]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    from scfm_cancer_eval.utils.validate_exp_constraints import validate_config_file

    merge_errors: list[dict[str, str]] = []
    constraint_errors: list[dict[str, Any]] = []
    constraint_warnings: list[dict[str, Any]] = []
    n = 0

    for p in sorted((yaml_root / "exp").rglob("*.yaml")):
        if ".ipynb_checkpoints" in p.parts:
            continue
        n += 1
        rel = str(p.relative_to(yaml_root))
        try:
            errs, warns = validate_config_file(rel, constraints_path=constraints_path)
        except Exception as exc:  # noqa: BLE001
            merge_errors.append({"path": rel, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if errs:
            constraint_errors.append({"path": rel, "errors": errs})
        if warns:
            constraint_warnings.append({"path": rel, "warnings": warns})

    return n, merge_errors, constraint_errors, constraint_warnings


def _print_text_report(data: dict[str, Any]) -> None:
    yp = data["yaml_parse"]
    print("=== YAML parse ===")
    print(f"files checked: {yp['files_checked']}")
    print(f"parse failures: {len(yp['errors'])}")
    for e in yp["errors"][:80]:
        print(f"  {e['path']}: {e['error']}")
    if len(yp["errors"]) > 80:
        print(f"  ... and {len(yp['errors']) - 80} more")

    mx = data.get("matrix_in_x")
    if mx is not None:
        print()
        print("=== matrix_in_x H5AD dry-run ===")
        print(f"records: {len(mx['records'])}  failures: {mx['failures']}")
        for row in mx["records"]:
            if row["kind"] == "skip":
                print(f"  skip: {row['yaml']} ({row.get('reason', '')})")
            elif row["kind"] == "error":
                print(f"  ERROR {row['yaml']}: {row.get('error', '')}")
            else:
                print(
                    f"  ok {row['yaml']}: matrix_in_x={row['matrix_in_x']!r} "
                    f"Xeff∈[{row['Xeff_min']:.4g},{row['Xeff_max']:.4g}]"
                )

    ex = data["experiments"]
    print()
    print("=== Experiments (merge + constraints) ===")
    print(f"exp configs: {ex['files_checked']}")
    print(f"merge/load failures: {len(ex['merge_errors'])}")
    for e in ex["merge_errors"][:40]:
        print(f"  {e['path']}: {e['error']}")
    if len(ex["merge_errors"]) > 40:
        print(f"  ... and {len(ex['merge_errors']) - 40} more")

    print(f"constraint errors: {len(ex['constraint_errors'])}")
    for block in ex["constraint_errors"][:40]:
        for msg in block["errors"]:
            print(f"  ERROR {block['path']}: {msg}")
    if len(ex["constraint_errors"]) > 40:
        print(f"  ... and {len(ex['constraint_errors']) - 40} more files with errors")

    n_warn_files = len(ex["constraint_warnings"])
    n_warn_lines = sum(len(b["warnings"]) for b in ex["constraint_warnings"])
    print(f"configs with warnings: {n_warn_files} (total warning lines: {n_warn_lines})")
    for block in ex["constraint_warnings"][:20]:
        print(f"  WARNING {block['path']}: {block['warnings'][0]}")
        if len(block["warnings"]) > 1:
            print(f"    ... +{len(block['warnings']) - 1} more")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=REPO,
        help="Repo root (default: parent of scripts/).",
    )
    parser.add_argument(
        "--yaml-dir",
        type=Path,
        default=None,
        help="YAML root relative to repo (default: src/scfm_cancer_eval/yaml).",
    )
    parser.add_argument(
        "--constraints",
        type=str,
        default=str(
            REPO / "src" / "scfm_cancer_eval" / "yaml" / "constraints" / "embedding_data_requirements.yaml"
        ),
        help="Path to embedding_data_requirements.yaml (absolute or under repo).",
    )
    parser.add_argument(
        "--matrix-in-x-dry-run",
        action="store_true",
        help="Scan H5ADs for yaml/datasets (slow); does not write YAML.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--warnings-as-errors",
        action="store_true",
        help="Treat constraint warnings as failures (exit 1).",
    )
    parser.add_argument(
        "--skip-parse",
        action="store_true",
        help="Skip yaml.safe_load pass.",
    )
    parser.add_argument(
        "--skip-exp",
        action="store_true",
        help="Skip experiment merge + constraint validation.",
    )
    args = parser.parse_args(argv)

    repo: Path = args.repo.resolve()
    yaml_root = (
        repo / (args.yaml_dir or Path("src") / "scfm_cancer_eval" / "yaml")
    ).resolve()
    if not yaml_root.is_dir():
        print(f"ERROR: yaml dir not found: {yaml_root}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(repo / "src"))

    constraints_path = args.constraints
    if not Path(constraints_path).is_absolute():
        constraints_path = str((repo / constraints_path).resolve())

    report: dict[str, Any] = {"repo": str(repo), "yaml_root": str(yaml_root)}

    if args.skip_parse:
        report["yaml_parse"] = {"files_checked": 0, "errors": [], "skipped": True}
    else:
        n, err = _parse_all_yaml(yaml_root)
        report["yaml_parse"] = {"files_checked": n, "errors": err}

    if args.matrix_in_x_dry_run:
        scan_mod = _load_scan_module()
        rows, fail = scan_mod.scan_dataset_matrix_in_x(write=False)
        report["matrix_in_x"] = {"records": rows, "failures": fail}
    else:
        report["matrix_in_x"] = None

    if args.skip_exp:
        report["experiments"] = {
            "files_checked": 0,
            "merge_errors": [],
            "constraint_errors": [],
            "constraint_warnings": [],
            "skipped": True,
        }
    else:
        n, me, ce, cw = _validate_experiments(yaml_root, constraints_path=constraints_path)
        report["experiments"] = {
            "files_checked": n,
            "merge_errors": me,
            "constraint_errors": ce,
            "constraint_warnings": cw,
        }

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        _print_text_report(report)

    yp = report["yaml_parse"]
    mx = report["matrix_in_x"] or {}
    ex = report["experiments"]

    fail = False
    if not args.skip_parse and yp.get("errors"):
        fail = True
    if args.matrix_in_x_dry_run and mx.get("failures", 0) > 0:
        fail = True
    if not args.skip_exp:
        if ex.get("merge_errors"):
            fail = True
        if ex.get("constraint_errors"):
            fail = True
        if args.warnings_as_errors and ex.get("constraint_warnings"):
            fail = True

    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
