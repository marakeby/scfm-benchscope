"""Small cartesian grid search over Geneformer finetune YAML params.

Writes one override experiment YAML per combo, runs each trial (in-process by
default), then aggregates ``milcv_metrics_mean.csv`` into a summary ranked by a
chosen metric (default: AUC).

Example::

    pixi run -e geneformer python -m scfm_cancer_eval.run.grid_search_finetune \\
        grids/gf_finetune_small.yaml

    # or via shell wrapper from repo root:
    bash run/grid_search_gf_finetune.sh
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from scfm_cancer_eval.setup_path import OUTPUT_PATH, PARAMS_PATH
from scfm_cancer_eval.utils.exp_yaml_merge import load_merged_experiment_config


_PARAM_KEYS = (
    "learning_rate",
    "epoch",
    "freeze_layers",
    "weight_decay",
    "warmup_ratio",
    "max_number_genes",
    "max_cells_per_bag",
    "max_cells_per_patient",
    "mil_chunk_size",
    "gradient_checkpointing",
    "use_amp",
)


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_file():
        return p.resolve()
    cand = Path(PARAMS_PATH) / path
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError(f"Config not found: {path} (also tried under {PARAMS_PATH})")


def _slug(value: Any) -> str:
    text = str(value)
    text = text.replace("e-0", "e-").replace("E-0", "e-")
    text = re.sub(r"[^A-Za-z0-9._-]+", "", text)
    return text


def _trial_tag(params: dict[str, Any]) -> str:
    parts = []
    for key in _PARAM_KEYS:
        if key in params:
            short = {
                "learning_rate": "lr",
                "epoch": "ep",
                "freeze_layers": "fr",
                "weight_decay": "wd",
                "warmup_ratio": "wu",
                "max_number_genes": "ng",
                "max_cells_per_bag": "bag",
                "max_cells_per_patient": "mcp",
                "mil_chunk_size": "chk",
                "gradient_checkpointing": "gc",
                "use_amp": "amp",
            }[key]
            parts.append(f"{short}{_slug(params[key])}")
    return "_".join(parts) if parts else "default"


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def build_trial_yaml(
    base_exp_path: Path,
    params: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    with open(base_exp_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}

    trial = dict(base)
    trial["run_id"] = run_id

    classification = dict(trial.get("classification") or {})
    clf_params = dict(classification.get("params") or {})
    clf_params.update(params)
    classification["params"] = clf_params
    trial["classification"] = classification
    return trial


def _find_metrics_csv(save_dir: Path) -> Path | None:
    cv_dir = save_dir / "cv"
    if not cv_dir.is_dir():
        return None
    for name in ("milcv_metrics_mean.csv", "cv_metrics_mean.csv", "avgcv_metrics_mean.csv"):
        cand = cv_dir / name
        if cand.is_file():
            return cand
    matches = sorted(cv_dir.glob("*cv_metrics_mean.csv"))
    return matches[0] if matches else None


def _metric_from_mean_csv(csv_path: Path, metric: str, model_col: str | None) -> float | None:
    df = pd.read_csv(csv_path, index_col=0)
    if metric not in df.index and "Metrics" in df.columns:
        df = df.set_index("Metrics")
    if metric not in df.index:
        return None
    row = df.loc[metric]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    if model_col and model_col in getattr(row, "index", []):
        val = row[model_col]
    else:
        numeric = pd.to_numeric(pd.Series(row), errors="coerce").dropna()
        if numeric.empty:
            return None
        val = numeric.iloc[0]
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def run_trial_inprocess(trial_yaml: Path, *, seed: int) -> tuple[int, Path | None, str | None]:
    from scfm_cancer_eval.run.run_exp import Experiment, set_random_seed

    set_random_seed(seed)
    experiment = None
    try:
        experiment = Experiment(str(trial_yaml), seed=seed)
        experiment.run()
        experiment._write_standard_reports()
        return 0, Path(experiment.save_dir), None
    except Exception as exc:
        save_dir = Path(experiment.save_dir) if experiment is not None else None
        return 1, save_dir, f"{type(exc).__name__}: {exc}"


def run_trial_subprocess(trial_yaml: Path, *, runner: list[str]) -> tuple[int, Path | None]:
    cmd = [*runner, str(trial_yaml)]
    print(f"[grid] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
    if proc.stderr:
        print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n", flush=True, file=sys.stderr)

    save_dir = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("save_dir:"):
            save_dir = Path(line.split(":", 1)[1].strip())
    return int(proc.returncode), save_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cartesian grid search over Geneformer finetune classification.params."
    )
    parser.add_argument(
        "grid_config",
        nargs="?",
        default="grids/gf_finetune_small.yaml",
        help="Grid YAML (path or relative to PARAMS_PATH). Default: grids/gf_finetune_small.yaml",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Directory for generated trial YAMLs and summary (default: output/grid_search/<name>).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write trial YAMLs and print planned trials without running training.",
    )
    parser.add_argument(
        "--runner",
        default=None,
        help=(
            "If set, launch each trial via this subprocess command "
            "(e.g. 'scfm-eval'). Default: run Experiment in-process."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("SCFM_EVAL_SEED", "42")),
        help="RNG seed forwarded to each trial (default: SCFM_EVAL_SEED or 42).",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Optional cap on number of trials (after expansion).",
    )
    args = parser.parse_args(argv)

    grid_path = _resolve_path(args.grid_config)
    with open(grid_path, "r", encoding="utf-8") as f:
        grid_cfg = yaml.safe_load(f) or {}

    base_exp = grid_cfg.get("base_exp")
    if not base_exp:
        raise SystemExit("grid YAML must define 'base_exp'")
    base_exp_path = _resolve_path(base_exp)

    grid = grid_cfg.get("grid") or {}
    unknown = set(grid) - set(_PARAM_KEYS)
    if unknown:
        raise SystemExit(f"Unsupported grid keys: {sorted(unknown)}; allowed: {_PARAM_KEYS}")

    metric = str(grid_cfg.get("metric", "AUC"))
    model_col = grid_cfg.get("model_col")
    trials = expand_grid(grid)
    if args.max_trials is not None:
        trials = trials[: max(0, args.max_trials)]

    workdir = Path(args.workdir) if args.workdir else Path(OUTPUT_PATH) / "grid_search" / grid_path.stem
    trials_dir = workdir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    base_merged = load_merged_experiment_config(str(base_exp_path))
    base_run_id = str(base_merged.get("run_id") or base_exp_path.stem)

    rows: list[dict[str, Any]] = []
    summary_path = workdir / "grid_summary.csv"
    best_path = workdir / "best_trial.json"
    status_path = workdir / "grid_status.json"

    def _flush_summary() -> None:
        summary = pd.DataFrame(rows)
        if metric in summary.columns and not summary.empty:
            summary = summary.sort_values(by=metric, ascending=False, na_position="last")
        summary.to_csv(summary_path, index=False)
        status = {
            "workdir": str(workdir),
            "output_path": str(OUTPUT_PATH),
            "n_trials_planned": len(trials),
            "n_trials_recorded": len(rows),
            "metric": metric,
            "summary_csv": str(summary_path),
        }
        if metric in summary.columns and summary[metric].notna().any():
            best_row = summary.loc[summary[metric].idxmax()]
            best = best_row.to_dict()
            with open(best_path, "w", encoding="utf-8") as f:
                json.dump(best, f, indent=2, default=str)
            status["best"] = best
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2, default=str)

    print(f"[grid] base_exp={base_exp_path}")
    print(f"[grid] n_trials={len(trials)} workdir={workdir}")
    print(f"[grid] OUTPUT_PATH={OUTPUT_PATH}")
    print(f"[grid] metric={metric}")
    print(f"[grid] summary will be written to {summary_path}", flush=True)

    for i, params in enumerate(trials, start=1):
        tag = _trial_tag(params)
        run_id = f"{base_run_id}_{tag}"
        trial_cfg = build_trial_yaml(base_exp_path, params, run_id=run_id)
        trial_path = trials_dir / f"{i:03d}_{tag}.yaml"
        with open(trial_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(trial_cfg, f, sort_keys=False, default_flow_style=False)

        row: dict[str, Any] = {
            "trial": i,
            "run_id": run_id,
            "trial_yaml": str(trial_path),
            "returncode": 0,
            **params,
            metric: None,
            "metrics_csv": None,
            "save_dir": None,
            "error": None,
        }

        print(f"[grid] trial {i}/{len(trials)} {tag}", flush=True)
        if args.dry_run:
            print(f"[grid] dry-run would train {trial_path}", flush=True)
            rows.append(row)
            _flush_summary()
            continue

        save_dir: Path | None = None
        rc = 1
        err: str | None = None
        try:
            if args.runner:
                rc, save_dir = run_trial_subprocess(
                    trial_path, runner=args.runner.split()
                )
            else:
                rc, save_dir, err = run_trial_inprocess(trial_path, seed=args.seed)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print(f"[grid] trial failed: {err}", flush=True)
            rc = 1
            if save_dir is None:
                matches = sorted(Path(OUTPUT_PATH).rglob(f"*_{run_id}"))
                if matches:
                    save_dir = matches[-1]

        if err:
            row["error"] = err
            print(f"[grid] trial error: {err}", flush=True)

        row["returncode"] = rc
        if save_dir is not None:
            row["save_dir"] = str(save_dir)
            # Persist pointer next to trial YAML for easy discovery mid-run.
            with open(trials_dir / f"{i:03d}_{tag}.save_dir", "w", encoding="utf-8") as f:
                f.write(str(save_dir) + "\n")
            metrics_csv = _find_metrics_csv(save_dir)
            if metrics_csv is not None:
                row["metrics_csv"] = str(metrics_csv)
                row[metric] = _metric_from_mean_csv(metrics_csv, metric, model_col)

        rows.append(row)
        _flush_summary()
        print(
            f"[grid] trial {i}/{len(trials)} rc={rc} {tag} {metric}={row.get(metric)} "
            f"save_dir={row.get('save_dir')}",
            flush=True,
        )

    _flush_summary()
    print(f"[grid] wrote {summary_path}")
    if best_path.is_file():
        print(f"[grid] wrote {best_path}")
    print(f"[grid] wrote {status_path}")

    if args.dry_run:
        return 0
    return 0 if all(r.get("returncode", 1) == 0 for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
