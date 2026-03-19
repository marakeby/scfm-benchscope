#!/usr/bin/env python3
"""
One-shot setup: download datasets, clone model repos, and fetch checkpoints.

Reads dataset_paths.yaml and model_paths.yaml from the setup/ directory. Requires
DATASETS_PATH and MODEL_CHECKPOINTS in environment (e.g. from .env loaded by the caller).

Usage:
  python setup/setup_download_all.py
  python setup/setup_download_all.py --datasets-only
  python setup/setup_download_all.py --repos-only
  python setup/setup_download_all.py --checkpoints-only
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

SETUP_DIR = Path(__file__).resolve().parent
REPO_ROOT = SETUP_DIR.parent


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file; require PyYAML."""
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_url(url: str, dest: Path) -> None:
    """Download url to dest; create parent dirs."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "scFM_eval-setup/1.0"})
    with urllib.request.urlopen(req) as resp:
        dest.write_bytes(resp.read())


def do_datasets(datasets_path: Path, manifest: dict[str, Any]) -> None:
    """Download all datasets from manifest to datasets_path."""
    print(f"=== Downloading datasets to {datasets_path} ===")
    for item in manifest.get("datasets", []):
        url = item["url"]
        rel = item["path"]
        dest = datasets_path / rel
        name = item.get("name", item.get("id", rel))
        if dest.exists():
            print(f"  {name} already exists, skip")
            continue
        print(f"  {name}...")
        download_url(url, dest)
    print("  Datasets done.")


def do_repos(repos_dir: Path, manifest: dict[str, Any]) -> None:
    """Clone all repos from manifest into repos_dir."""
    print(f"=== Cloning model repos to {repos_dir} ===")
    repos_dir.mkdir(parents=True, exist_ok=True)
    for item in manifest.get("repos", []):
        name = item["name"]
        url = item["url"]
        clone_dir = repos_dir / name
        if (clone_dir / ".git").exists():
            print(f"  {name} already cloned, skip")
            continue
        print(f"  Cloning {name}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(clone_dir)],
            check=False,
            capture_output=True,
        )
        if not (clone_dir / ".git").exists():
            subprocess.run(["git", "clone", url, str(clone_dir)], check=True)
    print("  Repos done.")


def do_checkpoints(
    checkpoints_base: Path, manifest: dict[str, Any], have_hf_cli: bool
) -> None:
    """Download or print instructions for each checkpoint in manifest."""
    print(f"=== Checkpoints under {checkpoints_base} ===")
    checkpoints_base.mkdir(parents=True, exist_ok=True)

    for item in manifest.get("checkpoints", []):
        ckpt_type = item.get("type", "")
        name = item.get("name", "unknown")
        local_dir = checkpoints_base / item.get("local_dir", name)

        if ckpt_type == "huggingface":
            if local_dir.exists() and any(local_dir.iterdir()):
                print(f"  {name} already exists, skip")
            elif have_hf_cli:
                print(f"  {name} (huggingface)...")
                subprocess.run(
                    [
                        "huggingface-cli",
                        "download",
                        item["repo_id"],
                        "--local-dir",
                        str(local_dir),
                        "--local-dir-use-symlinks",
                        "False",
                    ],
                    check=True,
                )
            else:
                print(
                    f"  {name}: install huggingface_hub and run: "
                    f"huggingface-cli download {item['repo_id']} --local-dir {local_dir}"
                )

        elif ckpt_type == "huggingface_file":
            local_dir.mkdir(parents=True, exist_ok=True)
            target_file = local_dir / item["file"]
            if target_file.exists():
                print(f"  {name} already exists, skip")
                continue
            if have_hf_cli:
                print(f"  {name}...")
                subprocess.run(
                    [
                        "huggingface-cli",
                        "download",
                        item["repo_id"],
                        item["file"],
                        "--local-dir",
                        str(local_dir),
                    ],
                    check=True,
                )
            else:
                print(
                    f"  {name}: run: huggingface-cli download {item['repo_id']} {item['file']} --local-dir {local_dir}"
                )

        elif ckpt_type == "url":
            local_dir.mkdir(parents=True, exist_ok=True)
            url = item["url"]
            filename = url.split("/")[-1].split("?")[0] or "downloaded"
            dest_file = local_dir / filename
            if item.get("extract"):
                extracted = local_dir / filename.replace(".tar.gz", "").replace(".tgz", "")
                if dest_file.exists() or (extracted.exists() and extracted.is_dir()):
                    print(f"  {name} already exists, skip")
                    continue
            elif dest_file.exists():
                print(f"  {name} already exists, skip")
                continue
            print(f"  {name}...")
            download_url(url, dest_file)
            if item.get("extract") and dest_file.suffix in (".gz", ".tgz"):
                with tarfile.open(dest_file, "r:gz") as tf:
                    tf.extractall(local_dir)
                dest_file.unlink()

        elif ckpt_type in ("gdrive", "instructions_only") or item.get("instructions_only"):
            msg = item.get("message") or (
                f"Checkpoint on Google Drive: {item.get('url', 'see model_paths.yaml')}. "
                f"Install gdown and run: gdown --folder <url> -O {local_dir}"
            )
            if not (local_dir.exists() and any(local_dir.iterdir())):
                print(f"  {name}: {msg}")

    print("  Checkpoints done.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download datasets, clone repos, fetch checkpoints")
    parser.add_argument("--datasets-only", action="store_true", help="Only download datasets")
    parser.add_argument("--repos-only", action="store_true", help="Only clone repos")
    parser.add_argument("--checkpoints-only", action="store_true", help="Only fetch checkpoints")
    args = parser.parse_args()

    do_datasets_run = args.datasets_only or (
        not args.datasets_only and not args.repos_only and not args.checkpoints_only
    )
    do_repos_run = args.repos_only or (
        not args.datasets_only and not args.repos_only and not args.checkpoints_only
    )
    do_checkpoints_run = args.checkpoints_only or (
        not args.datasets_only and not args.repos_only and not args.checkpoints_only
    )

    datasets_path = os.environ.get("DATASETS_PATH")
    model_checkpoints = os.environ.get("MODEL_CHECKPOINTS")
    if not datasets_path and do_datasets_run:
        raise SystemExit("Set DATASETS_PATH in .env (root for downloaded .h5ad files)")
    if not model_checkpoints and (do_repos_run or do_checkpoints_run):
        raise SystemExit("Set MODEL_CHECKPOINTS in .env (root for model checkpoints)")

    datasets_path = Path(datasets_path) if datasets_path else None
    model_checkpoints = Path(model_checkpoints) if model_checkpoints else None
    repos_dir = (
        Path(os.environ.get("REPOS_DIR", str(model_checkpoints.parent / "repos")))
        if model_checkpoints
        else Path(os.environ.get("REPOS_DIR", "."))
    )

    dataset_manifest_path = SETUP_DIR / "dataset_paths.yaml"
    model_manifest_path = SETUP_DIR / "model_paths.yaml"

    if do_datasets_run and datasets_path and dataset_manifest_path.exists():
        do_datasets(datasets_path, load_yaml(dataset_manifest_path))

    if do_repos_run and model_manifest_path.exists() and model_checkpoints is not None:
        do_repos(repos_dir, load_yaml(model_manifest_path))

    if do_checkpoints_run and model_checkpoints and model_manifest_path.exists():
        have_hf = shutil.which("huggingface-cli") is not None
        do_checkpoints(model_checkpoints, load_yaml(model_manifest_path), have_hf)

    print("Setup finished. Set DATASETS_PATH and MODEL_CHECKPOINTS in .env and run your eval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
