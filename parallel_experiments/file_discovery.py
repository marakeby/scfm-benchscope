"""YAML file discovery and filtering."""

from pathlib import Path

from parallel_experiments.config import ALLOWED_MODELS


def find_yaml_files(
    yaml_dir: Path,
    task_filter: str | None = None,
    task_list: list[str] | None = None,
    method_filter: str | None = None,
    dataset_filter: str | None = None,
    subgroup_filter: str | None = None,
) -> list[Path]:
    """Find all YAML config files matching the filters.

    Args:
        yaml_dir: Base directory containing YAML files.
        task_filter: Only include files from this task (e.g., 'batch_denoising').
                    Deprecated: use task_list instead.
        task_list: Only include files from these tasks (e.g., ['batch_denoising', 'cancer_chemo_identification']).
        method_filter: Only include files from this method (e.g., 'scimilarity').
        dataset_filter: Only include files with this dataset name (yaml stem).
        subgroup_filter: Only include files under task/method/<subgroup>/*.yaml (e.g. 'cancer_TME').

    Returns:
        List of paths to matching YAML files.
    """
    yaml_files = []

    # Find files in all model subdirectories (automatically discovers all models)
    # Iterates through all task directories and their method subdirectories
    for task_dir in yaml_dir.iterdir():
        if not task_dir.is_dir() or task_dir.name.startswith("!"):
            continue

        # Apply task filtering: either single task_filter or list of allowed tasks
        if task_list is not None:
            # Use task_list if provided (takes precedence)
            if task_dir.name not in task_list:
                continue
        elif task_filter:
            # Fall back to single task_filter for backward compatibility
            if task_dir.name != task_filter:
                continue

        # Look for method subdirectories
        for method_dir in task_dir.iterdir():
            if not method_dir.is_dir():
                continue

            method = method_dir.name

            # Only run allowed models (skip others). Also allow stack ablation variants (stack_*).
            if method not in ALLOWED_MODELS and not method.startswith("stack_"):
                continue

            # Apply method filter if specified (supports comma-separated list)
            if method_filter:
                allowed = {m.strip() for m in method_filter.split(",")}
                if method not in allowed:
                    continue

            # Find YAML files: direct (task/method/dataset.yaml) and nested (task/method/subgroup/dataset.yaml)
            for yaml_path in method_dir.rglob("*.yaml"):
                if not yaml_path.is_file():
                    continue

                # Apply subgroup filter: only keep files under method/<subgroup>/*.yaml
                if subgroup_filter:
                    try:
                        rel = yaml_path.relative_to(method_dir)
                    except ValueError:
                        continue
                    parts = rel.parts
                    if len(parts) < 2 or parts[0] != subgroup_filter:
                        continue

                dataset = yaml_path.stem

                # Apply dataset filter if specified
                if dataset_filter and dataset != dataset_filter:
                    continue

                yaml_files.append(yaml_path)

    return sorted(yaml_files)
