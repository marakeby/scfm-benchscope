"""Public library API for evaluating installed model adapters."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Mapping

from scfm_cancer_eval.contracts import (
    EvaluationModelConfig,
    ModelAdapter,
    RunResult,
)
from scfm_cancer_eval.utils.exp_yaml_merge import (
    deep_merge_dicts,
    load_merged_experiment_config,
)

ConfigInput = str | PathLike[str] | Mapping[str, Any]


@dataclass(frozen=True)
class EvaluationOptions:
    """Runtime options that do not belong to the model or dataset."""

    seed: int = 42
    max_cells: int | None = None
    max_cells_stratify: str | None = None
    evaluate_embedding: bool = True
    visualize: bool = False


def _load_config(source: ConfigInput) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return copy.deepcopy(dict(source))
    return load_merged_experiment_config(str(source))


def _adapter_embedding_config(
    model: ModelAdapter | EvaluationModelConfig,
    options: EvaluationOptions,
) -> dict[str, Any]:
    if isinstance(model, EvaluationModelConfig):
        return model.to_embedding_config(
            evaluate_embedding=options.evaluate_embedding,
            visualize=options.visualize,
        )

    model_id = str(
        getattr(model, "method", None) or model.__class__.__name__
    )
    output_key = getattr(model, "output_key", None)
    if not output_key or not callable(getattr(model, "fit_transform", None)):
        raise TypeError("model must provide output_key and fit_transform(loader)")
    return {
        "method": model_id,
        "output_key": str(output_key),
        "viz": bool(options.visualize),
        "eval": bool(options.evaluate_embedding),
        "params": {},
    }


def _build_evaluation_config(
    *,
    model: ModelAdapter | EvaluationModelConfig | None,
    experiment: ConfigInput | None,
    dataset: ConfigInput | None,
    task: ConfigInput | str | None,
    options: EvaluationOptions,
) -> dict[str, Any]:
    if (experiment is None) == (dataset is None):
        raise ValueError("provide exactly one of experiment or dataset")

    if experiment is not None:
        config = _load_config(experiment)
    else:
        config = _load_config(dataset)  # type: ignore[arg-type]
        config.setdefault("qc", {"skip": True})
        config.setdefault("preprocessing", {"skip": True})

        if task in (None, "embedding"):
            config["classification"] = {"skip": True}
        elif isinstance(task, (str, PathLike, Mapping)):
            config = deep_merge_dicts(config, _load_config(task))
        else:
            raise TypeError("task must be 'embedding', a config path, or a mapping")

    if model is not None:
        config["embedding"] = _adapter_embedding_config(model, options)
    elif "embedding" not in config:
        raise ValueError("a model is required when the config has no embedding section")

    config.setdefault("classification", {"skip": True})
    if not config.get("run_id"):
        method = config["embedding"].get("method", "model")
        config["run_id"] = f"{method}_evaluation"
    return config


def _runner_components():
    # Keep importing the public package lightweight; the runner imports the
    # scientific stack (scanpy, torch, scIB) only when an evaluation starts.
    from scfm_cancer_eval.run.run_exp import Experiment, set_random_seed

    return Experiment, set_random_seed


def evaluate(
    *,
    model: ModelAdapter | EvaluationModelConfig | None = None,
    experiment: ConfigInput | None = None,
    dataset: ConfigInput | None = None,
    task: ConfigInput | str | None = "embedding",
    output_dir: str | PathLike[str] | None = None,
    options: EvaluationOptions | None = None,
) -> RunResult:
    """Run an evaluation using either an existing experiment or simple pieces.

    A direct ``ModelAdapter`` bypasses YAML model registration. An
    ``EvaluationModelConfig`` or an existing experiment can still use the
    established import-string behavior.
    """
    resolved_options = options or EvaluationOptions()
    config = _build_evaluation_config(
        model=model,
        experiment=experiment,
        dataset=dataset,
        task=task,
        options=resolved_options,
    )

    source = experiment if experiment is not None else dataset
    source_path = None if isinstance(source, Mapping) else str(source)
    direct_adapter = model if model is not None and not isinstance(
        model, EvaluationModelConfig
    ) else None

    Experiment, set_random_seed = _runner_components()
    set_random_seed(resolved_options.seed)
    run = Experiment(
        source_path,
        resolved_config=config,
        model_adapter=direct_adapter,
        output_dir=output_dir,
        seed=resolved_options.seed,
        max_cells=resolved_options.max_cells,
        max_cells_stratify=resolved_options.max_cells_stratify,
    )
    run.run()
    run._write_standard_reports()

    output_path = Path(run.save_dir)
    return RunResult.from_path(
        output_path / "results.json",
        expected_run_id=run.run_id,
    )

