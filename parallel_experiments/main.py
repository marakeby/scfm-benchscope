"""Main entry point for parallel experiment runner."""

## examples of how to run the script:
# python -m parallel_experiments.main -w 8
# python -m parallel_experiments.main --tasks drug_response -w 4 --gpus 0,1
# python -m parallel_experiments.main --tasks cancer_survival -w 4 --gpus 0,1   # TCGA survival (all models)
# python -m parallel_experiments.main -w 8 --force
# python -m parallel_experiments.main --tasks batch_bio_integration -w 4 --gpus 0,1
# python -m parallel_experiments.main --tasks batch_integration,biological_signal_preservation,label_transfer
# python -m parallel_experiments.main -m scconcept -d perturbseq_competition --force
# python -m parallel_experiments.main --dry-run
# gpu allocation: python -m parallel_experiments.main --gpus 0,1 -w 2
import argparse
import json
import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from parallel_experiments.config import (
    ALLOWED_MODELS,
    GPU_LOCK_FILE,
    LOG_FILE,
    STATUS_FILE,
)
from parallel_experiments.experiment_checker import (
    check_existing_results,
    scan_completed_experiments,
)
from parallel_experiments.experiment_runner import (
    run_experiment_wrapper,
    run_single_experiment,
)
from parallel_experiments.file_discovery import find_yaml_files
from parallel_experiments.gpu_manager import initialize_gpu_lock_file
from parallel_experiments.logging_config import setup_logging
from parallel_experiments.models import ExperimentResult
from parallel_experiments.status_tracker import initialize_status_file
from parallel_experiments.temp_management import cleanup_temp_files
import parallel_experiments.temp_management  # Ensure initialization runs

# Suppress pandas hashing warnings for NaN/inf values (harmless, handled gracefully by pandas)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in cast",
    module="pandas.core.util.hashing",
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Parallel experiment runner for scFM_eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Parallel experiment runner for scFM_eval.

Runs multiple YAML configs in parallel across N workers.
By default, skips experiments that already have results.

Examples:
    # Run all new experiments with 8 workers (skips existing results)
    python -m parallel_experiments.main -w 8

    # Bio + batch integration + label transfer only (all models and datasets for these tasks)
    python -m parallel_experiments.main --tasks drug_response -w 8

    # Force rerun all experiments (including existing results)
    python -m parallel_experiments.main -w 8 --force

    # Run only batch_integration task (skips existing; includes nested configs e.g. scconcept/HBio_HTech/*.yaml)
    python -m parallel_experiments.main -t batch_integration

    # Run only cancer_TME configs under batch_bio_integration (chan2021 pre/post + bassez_pre)
    python -m parallel_experiments.main --tasks batch_bio_integration --subgroup cancer_TME -w 8

    # Run multiple specific tasks (comma-separated)
    python -m parallel_experiments.main --tasks batch_integration,biological_signal_preservation,label_transfer

    # Run only scconcept method on a specific dataset (force rerun)
    python -m parallel_experiments.main -m scconcept -d perturbseq_competition --force

    # Dry run to see what would be executed
    python -m parallel_experiments.main --dry-run

    # TCGA survival (all methods: stack, scgpt, state, geneformer, scfoundation, scconcept, uce)
    python -m parallel_experiments.main --tasks cancer_survival -w 4 --gpus 0,1
        """,
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default=None,
        help="Run only specific task (e.g., batch_denoising). Deprecated: use --tasks for multiple tasks.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Run only specific tasks (comma-separated list, e.g., 'batch_denoising,cancer_chemo_identification')",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default=None,
        help="Run only specific method (e.g., scimilarity)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=None,
        help="Run only specific dataset (e.g., gtex_v9)",
    )
    parser.add_argument(
        "-s",
        "--subgroup",
        type=str,
        default=None,
        help="Run only configs under task/method/<subgroup>/*.yaml (e.g., cancer_TME)",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        help="Run only specific YAML file (relative to yaml/)",
    )
    parser.add_argument(
        "--force",
        "--rerun",
        action="store_true",
        dest="force",
        help="Force rerun even if results already exist (default: skip existing results)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="GPU devices to use (e.g., '1' for GPU 1 only, '0,1' for GPUs 0 and 1). Sets CUDA_VISIBLE_DEVICES environment variable.",
    )
    parser.add_argument(
        "--run-training-heavy-baselines",
        action="store_true",
        dest="run_training_heavy_baselines",
        help="Run training-heavy baseline methods (scvi, scanvi). By default, these are skipped. Can also be disabled in YAML config by setting skip_training_heavy_baselines: false.",
    )
    parser.add_argument(
        "--yaml-dir",
        type=str,
        default=None,
        help="Custom YAML config directory (default: <repo>/yaml). Useful for AB testing with generated configs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for results and embeddings (default: setup_path.OUTPUT_PATH). Useful for AB testing.",
    )
    parser.add_argument(
        "--run-viz-summary-per-experiment",
        action="store_true",
        dest="run_viz_summary_per_experiment",
        help="Run plotting and summarization after each experiment. Default: defer both until all experiments finish (faster parallel runs).",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    # Apply custom paths via env vars BEFORE any setup_path imports happen in subprocesses
    if args.yaml_dir:
        os.environ["SCFM_EVAL_PARAMS_PATH"] = str(Path(args.yaml_dir).resolve())
    if args.output_dir:
        os.environ["OUTPUT_PATH"] = str(Path(args.output_dir).resolve())

    # Setup logging
    logger = setup_logging()

    # Setup paths
    base_dir = Path(__file__).parent.parent.resolve()
    yaml_dir = Path(args.yaml_dir).resolve() if args.yaml_dir else base_dir / "yaml"

    # Parse task list if provided
    task_list = None
    if args.tasks:
        task_list = [t.strip() for t in args.tasks.split(",")]
        logger.info(f"Filtering to tasks: {task_list}")
    if args.subgroup:
        logger.info(f"Filtering to subgroup: {args.subgroup}")
    elif args.task:
        # Backward compatibility: single task becomes a list
        task_list = [args.task]
        logger.info(f"Filtering to task: {args.task}")

    # Find YAML files
    if args.file:
        yaml_path = yaml_dir / args.file
        if not yaml_path.exists():
            logger.error(f"File not found: {yaml_path}")
            return 1
        yaml_files = [yaml_path]
    else:
        yaml_files = find_yaml_files(
            yaml_dir,
            task_filter=args.task,  # Keep for backward compatibility
            task_list=task_list,
            method_filter=args.method,
            dataset_filter=args.dataset,
            subgroup_filter=args.subgroup,
        )

    if not yaml_files:
        logger.error("No YAML files found matching the criteria!")
        return 1

    # Check which experiments already have results (before filtering)
    skip_existing = not args.force  # Default: skip existing (unless --force is used)

    # Fast scan: build set of completed experiments in one pass
    completed_set = scan_completed_experiments()

    # Categorize experiments using O(1) lookups
    existing_results = []
    new_experiments = []

    for yaml_path in yaml_files:
        if check_existing_results(yaml_path, yaml_dir, completed_set=completed_set):
            existing_results.append(yaml_path)
        else:
            new_experiments.append(yaml_path)

    # Clear status file at start of new session
    initialize_status_file()

    # Log start of new session
    logger.info("=" * 60)
    logger.info("NEW RUN SESSION STARTED")
    logger.info(f"Found {len(yaml_files)} YAML config files")
    logger.info(
        f"  - {len(existing_results)} experiments with full results (main + all baselines have expected n_metrics)"
    )
    logger.info(
        f"  - {len(new_experiments)} experiments to run (main or ≥1 baseline missing 11 metrics, or 2 for technical_repeats)"
    )
    if args.dry_run:
        logger.info("DRY RUN MODE - No experiments will be executed")
    else:
        logger.info(f"Using {args.workers} parallel workers")
    if skip_existing:
        logger.info(
            "Will SKIP experiments that already have results (use --force to rerun)"
        )
        logger.info(f"Will run {len(new_experiments)} new experiments")
    else:
        logger.info("Will FORCE RERUN all experiments (including existing results)")
        logger.info(f"Will run all {len(yaml_files)} experiments")
    # Parse GPU list
    gpu_list = None
    if args.gpus:
        try:
            gpu_list = [int(g.strip()) for g in args.gpus.split(",")]
            logger.info(
                f"Using GPUs: {gpu_list} (max 2 workers per GPU, memory check for STATE/scGPT)"
            )
        except ValueError:
            logger.error(
                f"Invalid GPU list format: {args.gpus}. Use comma-separated integers (e.g., '0,1')"
            )
            return 1
    else:
        logger.info("Using all available GPUs (CUDA_VISIBLE_DEVICES not set)")

    # Initialize GPU lock file
    if gpu_list:
        initialize_gpu_lock_file(gpu_list)
        logger.info(
            f"GPU management: max 2 workers per GPU, 10GB minimum for STATE/scGPT"
        )

    logger.info("=" * 60)

    # Filter out existing results if skip_existing is True
    if skip_existing:
        yaml_files = new_experiments
        if not yaml_files:
            logger.info("All experiments already have results. Nothing to run!")
            logger.info("Use --force to rerun all experiments.")
            return 0

    # Set environment variable for skip_training_heavy_baselines
    # Default: skip training-heavy baselines (set to '1')
    # If --run-training-heavy-baselines is used, set to '0' to run them
    if getattr(args, 'run_training_heavy_baselines', False):
        os.environ["SKIP_TRAINING_HEAVY_BASELINES"] = "0"
        logger.info("Running training-heavy baselines (scvi, scanvi) for all experiments")
    else:
        # Default behavior: skip training-heavy baselines
        os.environ["SKIP_TRAINING_HEAVY_BASELINES"] = "1"
        logger.info("Skipping training-heavy baselines (scvi, scanvi) for all experiments [DEFAULT]")

    # Defer plotting and summarization by default (run once at end)
    defer_viz_summary = not getattr(args, "run_viz_summary_per_experiment", False)
    if defer_viz_summary:
        logger.info(
            "Deferring plotting and summarization until all experiments finish (use --run-viz-summary-per-experiment to run per experiment)"
        )

    # Prepare arguments for parallel execution
    experiment_args = [
        (yaml_path, base_dir, yaml_dir, skip_existing, args.dry_run, gpu_list, defer_viz_summary)
        for yaml_path in yaml_files
    ]

    # ------------------------------------------------------------------
    # Run experiments in parallel, retrying GPU-skipped ones up to
    # MAX_GPU_RETRY_ROUNDS times (other experiments may free GPUs).
    # ------------------------------------------------------------------
    MAX_GPU_RETRY_ROUNDS = 3
    GPU_RETRY_WAIT_SECONDS = 120  # Wait between retry rounds

    results: list[ExperimentResult] = []
    pending_args = experiment_args  # Experiments still to run

    for round_idx in range(1 + MAX_GPU_RETRY_ROUNDS):
        if not pending_args:
            break

        round_label = "initial" if round_idx == 0 else f"retry #{round_idx}"
        if round_idx > 0:
            logger.info(
                f"GPU retry round {round_idx}/{MAX_GPU_RETRY_ROUNDS}: "
                f"retrying {len(pending_args)} experiment(s) that couldn't acquire a GPU"
            )

        round_results: list[ExperimentResult] = []

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_experiment_wrapper, exp_args): exp_args
                for exp_args in pending_args
            }

            for future in as_completed(futures):
                exp_args = futures[future]
                yaml_path = exp_args[0]
                rel_path = yaml_path.relative_to(yaml_dir)

                try:
                    result = future.result()
                    round_results.append(result)

                    if result.dry_run:
                        logger.info(f"[DRY RUN] Would run: {rel_path}")
                    elif result.skipped_species_mismatch:
                        logger.info(f"⏭ Skipped (species mismatch): {rel_path}")
                    elif result.skipped_no_gpu:
                        logger.warning(
                            f"⏳ Skipped (no GPU available): {rel_path} — will retry"
                        )
                    elif result.skipped:
                        logger.info(f"⏭ Skipped (exists): {rel_path}")
                    elif result.success:
                        logger.info(
                            f"✓ Success: {rel_path} ({result.duration_seconds:.1f}s)"
                        )
                    else:
                        logger.error(f"✗ Failed: {rel_path} - {result.error_message}")

                except Exception as e:
                    logger.error(f"✗ Exception for {rel_path}: {e}")
                    round_results.append(
                        ExperimentResult(
                            yaml_path=yaml_path,
                            success=False,
                            error_message=str(e),
                        )
                    )

        # Separate GPU-skipped from final results
        gpu_skipped = [r for r in round_results if r.skipped_no_gpu]
        non_gpu_skipped = [r for r in round_results if not r.skipped_no_gpu]
        results.extend(non_gpu_skipped)

        if not gpu_skipped:
            break  # All experiments ran (or failed for other reasons)

        if round_idx < MAX_GPU_RETRY_ROUNDS:
            # Rebuild pending args for GPU-skipped experiments
            gpu_skipped_paths = {r.yaml_path for r in gpu_skipped}
            pending_args = [
                ea for ea in pending_args if ea[0] in gpu_skipped_paths
            ]
            logger.info(
                f"Waiting {GPU_RETRY_WAIT_SECONDS}s before GPU retry round "
                f"{round_idx + 1} ({len(pending_args)} experiment(s))..."
            )
            time.sleep(GPU_RETRY_WAIT_SECONDS)
        else:
            # Exhausted retries — record these as final failures
            results.extend(gpu_skipped)
            logger.warning(
                f"{len(gpu_skipped)} experiment(s) could not acquire a GPU after "
                f"{MAX_GPU_RETRY_ROUNDS} retry rounds"
            )

    # Summary
    dry_run_count = sum(1 for r in results if r.dry_run)
    success_count = sum(
        1 for r in results if r.success and not r.skipped and not r.dry_run
    )
    failed_count = sum(
        1
        for r in results
        if not r.success and not r.skipped_species_mismatch and not r.skipped_no_gpu
    )
    skipped_exists_count = sum(
        1
        for r in results
        if r.skipped
        and not r.skipped_species_mismatch
        and not r.skipped_no_gpu
    )
    skipped_species_count = sum(1 for r in results if r.skipped_species_mismatch)
    skipped_gpu_count = sum(1 for r in results if r.skipped_no_gpu)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total YAML files found: {len(existing_results) + len(new_experiments)}")
    print(f"  - Already have results: {len(existing_results)}")
    print(f"  - New experiments: {len(new_experiments)}")
    print(f"\nExperiments processed: {len(results)}")

    if args.dry_run:
        print(f"Would run: {dry_run_count}")
        print(f"Would skip (existing): {skipped_exists_count}")
    else:
        print(f"Success: {success_count}")
        print(f"Failed:  {failed_count}")
        if skipped_exists_count > 0:
            print(f"Skipped (existing): {skipped_exists_count}")
        if skipped_species_count > 0:
            print(f"Skipped (species mismatch): {skipped_species_count}")
        if skipped_gpu_count > 0:
            print(f"Skipped (no GPU after retries): {skipped_gpu_count}")

    if failed_count > 0:
        print("\nFailed experiments:")
        for result in results:
            if not result.success and not result.skipped_species_mismatch and not result.skipped_no_gpu:
                rel_path = result.yaml_path.relative_to(yaml_dir)
                print(f"  - {rel_path}: {result.error_message}")
        return_code = 1
    elif skipped_gpu_count > 0:
        print("\nGPU-skipped experiments (rerun to retry):")
        for result in results:
            if result.skipped_no_gpu:
                rel_path = result.yaml_path.relative_to(yaml_dir)
                print(f"  - {rel_path}")
        return_code = 1
    else:
        return_code = 0

    # If we deferred viz/summary, run summarization once at the end (no per-experiment plotting)
    if (
        not args.dry_run
        and defer_viz_summary
        and (success_count > 0 or len(existing_results) > 0)
    ):
        logger.info("Running deferred summarization (all tasks, methods, datasets)...")
        try:
            import sys
            sys.path.insert(0, str(base_dir))
            from setup_path import OUTPUT_PATH
            from utils.results_summarizer import ResultsSummarizer
            summarizer = ResultsSummarizer(OUTPUT_PATH)
            summary_dir = OUTPUT_PATH / "summaries"
            summary_dir.mkdir(parents=True, exist_ok=True)
            summarizer.generate_all_summaries(save_dir=summary_dir)
            logger.info(f"Deferred summarization complete → {summary_dir}")
        except Exception as e:
            logger.warning(f"Deferred summarization failed: {e}. You can run it manually later.")

    # Clean up temp files before exiting
    cleanup_temp_files(base_dir)

    return return_code


if __name__ == "__main__":
    import sys

    sys.exit(main())
