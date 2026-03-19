"""
Evaluation runner module.

Handles running different types of evaluations (batch effects, biological signal, classification).
"""
from pathlib import Path
from typing import Dict, List, Optional
from anndata import AnnData
from utils.logs_ import get_logger
from setup_path import OUTPUT_PATH

logger = get_logger()


class EvaluationRunner:
    """
    Handles running configured evaluations.
    """
    
    def __init__(
        self,
        adata: AnnData,
        embedding_key: str,
        data_config: Dict,
        save_dir: str,
        log,
        metrics_batch_effects_dir: str = None,
        metrics_biological_signal_dir: str = None,
        metrics_classification_dir: str = None,
        metrics_annotation_dir: str = None,
        plots_evaluations_dir: str = None,
        task_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_subgroup: Optional[str] = None,
        evaluations_config: Optional[List[Dict]] = None,
    ):
        """
        Initialize evaluation runner.
        
        Args:
            adata: AnnData object with embeddings
            embedding_key: Key in adata.obsm containing embeddings
            data_config: Dataset configuration (for default keys)
            save_dir: Base directory to save results (for backward compatibility)
            log: Logger instance
            metrics_batch_effects_dir: Directory for batch effects metrics
            metrics_biological_signal_dir: Directory for biological signal metrics
            metrics_classification_dir: Directory for classification metrics
            metrics_annotation_dir: Directory for annotation metrics
            plots_evaluations_dir: Directory for evaluation plots
            task_name: Name of the task (e.g., 'batch_denoising')
            dataset_name: Name of the dataset (e.g., 'Bone_Marrow_RNA_ATAC_Neurips_2021')
            dataset_subgroup: Optional subgroup (e.g., 'HBio_HTech') for hierarchical organization
            evaluations_config: Full list of evaluation configs (for running bio/annotation on integration baselines)
        """
        self.adata = adata
        self.embedding_key = embedding_key
        self.data_config = data_config
        self.evaluations_config = evaluations_config or []
        self.save_dir = save_dir  # Keep for backward compatibility
        self.log = log
        
        # Store task, dataset, and subgroup names for integration baseline directories
        self.task_name = task_name or data_config.get('task_name', 'batch_denoising')
        self.dataset_name = dataset_name or data_config.get('dataset_name', 'unknown_dataset')
        self.dataset_subgroup = dataset_subgroup  # e.g., 'HBio_HTech' for hierarchical structure
        
        # Use organized subdirectories if provided, otherwise fall back to save_dir
        self.metrics_batch_effects_dir = metrics_batch_effects_dir or save_dir
        self.metrics_biological_signal_dir = metrics_biological_signal_dir or save_dir
        self.metrics_classification_dir = metrics_classification_dir or save_dir
        self.metrics_annotation_dir = metrics_annotation_dir or save_dir
        self.plots_evaluations_dir = plots_evaluations_dir or save_dir
        
        # Base metrics directory (for topology, trajectory, etc.)
        self.metrics_dir = Path(save_dir) / 'metrics' if save_dir else None
    
    def _get_integration_output_dir(self, integration_method: str) -> Path:
        """
        Get the output directory for an integration baseline method.
        
        Integration baselines are stored as separate "methods" at the same level as FMs:
        - Flat: OUTPUT_PATH / task / integration_method / dataset /
        - Hierarchical: OUTPUT_PATH / task / integration_method / subgroup / dataset /
        
        This mirrors the FM output structure and allows:
        - One-time computation per dataset (cached across FM evaluations)
        - Automatic inclusion in summaries as separate methods
        - Proper subgroup comparisons in ResultsSummarizer
        
        Args:
            integration_method: Name of the integration method (e.g., 'scvi', 'harmony', 'pca_qc')
        
        Returns:
            Path to the integration method's output directory for this dataset
        """
        if self.dataset_subgroup:
            # Hierarchical: task / method / subgroup / dataset
            integration_dir = OUTPUT_PATH / self.task_name / integration_method / self.dataset_subgroup / self.dataset_name
        else:
            # Flat: task / method / dataset
            integration_dir = OUTPUT_PATH / self.task_name / integration_method / self.dataset_name
        return integration_dir
    
    def _get_integration_metrics_path(self, integration_method: str) -> Path:
        """
        Get the path to the metrics CSV file for an integration baseline method.
        
        Args:
            integration_method: Name of the integration method
        
        Returns:
            Path to the batch_effects_metrics.csv file
        """
        integration_dir = self._get_integration_output_dir(integration_method)
        return integration_dir / 'metrics' / 'batch_effects_metrics.csv'
    
    def _check_integration_already_run(self, integration_method: str) -> bool:
        """
        Check if an integration baseline has already been run for this dataset.
        
        Args:
            integration_method: Name of the integration method
        
        Returns:
            True if results already exist, False otherwise
        """
        metrics_path = self._get_integration_metrics_path(integration_method)
        return metrics_path.exists()
    
    def run_batch_effects_evaluation(self, eval_config: Dict) -> None:
        """Run batch effects evaluation."""
        import os
        from evaluation.batch_effects import BatchEffectsEvaluator
        from evaluation.baseline_embeddings import INTEGRATION_METHODS, TRAINING_HEAVY_METHODS
        
        params = eval_config.get('params', {})
        batch_key = params.get('batch_key', self.data_config.get('batch_key', 'batch'))
        label_key = params.get('label_key', self.data_config.get('label_key', 'label'))
        metric = params.get('metric', 'euclidean')
        run_baselines = params.get('run_baselines', False)
        baseline_methods = params.get('baseline_methods', [])
        run_integration_baselines = params.get('run_integration_baselines', True)  # Default True
        # Check YAML config first, then environment variable
        # Default: True (skip training-heavy baselines by default)
        skip_training_heavy_baselines = params.get('skip_training_heavy_baselines', None)
        if skip_training_heavy_baselines is None:
            # Check environment variable (defaults to '1' = skip by default)
            skip_training_heavy_baselines = os.environ.get('SKIP_TRAINING_HEAVY_BASELINES', '1') == '1'
        integration_methods = params.get('integration_methods', [])
        
        # Evaluate main embedding
        evaluator = BatchEffectsEvaluator(
            adata=self.adata,
            embedding_key=self.embedding_key,
            batch_key=batch_key,
            label_key=label_key,
            save_dir=self.metrics_batch_effects_dir,
            plots_dir=self.plots_evaluations_dir,
            metric=metric,
            **{k: v for k, v in params.items() if k not in [
                'batch_key', 'label_key', 'metric', 'run_baselines', 'baseline_methods',
                'run_integration_baselines', 'integration_methods', 'skip_training_heavy_baselines'
            ]}
        )
        results = evaluator.evaluate()
        self.log.info(f"Batch effects evaluation complete. Results: {results}")
        
        # Create batch effects visualizations (PRE and POST embeddings)
        try:
            evaluator.visualize_batch_effects()
        except Exception as e:
            self.log.warning(f"Error creating batch effects visualizations: {e}", exc_info=True)
        
        # Run simple baseline evaluations if requested
        if run_baselines:
            self._run_baseline_batch_effects_evaluations(
                eval_config, batch_key, label_key, metric, baseline_methods
            )
        
        # Run integration baseline evaluations (scVI, scANVI, BBKNN, Harmony, etc.)
        # These are the standard scib benchmarks - run once per dataset
        if run_integration_baselines:
            # Filter out training-heavy methods if requested
            if skip_training_heavy_baselines:
                if not integration_methods:
                    # If no specific methods requested, use all except training-heavy ones
                    integration_methods = [m for m in INTEGRATION_METHODS if m not in TRAINING_HEAVY_METHODS]
                    self.log.info(f'Skipping training-heavy baselines: {TRAINING_HEAVY_METHODS}')
                else:
                    # Filter the requested methods to exclude training-heavy ones
                    filtered_methods = [m for m in integration_methods if m not in TRAINING_HEAVY_METHODS]
                    skipped = [m for m in integration_methods if m in TRAINING_HEAVY_METHODS]
                    if skipped:
                        self.log.info(f'Skipping training-heavy baselines: {skipped}')
                    integration_methods = filtered_methods
            
            self._run_integration_baseline_evaluations(
                eval_config, batch_key, label_key, metric, integration_methods
            )
    
    def run_integration_baselines_only(self, eval_config: Dict) -> None:
        """
        Run only integration baseline evaluations (harmony, pca_qc, scanorama, bbknn, etc.)
        without evaluating a main embedding.
        
        Use this to fill missing baselines for a (task, subgroup, dataset) without
        re-running an embedding model. Load data and preprocess, then call this with
        the batch_effects evaluation config.
        """
        import os
        from evaluation.baseline_embeddings import INTEGRATION_METHODS, TRAINING_HEAVY_METHODS
        
        params = eval_config.get('params', {})
        batch_key = params.get('batch_key', self.data_config.get('batch_key', 'batch'))
        label_key = params.get('label_key', self.data_config.get('label_key', 'label'))
        metric = params.get('metric', 'euclidean')
        skip_training_heavy_baselines = params.get('skip_training_heavy_baselines', None)
        if skip_training_heavy_baselines is None:
            skip_training_heavy_baselines = os.environ.get('SKIP_TRAINING_HEAVY_BASELINES', '1') == '1'
        integration_methods = params.get('integration_methods', [])
        
        if not integration_methods:
            integration_methods = INTEGRATION_METHODS
        if skip_training_heavy_baselines:
            integration_methods = [m for m in integration_methods if m not in TRAINING_HEAVY_METHODS]
            self.log.info(f'Skipping training-heavy baselines: {TRAINING_HEAVY_METHODS}')
        
        self.log.info('Running integration baselines only (no main embedding evaluation).')
        self._run_integration_baseline_evaluations(
            eval_config, batch_key, label_key, metric, integration_methods
        )
    
    def _run_baseline_batch_effects_evaluations(
        self,
        eval_config: Dict,
        batch_key: str,
        label_key: str,
        metric: str,
        baseline_methods: list
    ) -> None:
        """
        Run batch effects evaluation on baseline embeddings.
        
        Baseline evaluations are dataset-specific (not method-specific), so we:
        1. Check if baseline results already exist in the dataset
        2. Only run evaluations for baselines that don't exist yet
        3. Save results to adata.uns['baseline_batch_effects']
        
        NOTE: Always subsamples large datasets for efficiency.
        """
        from evaluation.batch_effects import BatchEffectsEvaluator
        from evaluation.baseline_embeddings import create_baseline_embedding, BASELINE_METHODS
        from utils.sampling import sample_adata_for_batch_integration
        
        params = eval_config.get('params', {})
        subsample_size = params.get('subsample_size', 5000)
        
        # Default baseline methods if not specified
        if not baseline_methods:
            baseline_methods = BASELINE_METHODS
        
        # Check which baselines already exist in the dataset
        existing_baselines = set()
        if 'baseline_batch_effects' in self.adata.uns:
            existing_baselines = set(self.adata.uns['baseline_batch_effects'].keys())
        
        # Also check if embeddings exist (they might have been computed but not evaluated)
        existing_embeddings = set()
        for key in self.adata.obsm.keys():
            if key.startswith('X_baseline_'):
                existing_embeddings.add(key)
        
        self.log.info(f'Checking baseline evaluations. Existing: {existing_baselines}')
        
        # Filter out baselines that already have results
        baselines_to_run = []
        for baseline_method in baseline_methods:
            if baseline_method not in BASELINE_METHODS:
                self.log.warning(f"Unknown baseline method: {baseline_method}. Skipping.")
                continue
            
            baseline_key = f'X_baseline_{baseline_method}'
            
            # Skip if results already exist
            if baseline_key in existing_baselines:
                self.log.info(f'Baseline {baseline_method} already evaluated. Skipping.')
                continue
            
            baselines_to_run.append(baseline_method)
        
        if not baselines_to_run:
            self.log.info('All requested baseline evaluations already exist in dataset. Skipping.')
            return
        
        self.log.info(f'Running baseline evaluations for methods: {baselines_to_run}')
        
        # Subsample for baseline computation (stratified by batch AND cell type)
        if self.adata.n_obs > subsample_size:
            self.log.info(f'Subsampling {self.adata.n_obs} -> {subsample_size} cells for baseline evaluations')
            adata_subsample = sample_adata_for_batch_integration(
                self.adata,
                batch_key=batch_key,
                label_key=label_key,
                sample_size=subsample_size,
                random_state=42
            )
        else:
            adata_subsample = self.adata
        
        for baseline_method in baselines_to_run:
            baseline_key = f'X_baseline_{baseline_method}'
            
            try:
                # Create baseline embedding on subsampled data
                self.log.info(f'Creating baseline embedding: {baseline_method}')
                baseline_embedding = create_baseline_embedding(
                    adata=adata_subsample,
                    method=baseline_method,
                    batch_key=batch_key,
                    label_key=label_key,
                    use_rep='X',
                    n_comps=50,
                    random_seed=42
                )
                # Store in subsampled adata obsm
                adata_subsample.obsm[baseline_key] = baseline_embedding
                
                # Evaluate baseline on subsampled data
                self.log.info(f'Evaluating baseline: {baseline_method}')
                baseline_evaluator = BatchEffectsEvaluator(
                    adata=adata_subsample,
                    embedding_key=baseline_key,
                    batch_key=batch_key,
                    label_key=label_key,
                    save_dir=self.metrics_batch_effects_dir,
                    metric=metric,
                    auto_subsample=False,  # Already subsampled
                    **{k: v for k, v in params.items() if k not in [
                        'batch_key', 'label_key', 'metric', 'run_baselines', 
                        'baseline_methods', 'subsample_size', 'auto_subsample'
                    ]}
                )
                baseline_results = baseline_evaluator.evaluate()
                self.log.info(f"Baseline evaluation complete for {baseline_method}. Results: {baseline_results}")
                
            except Exception as e:
                self.log.error(f"Error evaluating baseline {baseline_method}: {e}", exc_info=True)
    
    def _run_integration_baseline_evaluations(
        self,
        eval_config: Dict,
        batch_key: str,
        label_key: str,
        metric: str,
        integration_methods: list
    ) -> None:
        """
        Run batch effects evaluation on integration baseline methods.
        
        These are the standard scib benchmarks (scVI, scANVI, BBKNN, Harmony, Scanorama).
        
        **NEW ARCHITECTURE**: Each integration method is stored as a separate "method" directory:
        OUTPUT_PATH / task / integration_method / dataset / metrics / batch_effects_metrics.csv
        
        This allows:
        1. One-time computation per dataset (truly cached across all FM evaluations)
        2. Automatic inclusion in task-level summaries as separate methods
        3. Easy comparison between FMs and integration baselines in the summarizer
        
        NOTE: Always subsamples large datasets for efficiency (stratified by batch + cell type).
        
        Example structure after running:
        OUTPUT_PATH/batch_denoising/
        ├── scConcept/{dataset}/metrics/...  # FM results
        ├── scvi/{dataset}/metrics/...       # Integration baseline (separate method)
        ├── scanvi/{dataset}/metrics/...     # Integration baseline (separate method)
        └── summaries/                        # Aggregates ALL methods
        """
        from evaluation.batch_effects import BatchEffectsEvaluator
        from evaluation.baseline_embeddings import (
            create_baseline_embedding, 
            INTEGRATION_METHODS
        )
        from utils.sampling import sample_adata_for_batch_integration
        
        params = eval_config.get('params', {})
        subsample_size = params.get('subsample_size', 5000)
        
        # Default integration methods if not specified
        if not integration_methods:
            integration_methods = INTEGRATION_METHODS
        
        self.log.info('=' * 60)
        self.log.info('INTEGRATION BASELINES (scib benchmarks)')
        self.log.info('Each integration method is stored as a SEPARATE METHOD directory')
        self.log.info(f'Location: {OUTPUT_PATH / self.task_name}/<integration_method>/{self.dataset_name}/')
        self.log.info('=' * 60)
        
        # For each integration method, figure out exactly which evals are missing
        integrations_with_work = []  # list of (method, needs_batch, needs_bio, needs_ann)
        for integration_method in integration_methods:
            if integration_method not in INTEGRATION_METHODS:
                self.log.warning(f"Unknown integration method: {integration_method}. Skipping.")
                continue
            
            integration_dir = self._get_integration_output_dir(integration_method)
            metrics_dir = integration_dir / 'metrics'
            needs_batch = not (metrics_dir / 'batch_effects_metrics.csv').exists()
            needs_bio = not (metrics_dir / 'biological_signal_metrics.csv').exists()
            needs_ann = not (metrics_dir / 'annotation_round_robin_metrics.csv').exists()
            
            if not needs_batch and not needs_bio and not needs_ann:
                self.log.info(f'✓ Integration {integration_method} fully complete at: {metrics_dir}')
                continue
            
            missing = [name for name, needed in [
                ('batch_effects', needs_batch),
                ('biological_signal', needs_bio),
                ('annotation', needs_ann),
            ] if needed]
            self.log.info(f'⚠ {integration_method}: missing {", ".join(missing)} — will run.')
            integrations_with_work.append((integration_method, needs_batch, needs_bio, needs_ann))
        
        if not integrations_with_work:
            self.log.info('All requested integration baselines already fully evaluated. Skipping.')
            return
        
        # Subsample ONCE for all integration methods (stratified by batch AND cell type)
        if self.adata.n_obs > subsample_size:
            self.log.info(f'Subsampling {self.adata.n_obs} -> {subsample_size} cells for integration baselines')
            self.log.info('  (stratified by batch AND cell type for representative sampling)')
            adata_subsample = sample_adata_for_batch_integration(
                self.adata,
                batch_key=batch_key,
                label_key=label_key,
                sample_size=subsample_size,
                random_state=42
            )
        else:
            adata_subsample = self.adata
        
        for integration_method, needs_batch, needs_bio, needs_ann in integrations_with_work:
            embedding_key = f'X_{integration_method}'
            
            try:
                # Create the integration method's output directories
                integration_dir = self._get_integration_output_dir(integration_method)
                metrics_dir = integration_dir / 'metrics'
                plots_dir = integration_dir / 'plots' / 'evaluations'
                logs_dir = integration_dir / 'logs'
                config_dir = integration_dir / 'config'
                
                for dir_path in [metrics_dir, plots_dir, logs_dir, config_dir]:
                    dir_path.mkdir(parents=True, exist_ok=True)
                
                self.log.info(f'Running {integration_method} integration...')
                self.log.info(f'  Output directory: {integration_dir}')
                self.log.info(f'  Using {adata_subsample.n_obs} cells (subsampled)')
                
                # Generate the integration embedding (needed for any eval)
                integration_embedding = create_baseline_embedding(
                    adata=adata_subsample,
                    method=integration_method,
                    batch_key=batch_key,
                    label_key=label_key,
                    use_rep='X',
                    n_comps=50,
                    random_seed=42
                )
                adata_subsample.obsm[embedding_key] = integration_embedding
                self.log.info(f'  {integration_method} embedding shape: {integration_embedding.shape}')
                
                # ── batch_effects ──
                if needs_batch:
                    self.log.info(f'  Running batch_effects for {integration_method}...')
                    integration_evaluator = BatchEffectsEvaluator(
                        adata=adata_subsample,
                        embedding_key=embedding_key,
                        batch_key=batch_key,
                        label_key=label_key,
                        save_dir=metrics_dir,
                        plots_dir=plots_dir,
                        metric=metric,
                        auto_subsample=False,
                        **{k: v for k, v in params.items() if k not in [
                            'batch_key', 'label_key', 'metric', 'run_baselines', 'baseline_methods',
                            'run_integration_baselines', 'integration_methods', 'subsample_size',
                            'auto_subsample', 'skip_training_heavy_baselines'
                        ]}
                    )
                    integration_results = integration_evaluator.evaluate()
                    try:
                        integration_evaluator.visualize_batch_effects()
                    except Exception as e:
                        self.log.warning(f"  Error creating visualizations for {integration_method}: {e}")
                    if 'integration_batch_effects' not in self.adata.uns:
                        self.adata.uns['integration_batch_effects'] = {}
                    self.adata.uns['integration_batch_effects'][embedding_key] = integration_results
                    self.log.info(f"  ✓ batch_effects saved to {metrics_dir / 'batch_effects_metrics.csv'}")
                else:
                    self.log.info(f'  ✓ batch_effects already exists, skipping.')
                
                # ── biological_signal ──
                if needs_bio:
                    bio_config = next((c for c in self.evaluations_config if c.get('type') == 'biological_signal'), None)
                    if bio_config:
                        bio_params = bio_config.get('params', {})
                        self.log.info(f'  Running biological_signal for {integration_method}...')
                        from evaluation.biological_signal import BiologicalSignalEvaluator
                        # Filter out params that are passed explicitly to avoid
                        # "got multiple values for argument" TypeError
                        _bio_exclude = {'label_key', 'metric', 'auto_subsample'}
                        bio_evaluator = BiologicalSignalEvaluator(
                            adata=adata_subsample,
                            embedding_key=embedding_key,
                            label_key=bio_params.get('label_key', label_key),
                            save_dir=metrics_dir,
                            metric=bio_params.get('metric', 'euclidean'),
                            auto_subsample=False,  # already subsampled above
                            **{k: v for k, v in bio_params.items() if k not in _bio_exclude},
                        )
                        bio_evaluator.evaluate()
                        self.log.info(f"  ✓ biological_signal saved to {metrics_dir}")
                    else:
                        self.log.warning(f'  No biological_signal config in evaluations — cannot run.')
                else:
                    self.log.info(f'  ✓ biological_signal already exists, skipping.')
                
                # ── annotation (label transfer) ──
                if needs_ann:
                    ann_config = next((c for c in self.evaluations_config if c.get('type') == 'annotation'), None)
                    if ann_config:
                        ann_params = ann_config.get('params', {})
                        self.log.info(f'  Running annotation for {integration_method}...')
                        from evaluation.annotation import AnnotationEvaluator
                        ann_evaluator = AnnotationEvaluator(
                            adata=adata_subsample,
                            embedding_key=embedding_key,
                            label_key=ann_params.get('label_key', label_key),
                            save_dir=metrics_dir,
                            k=ann_params.get('k', 15),
                            metric=ann_params.get('metric', 'cosine'),
                            method=ann_params.get('method', 'sklearn_knn'),
                            split_strategies=ann_params.get('split_strategies', ['round_robin']),
                            round_robin_batch_key=ann_params.get('round_robin_batch_key') or batch_key,
                            **{k: v for k, v in ann_params.items() if k not in [
                                'label_key', 'k', 'metric', 'method', 'split_strategies', 'round_robin_batch_key',
                            ]},
                        )
                        ann_evaluator.evaluate()
                        self.log.info(f"  ✓ annotation saved to {metrics_dir}")
                    else:
                        self.log.warning(f'  No annotation config in evaluations — cannot run.')
                else:
                    self.log.info(f'  ✓ annotation already exists, skipping.')
                
                self.log.info(f"✓ Integration {integration_method} complete.")
                
            except ImportError as e:
                self.log.warning(
                    f"Integration method {integration_method} requires additional packages: {e}. "
                    f"Install the required package to run this baseline."
                )
            except Exception as e:
                self.log.error(f"Error evaluating integration {integration_method}: {e}", exc_info=True)
        
        self.log.info('Integration baseline evaluations complete.')

    def run_biological_signal_evaluation(self, eval_config: Dict) -> None:
        """Run biological signal evaluation."""
        from evaluation.biological_signal import BiologicalSignalEvaluator
        
        params = eval_config.get('params', {})
        label_key = params.get('label_key', self.data_config.get('label_key', 'label'))
        metric = params.get('metric', 'euclidean')
        
        evaluator = BiologicalSignalEvaluator(
            adata=self.adata,
            embedding_key=self.embedding_key,
            label_key=label_key,
            save_dir=self.metrics_biological_signal_dir,
            metric=metric,
            **{k: v for k, v in params.items() if k not in ['label_key', 'metric']}
        )
        results = evaluator.evaluate()
        self.log.info(f"Biological signal evaluation complete. Results: {results}")
    
    def run_classification_evaluation(
        self, 
        eval_config: Dict, 
        load_class_func,
        loader
    ) -> None:
        """
        Run classification evaluation (train and evaluate classifier).
        
        Args:
            eval_config: Evaluation configuration dictionary
            load_class_func: Function to load classes dynamically (from Experiment.load_class)
            loader: Data loader object (required by ClassifierPipeline)
        """
        params = eval_config.get('params', {}).copy()  # Copy to avoid modifying original
        
        # Extract module, class, viz, eval from params
        module = params.pop('module', 'models.classify')
        class_name = params.pop('class', 'ClassifierPipeline')
        viz = params.pop('viz', True)
        eval_ = params.pop('eval', True)
        
        # Remaining params are classifier parameters
        clf_params = params.copy()
        clf_params['save_dir'] = self.metrics_classification_dir
        clf_params['plots_dir'] = self.plots_evaluations_dir  # Separate plots directory
        clf_params['embedding_col'] = self.embedding_key
        
        # Build classifier config
        clf_config = {
            'module': module,
            'class': class_name,
            'params': clf_params,
            'viz': viz,
            'eval': eval_,
        }
        
        # Train and evaluate classifier
        ClfClass = load_class_func(module, class_name)
        clf = ClfClass(clf_config)
        clf.train(loader)
        
        self.log.info(f"Classification evaluation complete for embedding: {self.embedding_key}")
    
    def run_annotation_evaluation(self, eval_config: Dict) -> None:
        """Run annotation evaluation (KNN-based label transfer)."""
        from evaluation.annotation import AnnotationEvaluator
        
        params = eval_config.get('params', {})
        label_key = params.get('label_key', self.data_config.get('label_key', 'cell_type'))
        k = params.get('k', 15)
        metric = params.get('metric', 'cosine')
        method = params.get('method', 'sklearn_knn')
        split_strategies = params.get('split_strategies', ['even'])
        random_state = params.get('random_state', 42)
        min_cells_per_type = params.get('min_cells_per_type', 2)
        # Default round-robin batch column to dataset's batch_key (same as in dataset config)
        round_robin_batch_key = params.get('round_robin_batch_key') or self.data_config.get('batch_key')

        evaluator = AnnotationEvaluator(
            adata=self.adata,
            embedding_key=self.embedding_key,
            label_key=label_key,
            save_dir=self.metrics_annotation_dir,
            k=k,
            metric=metric,
            method=method,
            split_strategies=split_strategies,
            random_state=random_state,
            min_cells_per_type=min_cells_per_type,
            round_robin_batch_key=round_robin_batch_key,
            **{k: v for k, v in params.items() if k not in [
                'label_key', 'k', 'metric', 'method', 'split_strategies',
                'random_state', 'min_cells_per_type', 'round_robin_batch_key'
            ]}
        )
        results = evaluator.evaluate()
        self.log.info(f"Annotation evaluation complete. Results summary: {list(results.keys())}")
    
    def run_topology_evaluation(self, eval_config: Dict) -> None:
        """Run topology evaluation (TDA metrics for synthetic datasets).
        
        Implements CONCORD's full benchmark suite:
        - Mantel correlation (embedding and geodesic vs true topology)
        - Cell distance correlation
        - Trustworthiness (local neighborhood preservation)
        - Betti curve stability (persistent homology)
        - Betti number accuracy
        - State dispersion correlation
        """
        from evaluation.topology import TopologyEvaluator
        
        params = eval_config.get('params', {})
        topology_type = params.get('topology_type', 'trajectory')
        label_key = params.get('label_key', self.data_config.get('label_key', 'time'))
        metric = params.get('metric', 'euclidean')
        n_neighbors = params.get('n_neighbors', 30)
        
        # CONCORD benchmark parameters
        groundtruth_key = params.get('groundtruth_key', None)
        # Try to get groundtruth_key from adata.uns if not specified
        if groundtruth_key is None and 'groundtruth_key' in self.adata.uns:
            groundtruth_key = self.adata.uns['groundtruth_key']
            self.log.info(f"Using groundtruth_key from adata.uns: {groundtruth_key}")
        
        # Expected Betti numbers (try to get from adata.uns if not specified)
        expected_betti = params.get('expected_betti_numbers', None)
        if expected_betti is None and 'expected_betti_numbers' in self.adata.uns:
            expected_betti = self.adata.uns['expected_betti_numbers']
            self.log.info(f"Using expected_betti_numbers from adata.uns: {expected_betti}")
        
        # Betti curve computation (requires gtda)
        compute_betti = params.get('compute_betti', True)
        
        # Trustworthiness neighborhood sizes
        trustworthiness_neighbors = params.get(
            'trustworthiness_neighbors', 
            [10, 30, 50, 100]
        )
        
        # Create topology metrics directory
        topology_dir = self.metrics_dir / 'topology'
        topology_dir.mkdir(parents=True, exist_ok=True)
        
        evaluator = TopologyEvaluator(
            adata=self.adata,
            embedding_key=self.embedding_key,
            topology_type=topology_type,
            label_key=label_key,
            save_dir=topology_dir,
            metric=metric,
            n_neighbors=n_neighbors,
            groundtruth_key=groundtruth_key,
            expected_betti_numbers=expected_betti,
            compute_betti=compute_betti,
            trustworthiness_neighbors=trustworthiness_neighbors,
            **{k: v for k, v in params.items() if k not in [
                'topology_type', 'label_key', 'metric', 'n_neighbors',
                'groundtruth_key', 'expected_betti_numbers', 'compute_betti',
                'trustworthiness_neighbors'
            ]}
        )
        results = evaluator.evaluate()
        self.log.info(f"Topology evaluation complete. Results: {list(results.keys())}")
    
    def run_trajectory_evaluation(self, eval_config: Dict) -> None:
        """Run trajectory/pseudotime evaluation.
        
        Implements CONCORD's trajectory benchmark:
        - Compute pseudotime from embeddings via shortest path on k-NN graph
        - Correlate with ground truth time (Pearson, Spearman, Kendall)
        - Additional metrics: path curvature, monotonicity
        
        This is for datasets with developmental time/pseudotime ground truth
        (e.g., C. elegans embryo development, mouse intestine development).
        """
        from evaluation.trajectory import TrajectoryEvaluator
        from pathlib import Path
        
        params = eval_config.get('params', {})
        
        # Required: ground truth time key
        time_key = params.get('time_key', self.data_config.get('label_key', 'time'))
        
        # Optional: start/end points (auto-detect if not specified)
        start_point = params.get('start_point', None)
        end_point = params.get('end_point', None)
        auto_detect_endpoints = params.get('auto_detect_endpoints', True)
        
        # k-NN parameters
        k = params.get('k', 30)
        metric = params.get('metric', 'euclidean')
        
        # Correlation types
        correlation_types = params.get(
            'correlation_types', 
            ['pearsonr', 'spearmanr', 'kendalltau']
        )
        
        # Create trajectory metrics directory
        trajectory_dir = Path(self.save_dir) / 'metrics' / 'trajectory'
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        
        evaluator = TrajectoryEvaluator(
            adata=self.adata,
            embedding_key=self.embedding_key,
            time_key=time_key,
            save_dir=trajectory_dir,
            start_point=start_point,
            end_point=end_point,
            k=k,
            metric=metric,
            auto_detect_endpoints=auto_detect_endpoints,
            correlation_types=correlation_types,
            **{k: v for k, v in params.items() if k not in [
                'time_key', 'start_point', 'end_point', 'k', 'metric',
                'auto_detect_endpoints', 'correlation_types'
            ]}
        )
        results = evaluator.evaluate()
        self.log.info(f"Trajectory evaluation complete. Results: {results}")

    def run_survival_evaluation(self, eval_config: Dict) -> None:
        """Run survival analysis evaluation (Cox PH with site-preserved CV).

        Evaluates embeddings on survival prediction tasks using a linear Cox
        proportional hazards model.  Runs per-cancer-type with site-preserved
        cross-validation and L2 hyperparameter search.  See
        ``evaluation.survival.SurvivalEvaluator`` for full documentation.
        """
        from evaluation.survival import SurvivalEvaluator
        from pathlib import Path

        params = eval_config.get('params', {})

        event_col = params.get('event_col', 'DSS')
        time_col = params.get('time_col', 'DSS.time')
        cancer_type_col = params.get('cancer_type_col', 'cancer type abbreviation')
        n_folds = params.get('n_folds', 5)
        min_samples = params.get('min_samples', 50)
        min_events = params.get('min_events', 10)
        random_state = params.get('random_state', 42)

        # Optional: multiple endpoints
        raw_endpoints = params.get('endpoints', None)
        if raw_endpoints is not None:
            endpoints = [
                (ep['event_col'], ep['time_col'], ep.get('label', ep['event_col']))
                for ep in raw_endpoints
            ]
        else:
            endpoints = None

        # Alpha grid
        alpha_min = params.get('alpha_min', 10)
        alpha_max = params.get('alpha_max', 1e5)
        alpha_n = params.get('alpha_n', 25)
        import numpy as _np
        alpha_grid = _np.logspace(
            _np.log10(alpha_min), _np.log10(alpha_max), num=alpha_n
        )

        survival_dir = Path(self.save_dir) / 'metrics' / 'survival'
        survival_dir.mkdir(parents=True, exist_ok=True)

        plots_dir = Path(self.save_dir) / 'plots' / 'survival'
        plots_dir.mkdir(parents=True, exist_ok=True)

        n_pca_components = params.get('n_pca_components')  # None = auto: 100 if dim > 150

        evaluator = SurvivalEvaluator(
            adata=self.adata,
            embedding_key=self.embedding_key,
            event_col=event_col,
            time_col=time_col,
            cancer_type_col=cancer_type_col,
            save_dir=survival_dir,
            plots_dir=plots_dir,
            n_folds=n_folds,
            alpha_grid=alpha_grid,
            min_samples=min_samples,
            min_events=min_events,
            random_state=random_state,
            endpoints=endpoints,
            n_pca_components=n_pca_components,
        )
        results = evaluator.evaluate()
        self.log.info(f"Survival evaluation complete. Results saved to {survival_dir}")

    def run_drug_response_evaluation(self, eval_config: Dict) -> None:
        """Run drug response prediction evaluation.

        Evaluates embeddings on predicting drug sensitivity (AUC) from cell line
        embeddings using the DepMap PRISM dataset. Supports multiple generalization
        scenarios: new_cell_line, new_drug, new_both.
        """
        from evaluation.drug_response import DrugResponseEvaluator
        from pathlib import Path

        params = eval_config.get('params', {})

        cell_line_col = params.get('cell_line_col', 'depmap_id')
        drug_col = params.get('drug_col', 'broad_id')
        target_col = params.get('target_col', 'auc')
        tissue_col = params.get('tissue_col', 'cell_line_OncotreeLineage')
        moa_col = params.get('moa_col', 'moa')

        split_types = params.get('split_types', ['new_cell_line', 'new_drug', 'new_both'])
        test_size = params.get('test_size', 0.2)
        random_state = params.get('random_state', 42)

        model_type = params.get('model_type', 'both')
        task_type = params.get('task_type', 'regression')
        classification_threshold = params.get('classification_threshold', 0.9)
        classification_target = params.get('classification_target', 'zscore')
        z_score_domain = params.get('z_score_domain', 'per_lineage_drug')
        z_sensitive_threshold = params.get('z_sensitive_threshold', -1.0)
        z_resistant_threshold = params.get('z_resistant_threshold', 1.0)
        z_min_std = params.get('z_min_std', 1e-6)

        mlp_epochs = params.get('mlp_epochs', 100)
        mlp_patience = params.get('mlp_patience', 10)
        mlp_hidden_dims = params.get('mlp_hidden_dims', [256, 128, 64])
        rf_n_estimators = params.get('rf_n_estimators', 100)

        use_drug_features = params.get('use_drug_features', True)
        smiles_col = params.get('smiles_col', 'smiles')
        morgan_nbits = params.get('morgan_nbits', 1024)
        morgan_radius = params.get('morgan_radius', 2)

        drug_response_dir = Path(self.save_dir) / 'metrics' / 'drug_response'
        drug_response_dir.mkdir(parents=True, exist_ok=True)

        plots_dir = Path(self.save_dir) / 'plots' / 'drug_response'
        plots_dir.mkdir(parents=True, exist_ok=True)

        evaluator = DrugResponseEvaluator(
            adata=self.adata,
            embedding_key=self.embedding_key,
            cell_line_col=cell_line_col,
            drug_col=drug_col,
            target_col=target_col,
            tissue_col=tissue_col,
            moa_col=moa_col,
            save_dir=drug_response_dir,
            plots_dir=plots_dir,
            split_types=split_types,
            test_size=test_size,
            random_state=random_state,
            model_type=model_type,
            task_type=task_type,
            classification_threshold=classification_threshold,
            classification_target=classification_target,
            z_score_domain=z_score_domain,
            z_sensitive_threshold=z_sensitive_threshold,
            z_resistant_threshold=z_resistant_threshold,
            z_min_std=z_min_std,
            mlp_epochs=mlp_epochs,
            mlp_patience=mlp_patience,
            mlp_hidden_dims=mlp_hidden_dims,
            rf_n_estimators=rf_n_estimators,
            use_drug_features=use_drug_features,
            smiles_col=smiles_col,
            morgan_nbits=morgan_nbits,
            morgan_radius=morgan_radius,
        )
        results = evaluator.evaluate()
        self.log.info(f"Drug response evaluation complete. Results saved to {drug_response_dir}")
