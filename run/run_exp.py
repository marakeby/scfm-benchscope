"""
Main experiment runner for scFM_eval.
Handles configuration, data loading, preprocessing, feature extraction, and evaluation.
"""
import importlib
import sys
import os
import warnings
import tempfile
import atexit
import glob
from os.path import dirname, abspath, join, exists
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import anndata as ad
import multiprocessing


dir_path = dirname(dirname(abspath(__file__)))
sys.path.insert(0, dir_path)

from viz.visualization import EmbeddingVisualizer
from utils.logs_ import set_logging
from utils.data_state import DataState, get_data_state, set_data_state, ensure_state, get_state_summary
from setup_path import OUTPUT_PATH, PARAMS_PATH, EMBEDDINGS_PATH, TEMP_PATH
from run.utils import timing, get_configs, get_embedding_key, configure_temp_directory, cleanup_temp_files
from run.evaluation_runner import EvaluationRunner



# Configure temp directory on module import
configure_temp_directory()

# Register cleanup function to run on exit
atexit.register(cleanup_temp_files)


class Experiment:
    """
    Handles the execution of a machine learning experiment defined by a YAML config.
    Handles loading data, preprocessing, feature extraction, and evaluation.
    """
    def __init__(self, config_path):
        """
        Initialize the Experiment with a given config path.
        Sets up directories, logging, and loads configuration.
        """
        # Handle both relative and absolute config paths
        if os.path.isabs(config_path):
            self.config_path = config_path
        else:
            # Avoid doubling "yaml/" when user passes path relative to repo root (e.g. yaml/task/method/config.yaml)
            rel = config_path.replace("\\", os.sep)
            if rel.startswith("yaml" + os.sep) or rel == "yaml":
                rel = rel[len("yaml") + 1 :].lstrip(os.sep)
            self.config_path = join(PARAMS_PATH, rel)
        
        # Verify config file exists
        if not exists(self.config_path):
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                f"Original path provided: {config_path if not os.path.isabs(config_path) else 'N/A'}\n"
                f"PARAMS_PATH: {PARAMS_PATH}"
            )
        
        self.run_id, self.data_config, self.qc_config, self.preproc_config, self.hvg, self.feat_config, self.evaluations_config = get_configs(self.config_path)

        self.vis_embedding = bool(self.feat_config.get('viz', False))
        
        # Extract dataset_name, task_name, method_name from config (and optional dataset_subgroup from path)
        # Path can be: task/method/dataset.yaml (flat) or task/method/subgroup/dataset.yaml (hierarchical)
        self.dataset_name = self.data_config.get('dataset_name')
        self.task_name = self.data_config.get('task_name')
        self.method_name = self.feat_config.get('method')
        self.dataset_subgroup = None  # e.g. HBio_HTech when path is batch_integration/scconcept/HBio_HTech/perturbseq_competition.yaml
        
        # Extract dataset_subgroup from config path if hierarchical structure is used
        # Path structure: yaml/task_dir/method/[subgroup/]dataset.yaml
        # If 4 levels deep (task_dir/method/subgroup/dataset.yaml), extract subgroup
        try:
            if os.path.isabs(config_path) and str(config_path).startswith(str(PARAMS_PATH)):
                rel_path = os.path.relpath(config_path, PARAMS_PATH)
            else:
                rel_path = config_path
            path_parts = rel_path.split(os.sep)
            # If 4+ levels deep: task_dir/method/subgroup/dataset.yaml
            if len(path_parts) >= 4:
                # path_parts[0] = task_dir (e.g., batch_integration)
                # path_parts[1] = method (e.g., scconcept)
                # path_parts[2] = subgroup (e.g., HBio_HTech)
                # path_parts[3] = dataset.yaml (e.g., perturbseq_competition.yaml)
                self.dataset_subgroup = path_parts[2]
        except Exception:
            pass  # If path parsing fails, just don't set subgroup
        
        # Validate that all required fields are present
        if self.task_name is None:
            raise ValueError(
                f"task_name is required but not found in config or path. "
                f"Please set 'task_name' in the 'dataset' section of the config file, "
                f"or ensure the config path follows the pattern: task_name/method_name/dataset_name.yaml. "
                f"Config path: {config_path}"
            )
        if self.method_name is None:
            raise ValueError(
                f"method is required but not found in config or path. "
                f"Please set 'method' in the 'embedding' section of the config file, "
                f"or ensure the config path follows the pattern: task_name/method_name/dataset_name.yaml. "
                f"Config path: {config_path}"
            )
        if self.dataset_name is None:
            raise ValueError(
                f"dataset_name is required but not found in config. "
                f"Please set 'dataset_name' in the 'dataset' section of the config file. "
                f"Config path: {config_path}"
            )
        
        # Build output path: OUTPUT_PATH / task / method / [subgroup /] dataset /
        # When YAMLs are under task/method/subgroup/*.yaml, include subgroup in path (e.g. batch_denoising/scConcept/HBio_HTech/perturbseq_competition)
        config_filename = os.path.basename(self.config_path)
        save_dir = OUTPUT_PATH / self.task_name / self.method_name
        if self.dataset_subgroup:
            save_dir = save_dir / self.dataset_subgroup / self.dataset_name
        else:
            save_dir = save_dir / self.dataset_name
        save_dir = str(save_dir)
        self.save_dir = save_dir + f'_{self.run_id}' if self.run_id else save_dir

        # Create organized subdirectories
        self.config_dir = join(self.save_dir, 'config')
        self.logs_dir = join(self.save_dir, 'logs')
        self.plots_dir = join(self.save_dir, 'plots')
        self.metrics_dir = join(self.save_dir, 'metrics')
        
        # Create subdirectories for plots
        self.plots_embeddings_dir = join(self.plots_dir, 'embeddings')
        self.plots_distributions_dir = join(self.plots_dir, 'distributions')
        self.plots_evaluations_dir = join(self.plots_dir, 'evaluations')
        
        # Metrics are saved directly to metrics_dir (no nested subdirectories)
        # Since each YAML runs one task, we don't need task-specific subdirectories
        self.metrics_batch_effects_dir = self.metrics_dir
        self.metrics_biological_signal_dir = self.metrics_dir
        self.metrics_classification_dir = self.metrics_dir
        self.metrics_annotation_dir = self.metrics_dir
        
        # Create all directories (exist_ok=True for parallel runs where another worker may create first)
        for dir_path in [self.save_dir, self.config_dir, self.logs_dir, self.plots_dir,
                        self.metrics_dir, self.plots_embeddings_dir,
                        self.plots_distributions_dir, self.plots_evaluations_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Copy config_file.yaml to the config subdirectory
        shutil.copyfile(self.config_path, join(self.config_dir, config_filename))

        # Set logging format (save logs to logs subdirectory)
        self.log = set_logging(self.logs_dir)

        # Placeholders for data and embeddings
        self.data = None
        self.embedding = None
        self.embedding_key = None
        
        # Set up embedding cache path: EMBEDDINGS_PATH / dataset_name / method.h5ad
        # Use dataset_name from config (already set above)
        # For integration baselines (harmony, bbknn, scanorama, pca_qc), the embedding
        # depends on batch_key, so include the subgroup in the cache filename to avoid
        # stale cache reuse across different batch key configs of the same dataset.
        self.embeddings_dir = EMBEDDINGS_PATH / self.dataset_name
        _integration_baselines = {'harmony', 'bbknn', 'scanorama', 'pca_qc', 'scvi', 'scanvi'}
        if self.method_name.lower() in _integration_baselines and self.dataset_subgroup:
            self.embedding_cache_path = self.embeddings_dir / f'{self.method_name}_{self.dataset_subgroup}.h5ad'
        else:
            self.embedding_cache_path = self.embeddings_dir / f'{self.method_name}.h5ad'
        
        # Set up PCA/UMAP cache path: EMBEDDINGS_PATH / dataset_name / pca_umap.h5ad
        # This is dataset-specific (not method-specific) so can be reused across methods
        self.pca_umap_cache_path = self.embeddings_dir / 'pca_umap.h5ad'

    @timing
    def run(self):
        """
        Run the complete experiment workflow.
        
        Workflow:
        1. Load data and prepare gene identifiers
        2. Check if data is normalized; normalize if needed (BEFORE embeddings)
        3. Generate embeddings for ALL cells (no QC) OR load cached embeddings
        4. Apply QC/preprocessing (for evaluations only)
        5. Run evaluations
        """
        # Step 1: Load data
        self.load_data()
        self.loader.ensure_both_gene_identifiers(normalize=True)
        self.data = self.loader.adata
        
        # Step 2: Check and normalize data BEFORE embedding generation
        # This ensures embeddings are generated from normalized data, and extractors
        # won't try to normalize again (they check adata.uns['preprocessed'] flag)
        self.log.info('Checking if data is already normalized...')
        self.check_and_normalize_for_embeddings()
        
        # Step 3: Get embeddings (generate or load from cache)
        embeddings_exist = self.check_model_embeddings_in_dataset()
        if embeddings_exist:
            self.log.info('Loading cached embeddings...')
            try:
                self.load_cached_embeddings()
            except ValueError as e:
                if "Embedding cache does not contain" in str(e) and self.feat_config.get(
                    "recompute_if_cache_mismatch", True
                ):
                    self.log.warning(
                        "Cache cell set does not match current dataset (e.g. data path/QC changed). "
                        "Recomputing embeddings and overwriting cache."
                    )
                    self.generate_embeddings()
                    self.save_embeddings()
                else:
                    raise
        else:
            self.log.info('Generating embeddings for all cells...')
            self.generate_embeddings()
            self.save_embeddings()
        
        # Step 4: Apply QC/preprocessing (for evaluations)
        self.log.info('Applying QC and preprocessing for evaluations...')
        self.qc_data()
        self.preprocess_data()
        # Note: HVG filtering is skipped by default because:
        # 1. Embeddings are computed BEFORE HVG filtering
        # 2. Downstream evaluations use embeddings, not expression matrix
        # 3. HVG on non-log-transformed data can cause overflow errors
        # Only run HVG if explicitly required (hvg.required: true)
        if self.hvg and self.hvg.get('required', False):
            self.filter_hvg()
        else:
            self.log.info('Skipping HVG filtering (embeddings already computed, not needed for evaluations)')
        
        # Update embedding reference after filtering (obsm was auto-filtered with adata)
        self.embedding = self.loader.adata.obsm[self.embedding_key]
        self.log.info(f'After QC: {self.embedding.shape[0]} cells with embeddings')
        
        # Step 5: Run evaluations (skip embedding viz when deferred, e.g. parallel runner)
        skip_plotting = os.environ.get('SKIP_PLOTTING', '0') == '1'
        if self.vis_embedding and not skip_plotting:
            self.visualize_embedding()
        self.run_evaluations()

        # Step 6: Generate summaries (can be skipped via env var or YAML; deferred in parallel runs)
        skip_summaries = (
            os.environ.get('SKIP_SUMMARIES', '0') == '1'
            or self.data_config.get('skip_summaries', False)
        )
        if skip_summaries:
            self.log.info('Skipping summary generation (SKIP_SUMMARIES=1 or skip_summaries: true)')
        else:
            self.generate_summaries()

    def load_class(self, module_path, class_name):
        """
        Dynamically import and return a class by module and name.

        Args:
            module_path (str): Python module path.
            class_name (str): Name of the class to import.

        Returns:
            type: The class object.
        """
        module = importlib.import_module(module_path)
        return getattr(module, class_name)


    @timing
    def load_data(self):
        """Load the dataset."""
        loader_config = self.data_config.copy()
        LoaderClass = self.load_class(loader_config['module'], loader_config['class'])
        self.log.info(f'Data Loader config: {loader_config}')
        self.log.info('Loading dataset...')
        self.loader = LoaderClass(loader_config)
        self.data = self.loader.load()
        self.log.info(f'Dataset loaded successfully. Shape: {self.data.shape}')
    
    @timing
    def check_and_normalize_for_embeddings(self):
        """
        Ensure data is in the correct state for embedding generation.
        
        Uses the centralized data_state module for clean state management.
        
        Model-specific handling:
        - STACK, scConcept: Expect NORMALIZED (TP10K) data, apply log1p internally
        - Other methods: Expect LOG1P (TP10K + log1p) data
        
        The data_state module handles all detection and transformation logic.
        """
        preproc_config = self.preproc_config
        method_name = self.method_name.lower()

        # YAML override: declare data state so we don't rely on heuristics (avoids e.g. log1p detected as normalized)
        yaml_data_state = preproc_config.get('data_state')
        if yaml_data_state is not None:
            try:
                declared = DataState(str(yaml_data_state).strip().lower())
                set_data_state(self.loader.adata, declared)
                self.log.info(f'Data state set from YAML (preprocessing.data_state): {declared.value}')
            except ValueError:
                self.log.warning(
                    f'Invalid preprocessing.data_state="{yaml_data_state}"; use one of raw, normalized, log1p. Ignoring.'
                )

        # Log current state
        self.log.info(f'Data state before preprocessing: {get_state_summary(self.loader.adata)}')
        
        # Determine target state based on method
        # Models that apply log1p internally expect NORMALIZED (not LOG1P)
        models_apply_log1p_internally = ('stack', 'scconcept', 'sc concept', 'scsimilarity')
        skip_log1p_for_method = any(m in method_name for m in models_apply_log1p_internally)
        
        # Allow YAML override
        yaml_log1p_setting = preproc_config.get('apply_log1p_for_embeddings')
        if yaml_log1p_setting is False:
            skip_log1p_for_method = True
            self.log.info('YAML config: apply_log1p_for_embeddings=False')
        elif yaml_log1p_setting is True:
            skip_log1p_for_method = False
            self.log.info('YAML config: apply_log1p_for_embeddings=True')
        
        # Set target state
        if skip_log1p_for_method:
            target_state = DataState.NORMALIZED
            self.log.info(f'{method_name} applies log1p internally → target state: NORMALIZED')
        else:
            target_state = DataState.LOG1P
            self.log.info(f'{method_name} expects log1p data → target state: LOG1P')
        
        # Get current state
        current_state = get_data_state(self.loader.adata)
        
        # Transform if needed
        if current_state == target_state:
            self.log.info(f'Data already in {target_state.value} state, no transformation needed')
            set_data_state(self.loader.adata, current_state)  # Ensure flag is set
        elif current_state == DataState.LOG1P and target_state == DataState.NORMALIZED:
            # Data is log1p but model expects NORMALIZED; extractor uses get_data_state(adata) and skips internal log1p
            self.log.info(
                f'Data is LOG1P but {method_name} expects NORMALIZED. '
                f'Extractor will skip internal log1p to avoid double transformation.'
            )
            # Keep as LOG1P; set flag so get_data_state stays LOG1P for extractor
            set_data_state(self.loader.adata, DataState.LOG1P)
        else:
            # Apply transformation
            target_sum = preproc_config.get('target_sum', 10000)
            ensure_state(self.loader.adata, target_state, target_sum=target_sum)
        
        self.log.info(f'Data state after preprocessing: {get_state_summary(self.loader.adata)}')
    
    def check_model_embeddings_in_dataset(self) -> bool:
        """
        Check if cached embeddings exist for this dataset/method combination.
        
        Returns:
            bool: True if embeddings cache file exists, False otherwise.
        """
        return self.embedding_cache_path.exists()
    
    @timing
    def load_cached_embeddings(self):
        """Load cached embeddings and align with current adata by cell index."""
        self.embedding_key = get_embedding_key(self.feat_config)

        self.log.info(f'Loading cached embeddings from: {self.embedding_cache_path}')
        emb_adata = ad.read_h5ad(self.embedding_cache_path)

        # Ensure all current adata cells exist in the embedding cache (avoid "not valid obs/var names" from AnnData)
        current_idx = self.loader.adata.obs.index
        emb_names = emb_adata.obs_names
        missing = current_idx.difference(emb_names)
        if len(missing) > 0:
            sample = list(missing[:5])
            raise ValueError(
                f"Embedding cache does not contain {len(missing)} cell(s) from the current dataset. "
                f"Cache has {emb_adata.n_obs} cells; current adata has {self.loader.adata.n_obs}. "
                f"Example missing obs names: {sample}. "
                "Regenerate embeddings for this dataset or use a cache produced from the same data (same path/QC)."
            )

        # Align by cell index (embeddings were generated for all cells)
        self.embedding = emb_adata[current_idx].obsm[self.embedding_key]
        self.loader.adata.obsm[self.embedding_key] = self.embedding

        self.log.info(f'Loaded embeddings. Key: {self.embedding_key}, Shape: {self.embedding.shape}')
    
    @timing
    def qc_data(self):
        """Apply quality control filters to the dataset.

        Supports from YAML 'qc' section:
        - min_genes, min_cells: threshold-based filtering (filter_cells / filter_genes).
        - remove_outliers: if true, apply MAD-based outlier removal (total counts, genes, pct_mt).
        - mad_nmads: optional dict (total_counts, genes_by_counts, pct_counts_mt, ...) n MADs.
        - pct_counts_mt_max: remove cells with pct_mt > this (default 8).
        - use_top20: if true, also use pct_counts_in_top_20_genes in MAD removal.
        """
        qc_config = self.qc_config
        if ('skip' in qc_config) and (qc_config['skip'] is True):
            return
        qc_params = {
            'min_genes': qc_config.get('min_genes'),
            'min_cells': qc_config.get('min_cells'),
            'remove_outliers': qc_config.get('remove_outliers', False),
            'mad_nmads': qc_config.get('mad_nmads'),
            'pct_counts_mt_max': qc_config.get('pct_counts_mt_max', 8.0),
            'use_top20': qc_config.get('use_top20', False),
        }
        self.loader.qc(**qc_params)

    @timing
    def preprocess_data(self):
        """
        Apply preprocessing steps for evaluations.
        
        Note: Normalization is already done before embedding generation.
        This method now only handles plotting distributions for evaluations.
        If data was already normalized, this will detect and skip re-normalization.
        """
        preproc_config = self.preproc_config
        
        self.log.info('=' * 60)
        self.log.info('Checking preprocessing status for evaluations')
        self.log.info('=' * 60)
        
        # Check if data is already preprocessed (from embedding generation step)
        if hasattr(self.loader.adata, 'uns') and self.loader.adata.uns.get('preprocessed', False):
            self.log.info('Data already preprocessed before embeddings. Skipping normalization.')
            # Still plot distribution for evaluations
            if self.plots_distributions_dir:
                self.loader.plot_expression_distribution(
                    self.plots_distributions_dir, 
                    title_prefix="Post-normalization "
                )
            return
        
        # If somehow not preprocessed, apply preprocessing now
        self.log.info('Applying preprocessing (normalization + log1p) for evaluations...')
        scale_params = {
            'normalize': preproc_config.get('normalize'),
            'target_sum': preproc_config.get('target_sum'),
            'apply_log1p': preproc_config.get('apply_log1p'),
            'method': preproc_config.get('method'),
            'plots_dir': self.plots_distributions_dir
        }
        self.loader.scale(**scale_params)
        self.log.info('Preprocessing complete.')

    @timing
    def filter_hvg(self):
        """Select High Variant Genes only, if specified in config."""
        hvg_config = self.hvg
        if ('skip' in hvg_config) and (hvg_config['skip'] is True):
            self.log.info('Skipping HVG. Analyzing all genes')
            return
        # Filter config to only include parameters accepted by hvg()
        hvg_params = {
            'n_top_genes': hvg_config.get('n_top_genes'),
            'flavor': hvg_config.get('flavor'),
            'batch_key': hvg_config.get('batch_key')
        }
        self.loader.hvg(**hvg_params)

    @timing
    def generate_embeddings(self):
        """Extract features or embeddings from the dataset.
        
        Uses the ExtractorRegistry to get the appropriate extractor for the method.
        The registry handles:
        - Environment switching for models requiring separate envs
        - Subprocess dispatch when needed
        - Default parameter merging
        """
        from extractors import get_extractor, ExtractorRegistry
        
        self.log.info('=' * 60)
        self.log.info('Extract features or embeddings from the dataset')
        self.log.info('=' * 60)
        feat_config = self.feat_config
        
        # Extract embeddings
        if 'methods' in feat_config and isinstance(feat_config['methods'], list):
            # Multiple methods: extract each and concatenate
            self.log.info(f'Extracting embeddings from {len(feat_config["methods"])} methods')
            embeddings_list = []
            
            for method_config in feat_config['methods']:
                method_name = method_config['name']
                self.log.info(f'Extracting embeddings using {method_name}')
                method_key = f'X_{method_name.lower()}'
                
                # Build config for this method
                single_config = method_config.get('params', {}).copy()
                single_config['save_dir'] = self.save_dir
                
                # Get extractor from registry
                try:
                    extractor = get_extractor(method_name, single_config)
                    method_embedding = extractor.fit_transform(self.loader)
                    embeddings_list.append(method_embedding)
                except Exception as e:
                    # Check if it's a species mismatch (should skip gracefully)
                    from extractors.scconcept.extract import SpeciesMismatchError
                    if isinstance(e, SpeciesMismatchError) or 'species mismatch' in str(e).lower() or 'non-human gene' in str(e).lower():
                        self.log.warning(
                            f"⏭ Skipping {method_name} extraction due to species mismatch: {e}"
                        )
                        continue
                    else:
                        raise
                
                # Store individual embeddings in adata.obsm
                if method_key not in self.loader.adata.obsm:
                    self.loader.adata.obsm[method_key] = method_embedding
            
            # Check if any embeddings were successfully extracted
            if len(embeddings_list) == 0:
                self.log.error(
                    "All embedding extraction methods were skipped (likely due to species mismatch). "
                    "Cannot proceed without embeddings."
                )
                raise ValueError(
                    "No embeddings extracted. All methods were skipped, likely due to species mismatch."
                )
            
            # Join embeddings based on method
            joining_method = feat_config.get('embedding_joining_method', 'concatenate')
            if joining_method == 'concatenate':
                self.embedding = np.concatenate(embeddings_list, axis=1)
                self.log.info(f'Concatenated embeddings: {[e.shape[1] for e in embeddings_list]} -> {self.embedding.shape[1]}')
            else:
                raise ValueError(f"Unsupported embedding joining method: {joining_method}")
            
            # Set embedding key (depends on joining method)
            self.embedding_key = get_embedding_key(feat_config)
            self.loader.adata.obsm[self.embedding_key] = self.embedding
            
        else:
            # Single method: use registry to get extractor
            self.embedding_key = get_embedding_key(feat_config)
            method_name = feat_config.get('method', self.method_name)
            
            # Build config with params
            config = feat_config.get('params', {}).copy()
            config['save_dir'] = self.save_dir
            
            # Log registry info
            if ExtractorRegistry.needs_separate_env(method_name):
                self.log.info(f'{method_name} requires separate environment')
            
            # Get extractor from registry
            extractor = get_extractor(method_name, config)
            self.embedding = extractor.fit_transform(self.loader)
            self.loader.adata.obsm[self.embedding_key] = self.embedding
            
            self.log.info(f'Extracted {method_name} embeddings: shape {self.embedding.shape}')
    
    @timing
    def save_embeddings(self):
        """
        Save embeddings to a lightweight h5ad cache file.
        
        Saves only obs.index (cell IDs) and obsm (embeddings) for easy reuse.
        """
        # Create embeddings directory if needed
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Create lightweight adata with just cell indices and embeddings
        emb_adata = ad.AnnData(obs=self.loader.adata.obs[[]])  # Empty obs, keeps index
        emb_adata.obsm[self.embedding_key] = self.embedding
        emb_adata.uns['params'] = {
            'method': self.method_name,
            'dataset': self.dataset_name,
            'embedding_key': self.embedding_key,
            'n_cells': self.embedding.shape[0],
            'embedding_dim': self.embedding.shape[1],
        }
        
        # Save to cache
        emb_adata.write_h5ad(self.embedding_cache_path)
        self.log.info(f'Saved embeddings to: {self.embedding_cache_path}')

    def _sanitize_obs_for_evaluations(self) -> None:
        """
        Ensure label and batch columns in adata.obs are string type with no NaN.

        Prevents TypeError in evaluators (np.unique, BRAS, NMI/ARI) when columns
        contain mixed types (str/float) or NaN (e.g. from pandas or h5ad).
        """
        adata = self.loader.adata
        label_key = self.data_config.get('label_key')
        batch_key = self.data_config.get('batch_key')
        missing_sentinel = '_missing_'
        for key in (label_key, batch_key):
            if not key or key not in adata.obs:
                continue
            col = adata.obs[key]
            # Coerce to string; fill NaN/NA so comparisons and np.unique work
            if pd.api.types.is_categorical_dtype(col):
                adata.obs[key] = col.astype(str).replace('nan', missing_sentinel)
            else:
                adata.obs[key] = col.fillna(missing_sentinel).astype(str).replace('nan', missing_sentinel)
            n_missing = (adata.obs[key] == missing_sentinel).sum()
            if n_missing > 0:
                self.log.warning(f"obs['{key}']: {n_missing} missing values set to '{missing_sentinel}'")

    def visualize_embedding(self):
        """
        Generate and save embedding visualizations using EmbeddingVisualizer.
        Also computes and saves PCA and UMAP to the original dataset (with caching).
        """
        visualizer = EmbeddingVisualizer(
            self.embedding, 
            self.loader.adata.obs, 
            save_dir=self.plots_embeddings_dir
        )
        visualizer.plot(
            adata_original=self.loader.adata,
            embedding_key=self.embedding_key,
            save_to_adata=True,
            pca_umap_cache_path=self.pca_umap_cache_path
        )

        
    @timing
    def run_evaluations(self):
        """
        Run configured evaluations.
        
        Evaluations are configured in the 'evaluations' section of the config.
        Supported evaluation types:
        - 'biological_signal': Cell type preservation and clustering quality
        - 'batch_effects': Batch correction quality
        - 'classification': Downstream classification task performance
        - 'annotation': KNN-based cell type annotation (label transfer from reference to query)
        
        Each evaluation should have:
        - type: evaluation type
        - params: parameters for the evaluation
        - skip: optional, if True, skip this evaluation
        """
        if not self.evaluations_config:
            self.log.info('No evaluations configured. Skipping evaluation step.')
            return

        # Sanitize label/batch columns so all evaluators see consistent string types
        # (avoids TypeError: '<' not supported between 'str' and 'float' in np.unique, BRAS, etc.)
        self._sanitize_obs_for_evaluations()
        
        self.log.info('=' * 60)
        self.log.info('Running configured evaluations')
        self.log.info('=' * 60)
        
        # Initialize evaluation runner with organized directories
        eval_runner = EvaluationRunner(
            adata=self.loader.adata,
            embedding_key=self.embedding_key,
            data_config=self.data_config,
            save_dir=self.save_dir,
            log=self.log,
            metrics_batch_effects_dir=self.metrics_batch_effects_dir,
            metrics_biological_signal_dir=self.metrics_biological_signal_dir,
            metrics_classification_dir=self.metrics_classification_dir,
            metrics_annotation_dir=self.metrics_annotation_dir,
            plots_evaluations_dir=self.plots_evaluations_dir,
            task_name=self.task_name,
            dataset_name=self.dataset_name,
            dataset_subgroup=self.dataset_subgroup,
            evaluations_config=self.evaluations_config,
        )
        
        metrics_dir = Path(self.metrics_dir) if self.metrics_dir else None
        for eval_config in self.evaluations_config:
            if eval_config.get('skip', False):
                self.log.info(f"Skipping evaluation: {eval_config.get('type', 'unknown')}")
                continue
            
            eval_type = eval_config.get('type')
            if not eval_type:
                self.log.warning(f"Evaluation config missing 'type', skipping: {eval_config}")
                continue
            
            # For batch_bio_integration, skip batch_effects and biological_signal if outputs exist,
            # so that re-runs (e.g. to fix annotation metric) only re-execute annotation.
            if metrics_dir and self.task_name == 'batch_bio_integration':
                if eval_type == 'batch_effects':
                    be_file = metrics_dir / 'batch_effects' / 'batch_effects_metrics.csv'
                    if not be_file.exists():
                        be_file = metrics_dir / 'batch_effects_metrics.csv'
                    if be_file.exists():
                        self.log.info(f"Skipping evaluation: {eval_type} (output already exists)")
                        continue
                elif eval_type == 'biological_signal':
                    bio_file = metrics_dir / 'biological_signal' / 'biological_signal_metrics.csv'
                    if not bio_file.exists():
                        bio_file = metrics_dir / 'biological_signal_metrics.csv'
                    if bio_file.exists():
                        self.log.info(f"Skipping evaluation: {eval_type} (output already exists)")
                        continue
            
            self.log.info(f"Running evaluation: {eval_type}")
            
            try:
                if eval_type == 'batch_effects':
                    eval_runner.run_batch_effects_evaluation(eval_config)
                elif eval_type == 'biological_signal':
                    eval_runner.run_biological_signal_evaluation(eval_config)
                elif eval_type == 'classification':
                    eval_runner.run_classification_evaluation(eval_config, self.load_class, self.loader)
                elif eval_type == 'annotation':
                    eval_runner.run_annotation_evaluation(eval_config)
                elif eval_type == 'topology':
                    eval_runner.run_topology_evaluation(eval_config)
                elif eval_type == 'trajectory':
                    eval_runner.run_trajectory_evaluation(eval_config)
                elif eval_type == 'drug_response':
                    eval_runner.run_drug_response_evaluation(eval_config)
                elif eval_type == 'survival':
                    eval_runner.run_survival_evaluation(eval_config)
                else:
                    self.log.warning(f"Unknown evaluation type: {eval_type}. Skipping.")
            except Exception as e:
                self.log.error(f"Error running evaluation {eval_type}: {e}", exc_info=True)
    

    
    def generate_summaries(self):
        """
        Generate summary comparison tables at multiple levels of aggregation.

        Tables and boxplots use metric values in their original range and direction
        (see utils/metric_definitions.py); only global_score is normalized for ranking.
        
        Creates scoped summaries at each level:
        - Subgroup-level (if applicable): Datasets within task/method/subgroup
        - Method-level: Only datasets for this specific method within this task
        - Task-level: All methods and datasets for this specific task (with subgroup stratification if applicable)
        - Cross-task: All tasks, methods, and datasets (global view)
        
        Structure: OUTPUT_PATH / task / method / [subgroup /] dataset /
        """
        try:
            from utils.results_summarizer import ResultsSummarizer
            
            task_name = self.task_name
            method_name = self.method_name
            dataset_subgroup = self.dataset_subgroup
            
            # If we have a subgroup, generate subgroup-level summaries first
            if dataset_subgroup:
                # Subgroup-level summaries: task/method/subgroup/summaries
                # Compares datasets within this specific subgroup for this method
                subgroup_dir = OUTPUT_PATH / task_name / method_name / dataset_subgroup
                subgroup_summaries_dir = subgroup_dir / 'summaries'
                summarizer = ResultsSummarizer(subgroup_dir, scope='method_subgroup')
                summarizer.generate_all_summaries(save_dir=subgroup_summaries_dir)
                self.log.info(f"Generated subgroup-level summaries for {method_name}/{dataset_subgroup}")
            
            # Generate summaries for this method within this task (scoped to method directory)
            # Only includes datasets for this specific method (all subgroups if hierarchical)
            method_dir = OUTPUT_PATH / task_name / method_name
            method_summaries_dir = method_dir / 'summaries'
            summarizer = ResultsSummarizer(method_dir)
            summarizer.generate_all_summaries(save_dir=method_summaries_dir)
            self.log.info(f"Generated method-level summaries for {method_name} on task {task_name}")
            
            # Generate summaries for this task (scoped to task directory)
            # Includes all methods and datasets for this specific task
            # Will include subgroup comparisons if subgroups exist
            task_dir = OUTPUT_PATH / task_name
            task_summaries_dir = task_dir / 'summaries'
            summarizer = ResultsSummarizer(task_dir)
            summarizer.generate_all_summaries(save_dir=task_summaries_dir)
            self.log.info(f"Generated task-level summaries for task: {task_name}")
            
            # Generate cross-task summaries (global scope)
            # Includes all tasks, methods, and datasets
            cross_task_summaries_dir = OUTPUT_PATH / 'summaries'
            summarizer = ResultsSummarizer(OUTPUT_PATH)
            summarizer.generate_all_summaries(save_dir=cross_task_summaries_dir)
            self.log.info("Generated cross-task summaries (all methods, datasets, tasks)")
            
        except Exception as e:
            self.log.warning(f"Error generating summaries: {e}. Continuing...")
            import traceback
            self.log.warning(traceback.format_exc())
            # Don't fail the experiment if summary generation fails


def main():
    """
    Main entry point for running an experiment from the command line.
    Sets random seed, parses config path, runs experiment, and saves timing log.
    """
    from run.utils import set_random_seed, _get_timing_log
    
    try:
        set_random_seed(42)
        config_path = sys.argv[1] if len(sys.argv) > 1 else 'experiment.yaml'
        experiment = Experiment(config_path)
        experiment.run()
        timing_df = pd.DataFrame(_get_timing_log())
        timing_df.to_csv(join(experiment.save_dir, 'timing.csv'))
    finally:
        # Clean up temp files before exiting
        cleanup_temp_files()

if __name__ == '__main__':
    main()