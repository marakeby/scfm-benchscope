"""
KNN-based cell type annotation evaluation module.

Evaluates how well embeddings can be used to transfer cell type labels from a reference
dataset to a query dataset using k-nearest neighbors (KNN).

Supports multiple splitting strategies:
1. Even split: 50/50 per cell type
2. Fixed splits: Pre-set reference/query by obs column values (e.g. one batch/donor as ref, another as query)
3. Round-robin: Automatic ref/query by batch; for N batches, run N splits (query=batch_i, reference=batch_{(i+1)%N}), report mean ± std.

Example YAML for round-robin (uses dataset.batch_key by default):

  evaluations:
    - type: annotation
      params:
        label_key: cell_type
        split_strategies: [round_robin]
        # round_robin_batch_key defaults to dataset.batch_key (e.g. batch, donor_id, Site)
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal
from anndata import AnnData
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import scanpy as sc
from utils.logs_ import get_logger
from evaluation.eval import eval_classifier

logger = get_logger()


class AnnotationEvaluator:
    """
    Evaluates KNN-based cell type annotation performance.
    
    Splits data into reference (labeled) and query (unlabeled) sets, then uses
    KNN to transfer labels from reference to query.
    """
    
    def __init__(
        self,
        adata: AnnData,
        embedding_key: str,
        label_key: str = 'cell_type',
        save_dir: str = None,
        k: int = 15,
        metric: str = 'cosine',
        method: Literal['sklearn_knn', 'scanpy_ingest'] = 'sklearn_knn',
        split_strategies: List[Literal['even', 'fixed', 'round_robin']] = ['even'],
        random_state: int = 42,
        min_cells_per_type: int = 2,
        max_sample_size: int = 50000,
        fixed_splits: Optional[List[Dict]] = None,
        round_robin_batch_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize annotation evaluator.
        
        Args:
            adata: AnnData object with embeddings and labels
            embedding_key: Key in adata.obsm containing embeddings
            label_key: Key in adata.obs containing cell type labels
            save_dir: Directory to save results
            k: Number of nearest neighbors for KNN
            metric: Distance metric ('cosine', 'euclidean', etc.)
            method: Annotation method ('sklearn_knn' or 'scanpy_ingest')
            split_strategies: List of splitting strategies to use (even, fixed, round_robin)
            random_state: Random seed for reproducibility
            min_cells_per_type: Minimum cells per type required for splitting
            max_sample_size: Maximum number of cells to sample before splitting (for lightweight evaluation)
            fixed_splits: For strategy 'fixed', list of dicts with name, reference_key, reference_values,
                query_key, query_values (obs column and values to define reference/query subsets).
            round_robin_batch_key: For strategy 'round_robin', obs column for batch IDs. Defaults to dataset
                batch_key when not set (same column as in the YAML dataset.batch_key).
            **kwargs: Additional parameters
        """
        self.adata = adata
        self.embedding_key = embedding_key
        self.label_key = label_key
        self.save_dir = Path(save_dir) if save_dir else None
        self.k = k
        self.metric = metric
        self.method = method
        self.split_strategies = split_strategies
        self.random_state = random_state
        self.min_cells_per_type = min_cells_per_type
        self.max_sample_size = max_sample_size
        self.fixed_splits = fixed_splits if fixed_splits is not None else kwargs.get('fixed_splits', [])
        self.round_robin_batch_key = round_robin_batch_key or kwargs.get('round_robin_batch_key')
        # Cap ref/query size per split to avoid OOM in KNN (fixed/round_robin use full adata)
        self.max_cells_per_split_side: int = int(kwargs.get("max_cells_per_split_side", 25000))
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate embedding exists
        if embedding_key not in adata.obsm:
            raise ValueError(f"Embedding key '{embedding_key}' not found in adata.obsm. Available: {list(adata.obsm.keys())}")
        
        # Validate label key exists
        if label_key not in adata.obs:
            raise ValueError(f"Label key '{label_key}' not found in adata.obs. Available: {list(adata.obs.columns)}")
    
    def evaluate(self) -> Dict[str, Dict]:
        """
        Run annotation evaluation with all specified splitting strategies.
        
        Returns:
            Dictionary mapping split strategy names to results dictionaries
        """
        logger.info(f'Evaluating annotation performance for embedding: {self.embedding_key}')
        logger.info(f'Using method: {self.method}, k={self.k}, metric={self.metric}')
        logger.info(f'Split strategies: {self.split_strategies}')
        
        all_results = {}
        
        # Get embeddings and labels (coerce to string to avoid mixed-type sort errors in np.unique)
        X = self.adata.obsm[self.embedding_key]
        y_raw = self.adata.obs[self.label_key]
        y = np.asarray(
            y_raw.fillna('_missing_').astype(str).replace('nan', '_missing_').values,
            dtype=object,
        )

        # Get unique labels and their counts
        unique_labels, label_counts = np.unique(y, return_counts=True)
        label_names = unique_labels.tolist()
        logger.info(f'Found {len(label_names)} cell types: {label_names}')
        logger.info(f'Cell type counts: {dict(zip(label_names, label_counts))}')
        
        # Filter out cell types with too few cells
        valid_mask = label_counts >= self.min_cells_per_type
        valid_labels = unique_labels[valid_mask]
        valid_label_names = [lbl for lbl in label_names if lbl in valid_labels]
        
        if len(valid_label_names) < len(label_names):
            logger.warning(
                f'Filtering out {len(label_names) - len(valid_label_names)} cell types '
                f'with < {self.min_cells_per_type} cells. '
                f'Valid types: {valid_label_names}'
            )
            # Filter data to only valid labels
            valid_cell_mask = np.isin(y, valid_labels)
            X = X[valid_cell_mask]
            y = y[valid_cell_mask]
            label_names = valid_label_names
        
        # Sample balanced subset if dataset is too large
        if X.shape[0] > self.max_sample_size:
            logger.info(f'Dataset has {X.shape[0]} cells, sampling balanced subset of max {self.max_sample_size} cells')
            X, y, label_names = self._sample_balanced_subset(X, y, label_names)
            logger.info(f'After sampling: {X.shape[0]} cells')
            # Recompute label counts after sampling (y already string-coerced above)
            unique_labels, label_counts = np.unique(y, return_counts=True)
            logger.info(f'Sampled cell type counts: {dict(zip(label_names, label_counts))}')
        
        # Run evaluation for each split strategy and save incrementally
        if 'even' in self.split_strategies:
            logger.info('=' * 60)
            logger.info('Running EVEN split strategy (50/50 per cell type)')
            logger.info('=' * 60)
            even_results = self._evaluate_even_split(X, y, label_names)
            all_results['even'] = even_results
            # Save immediately after each strategy
            self._save_strategy_results('even', even_results, label_names)
        
        if 'fixed' in self.split_strategies:
            if not self.fixed_splits:
                raise ValueError(
                    "split_strategies contains 'fixed' but fixed_splits is empty or not provided. "
                    "Provide fixed_splits as a list of dicts with keys: name, reference_key, "
                    "reference_values, query_key, query_values."
                )
            logger.info('=' * 60)
            logger.info('Running FIXED split strategy (pre-set reference/query by obs)')
            logger.info('=' * 60)
            fixed_results = self._evaluate_fixed_splits()
            for name, result in fixed_results.items():
                all_results[name] = result
                lbl_names = result.get('label_names', [])
                self._save_strategy_results(name, result, lbl_names)

        if 'round_robin' in self.split_strategies:
            if not self.round_robin_batch_key:
                raise ValueError(
                    "split_strategies contains 'round_robin' but no batch column is available. "
                    "Set round_robin_batch_key in params or ensure dataset config has batch_key (used by default)."
                )
            logger.info('=' * 60)
            logger.info('Running ROUND-ROBIN split strategy (query=batch_i, reference=batch_{i+1})')
            logger.info('=' * 60)
            round_robin_results = self._evaluate_round_robin()
            if round_robin_results:
                all_results['round_robin'] = round_robin_results
                lbl_names = round_robin_results.get('label_names', [])
                self._save_strategy_results('round_robin', round_robin_results, lbl_names)

        # Create/update summary across all strategies
        self._save_summary(all_results, label_names)
        
        # Save to adata.uns
        if 'annotation' not in self.adata.uns:
            self.adata.uns['annotation'] = {}
        self.adata.uns['annotation'][self.embedding_key] = all_results
        logger.info(f"Saved annotation results to adata.uns['annotation']['{self.embedding_key}']")
        
        return all_results
    
    def _sample_balanced_subset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        label_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Sample a balanced subset of cells across all cell types.
        
        Samples equally from each cell type up to max_sample_size total cells.
        If a cell type has fewer cells than the per-type quota, all cells are taken.
        Remaining quota is distributed equally among cell types that can accommodate more.
        
        Args:
            X: Embeddings array (n_cells, n_features)
            y: Labels array (n_cells,)
            label_names: List of unique label names
        
        Returns:
            Tuple of (sampled_X, sampled_y, label_names)
        """
        n_total = X.shape[0]
        n_types = len(label_names)
        max_sample = self.max_sample_size
        
        if n_total <= max_sample:
            return X, y, label_names
        
        # Create label mapping
        label_to_int = {lbl: i for i, lbl in enumerate(label_names)}
        y_int = np.array([label_to_int[lbl] for lbl in y])
        
        # Get unique labels
        unique_labels = np.unique(y_int)
        
        # Initialize sampling: try equal per type
        target_per_type = max_sample // n_types
        sampled_indices = []
        
        rng = np.random.default_rng(self.random_state)
        
        # First pass: sample up to target_per_type from each type
        for label in unique_labels:
            label_mask = (y_int == label)
            label_indices = np.where(label_mask)[0]
            n_available = len(label_indices)
            n_to_sample = min(target_per_type, n_available)
            
            if n_to_sample > 0:
                sampled = rng.choice(label_indices, size=n_to_sample, replace=False)
                sampled_indices.extend(sampled.tolist())
        
        n_sampled = len(sampled_indices)
        
        # Second pass: if we haven't reached max_sample, distribute remaining quota
        if n_sampled < max_sample:
            remaining_quota = max_sample - n_sampled
            
            # Find cell types that can accommodate more samples
            available_types = []
            for label in unique_labels:
                label_mask = (y_int == label)
                label_indices = np.where(label_mask)[0]
                already_sampled = sum(1 for idx in sampled_indices if y_int[idx] == label)
                n_remaining = len(label_indices) - already_sampled
                if n_remaining > 0:
                    available_types.append((label, n_remaining))
            
            if available_types:
                # Distribute remaining quota equally
                quota_per_type = remaining_quota // len(available_types)
                extra_per_type = remaining_quota % len(available_types)
                
                for i, (label, n_remaining) in enumerate(available_types):
                    label_mask = (y_int == label)
                    label_indices = np.where(label_mask)[0]
                    already_sampled = set(sampled_indices)
                    available_indices = [idx for idx in label_indices if idx not in already_sampled]
                    
                    # Add extra cell to first few types if there's a remainder
                    n_to_add = quota_per_type + (1 if i < extra_per_type else 0)
                    n_to_add = min(n_to_add, len(available_indices))
                    
                    if n_to_add > 0:
                        additional = rng.choice(available_indices, size=n_to_add, replace=False)
                        sampled_indices.extend(additional.tolist())
        
        sampled_indices = np.array(sampled_indices)
        sampled_X = X[sampled_indices]
        sampled_y = y[sampled_indices]
        
        return sampled_X, sampled_y, label_names

    def _subsample_stratified(
        self,
        X: np.ndarray,
        y: np.ndarray,
        label_names: List[str],
        max_cells: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subsample (X, y) to at most max_cells while preserving label proportions.

        Used in fixed/round_robin splits to avoid OOM when ref or query is huge.
        """
        n = X.shape[0]
        if n <= max_cells:
            return X, y
        indices = np.arange(n)
        # Stratified: sample proportionally from each label
        sampled = []
        for lbl in label_names:
            mask = (y == lbl)
            idx_lbl = indices[mask]
            n_lbl = len(idx_lbl)
            if n_lbl == 0:
                continue
            # Target count for this label: proportional to max_cells
            target = max(1, int(round(n_lbl * max_cells / n)))
            target = min(target, n_lbl)
            chosen = rng.choice(idx_lbl, size=target, replace=False)
            sampled.extend(chosen.tolist())
        sampled = np.array(sampled)
        # If we overshot (e.g. rounding), trim to max_cells
        if len(sampled) > max_cells:
            sampled = rng.choice(sampled, size=max_cells, replace=False)
        return X[sampled], y[sampled]

    def _evaluate_even_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        label_names: List[str]
    ) -> Dict:
        """Evaluate with even 50/50 split per cell type."""
        from sklearn.model_selection import train_test_split
        
        # Create label mapping to integers
        label_to_int = {lbl: i for i, lbl in enumerate(label_names)}
        y_int = np.array([label_to_int[lbl] for lbl in y])
        
        # Split 50/50 per cell type
        X_ref, X_query, y_ref, y_query = train_test_split(
            X, y_int,
            test_size=0.5,
            stratify=y_int,
            random_state=self.random_state
        )
        
        logger.info(f'Even split: Reference={X_ref.shape[0]} cells, Query={X_query.shape[0]} cells')
        
        # Annotate query using reference
        y_pred, y_pred_score = self._annotate_with_knn(X_ref, y_ref, X_query, label_names)
        
        # Evaluate
        metrics_df, cls_report, per_class_df = eval_classifier(
            y_query, y_pred, y_pred_score, 'annotation_even', label_names
        )
        
        # Compute and save confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_query, y_pred, labels=range(len(label_names)))
        cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
        
        return {
            'metrics': metrics_df,
            'classification_report': cls_report,
            'per_class_metrics': per_class_df,
            'confusion_matrix': cm_df,
            'n_ref': X_ref.shape[0],
            'n_query': X_query.shape[0]
        }

    def _evaluate_fixed_splits(
        self,
        splits_list: Optional[List[Dict]] = None,
    ) -> Dict[str, Dict]:
        """
        Evaluate with pre-set reference/query splits defined by obs column values.

        Each fixed split has: name, reference_key, reference_values, query_key, query_values.
        Reference and query cells are subset by obs[ref_key].isin(ref_values) and
        obs[query_key].isin(query_values). Labels for evaluation come from label_key.

        Args:
            splits_list: Optional list of split configs. If None, uses self.fixed_splits.

        Returns:
            Dict mapping strategy names ('fixed_<name>') to result dicts.
        """
        from sklearn.metrics import confusion_matrix

        splits_to_use = splits_list if splits_list is not None else self.fixed_splits
        X_full = self.adata.obsm[self.embedding_key]
        y_full = self.adata.obs[self.label_key].values

        results = {}
        for split_cfg in splits_to_use:
            name = split_cfg.get('name')
            if not name:
                raise ValueError("Each fixed split must have a 'name' key.")
            ref_key = split_cfg.get('reference_key')
            ref_values = split_cfg.get('reference_values')
            query_key = split_cfg.get('query_key')
            query_values = split_cfg.get('query_values')
            if not all([ref_key, ref_values is not None, query_key, query_values is not None]):
                raise ValueError(
                    f"Fixed split '{name}' must have reference_key, reference_values, "
                    "query_key, query_values."
                )
            if ref_key not in self.adata.obs.columns:
                raise ValueError(f"reference_key '{ref_key}' not in adata.obs. Available: {list(self.adata.obs.columns)}")
            if query_key not in self.adata.obs.columns:
                raise ValueError(f"query_key '{query_key}' not in adata.obs. Available: {list(self.adata.obs.columns)}")
            
            ref_values = list(ref_values) if not isinstance(ref_values, list) else ref_values
            query_values = list(query_values) if not isinstance(query_values, list) else query_values
            
            ref_mask = self.adata.obs[ref_key].isin(ref_values).values
            query_mask = self.adata.obs[query_key].isin(query_values).values
            
            X_ref = X_full[ref_mask]
            y_ref = y_full[ref_mask]
            X_query = X_full[query_mask]
            y_query = y_full[query_mask]
            
            logger.info(
                f"Fixed split '{name}': reference {ref_key}={ref_values} -> {X_ref.shape[0]} cells, "
                f"query {query_key}={query_values} -> {X_query.shape[0]} cells"
            )
            
            if X_ref.shape[0] == 0:
                logger.warning(f"Fixed split '{name}': reference has 0 cells, skipping.")
                continue
            if X_query.shape[0] == 0:
                logger.warning(f"Fixed split '{name}': query has 0 cells, skipping.")
                continue
            
            # Label set: union of ref and query; then restrict to types with enough ref cells
            all_labels = np.unique(np.concatenate([y_ref, y_query]))
            label_names = sorted(all_labels.tolist())
            
            ref_counts = pd.Series(y_ref).value_counts()
            valid_label_names = [
                lbl for lbl in label_names
                if ref_counts.get(lbl, 0) >= self.min_cells_per_type
            ]
            if len(valid_label_names) < len(label_names):
                logger.warning(
                    f"Fixed split '{name}': keeping {len(valid_label_names)} cell types with "
                    f">= {self.min_cells_per_type} cells in reference. Dropped: "
                    f"{[l for l in label_names if l not in valid_label_names]}"
                )
                label_names = valid_label_names
            
            if not label_names:
                logger.warning(f"Fixed split '{name}': no cell types with enough reference cells, skipping.")
                continue
            
            # Restrict ref and query to valid labels only
            valid_set = set(label_names)
            ref_keep = np.isin(y_ref, label_names)
            query_keep = np.isin(y_query, label_names)
            X_ref = X_ref[ref_keep]
            y_ref = y_ref[ref_keep]
            X_query = X_query[query_keep]
            y_query = y_query[query_keep]
            
            if X_query.shape[0] == 0:
                logger.warning(f"Fixed split '{name}': no query cells with valid labels, skipping.")
                continue
            
            if X_ref.shape[0] < self.k:
                logger.warning(
                    f"Fixed split '{name}': reference has {X_ref.shape[0]} cells < k={self.k}, skipping."
                )
                continue

            # Subsample ref/query if too large to avoid OOM in KNN (fixed/round_robin use full adata)
            cap = self.max_cells_per_split_side
            rng = np.random.default_rng(self.random_state)
            if X_ref.shape[0] > cap or X_query.shape[0] > cap:
                if X_ref.shape[0] > cap:
                    X_ref, y_ref = self._subsample_stratified(
                        X_ref, y_ref, label_names, cap, rng
                    )
                    logger.info(
                        f"Fixed split '{name}': subsampled reference to {X_ref.shape[0]} cells (cap={cap})"
                    )
                if X_query.shape[0] > cap:
                    X_query, y_query = self._subsample_stratified(
                        X_query, y_query, label_names, cap, rng
                    )
                    logger.info(
                        f"Fixed split '{name}': subsampled query to {X_query.shape[0]} cells (cap={cap})"
                    )

            label_to_int = {lbl: i for i, lbl in enumerate(label_names)}
            y_ref_int = np.array([label_to_int[l] for l in y_ref])
            y_query_int = np.array([label_to_int[l] for l in y_query])
            
            # Annotate query using reference
            y_pred, y_pred_score = self._annotate_with_knn(X_ref, y_ref_int, X_query, label_names)
            
            # Evaluate
            strategy_name = f'fixed_{name}'
            metrics_df, cls_report, per_class_df = eval_classifier(
                y_query_int, y_pred, y_pred_score, strategy_name, label_names
            )
            
            cm = confusion_matrix(y_query_int, y_pred, labels=range(len(label_names)))
            cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
            
            result = {
                'metrics': metrics_df,
                'classification_report': cls_report,
                'per_class_metrics': per_class_df,
                'confusion_matrix': cm_df,
                'label_names': label_names,
                'n_ref': X_ref.shape[0],
                'n_query': X_query.shape[0],
            }
            results[strategy_name] = result

        return results

    def _evaluate_round_robin(self) -> Dict:
        """
        Evaluate with round-robin batch splits: for N batches, run N splits where
        split i has query=batch_i and reference=batch_{(i+1) % N}. Report mean ± std
        across the N runs (O(N) scaling).

        Uses round_robin_batch_key in adata.obs to get unique batch labels.

        Returns:
            Single result dict with aggregated metrics (mean ± std), or empty dict if < 2 batches.
        """
        batch_key = self.round_robin_batch_key
        if batch_key not in self.adata.obs.columns:
            raise ValueError(
                f"round_robin_batch_key '{batch_key}' not in adata.obs. "
                f"Available: {list(self.adata.obs.columns)}"
            )

        batches = sorted(self.adata.obs[batch_key].astype(str).unique().tolist())
        N = len(batches)
        if N < 2:
            logger.warning(
                f"Round-robin requires at least 2 batches; found {N} in '{batch_key}'. Skipping."
            )
            return {}

        logger.info(f"Round-robin: {N} batches = {batches}")

        # Build N splits: query = batch_i, reference = batch_{(i+1) % N}
        splits_list = [
            {
                'name': str(i),
                'reference_key': batch_key,
                'reference_values': [batches[(i + 1) % N]],
                'query_key': batch_key,
                'query_values': [batches[i]],
            }
            for i in range(N)
        ]
        per_split_results = self._evaluate_fixed_splits(splits_list=splits_list)

        if not per_split_results:
            return {}

        # Aggregate: mean ± std across N runs
        all_metrics = [r['metrics'] for r in per_split_results.values()]
        all_per_class = [r['per_class_metrics'] for r in per_split_results.values()]
        all_cms = [r['confusion_matrix'] for r in per_split_results.values() if r.get('confusion_matrix') is not None]

        metrics_aggregated = self._aggregate_cv_metrics(all_metrics, 'round_robin')
        per_class_aggregated = self._aggregate_cv_per_class_metrics(all_per_class, 'round_robin')

        cm_aggregated = None
        if all_cms:
            try:
                label_names = list(per_split_results.values())[0].get('label_names', [])
                cm_list = [cm_df.values for cm_df in all_cms]
                cm_mean = np.mean(cm_list, axis=0)
                cm_aggregated = pd.DataFrame(cm_mean, index=label_names, columns=label_names)
            except Exception as e:
                logger.warning(f"Error aggregating round-robin confusion matrices: {e}")

        label_names = list(per_split_results.values())[0].get('label_names', [])
        return {
            'metrics': metrics_aggregated,
            'per_class_metrics': per_class_aggregated,
            'confusion_matrix': cm_aggregated,
            'label_names': label_names,
            'n_splits': N,
            'batches': batches,
        }

    def _annotate_with_knn(
        self,
        X_ref: np.ndarray,
        y_ref: np.ndarray,
        X_query: np.ndarray,
        label_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Annotate query cells using KNN from reference.
        
        Args:
            X_ref: Reference embeddings (n_ref, n_features)
            y_ref: Reference labels as integers (n_ref,)
            X_query: Query embeddings (n_query, n_features)
            label_names: List of label names (for mapping)
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.method == 'sklearn_knn':
            # Use sklearn KNeighborsClassifier
            knn = KNeighborsClassifier(
                n_neighbors=self.k,
                metric=self.metric,
                weights='distance'  # Weight by distance
            )
            knn.fit(X_ref, y_ref)
            y_pred = knn.predict(X_query)
            y_pred_score = knn.predict_proba(X_query)
            
        elif self.method == 'scanpy_ingest':
            # Use scanpy ingest (requires AnnData objects)
            # Note: scanpy ingest works on PCA space, so we need to compute PCA first
            # Create temporary AnnData objects
            adata_ref = AnnData(X_ref)
            adata_ref.obs[self.label_key] = pd.Categorical([label_names[i] for i in y_ref])
            adata_query = AnnData(X_query)
            
            # Compute PCA on reference (scanpy ingest requires PCA)
            n_comps = min(50, min(X_ref.shape[0], X_ref.shape[1]) - 1)
            if n_comps < 1:
                n_comps = 1
            sc.tl.pca(adata_ref, n_comps=n_comps, random_state=self.random_state)
            
            # Ingest query onto reference PCA space
            sc.tl.ingest(adata_query, adata_ref, obs=self.label_key)
            
            # Get predictions
            y_pred_str = adata_query.obs[self.label_key].values
            label_to_int = {lbl: i for i, lbl in enumerate(label_names)}
            y_pred = np.array([label_to_int.get(lbl, 0) for lbl in y_pred_str])
            
            # For scanpy ingest, we don't get probabilities directly
            # Use a simple KNN to get probabilities for consistency
            knn = KNeighborsClassifier(
                n_neighbors=self.k,
                metric=self.metric,
                weights='distance'
            )
            knn.fit(X_ref, y_ref)
            y_pred_score = knn.predict_proba(X_query)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return y_pred, y_pred_score
    
    def _aggregate_cv_metrics(
        self,
        metrics_list: List[pd.DataFrame],
        estimator_name: str
    ) -> pd.DataFrame:
        """Aggregate metrics across CV folds (mean ± std)."""
        if not metrics_list:
            return pd.DataFrame()
        
        # Stack all metrics
        all_metrics = pd.concat(metrics_list, axis=1)
        
        # Compute mean and std
        mean_metrics = all_metrics.mean(axis=1)
        std_metrics = all_metrics.std(axis=1)
        
        # Create aggregated DataFrame
        aggregated = pd.DataFrame({
            'mean': mean_metrics,
            'std': std_metrics
        })
        aggregated.index.name = 'Metrics'
        
        # Also create a formatted version
        formatted = pd.DataFrame({
            estimator_name: [
                f"{mean:.4f} ± {std:.4f}" if not np.isnan(std) else f"{mean:.4f}"
                for mean, std in zip(mean_metrics, std_metrics)
            ]
        }, index=mean_metrics.index)
        formatted.index.name = 'Metrics'
        
        return formatted
    
    def _aggregate_cv_per_class_metrics(
        self,
        per_class_list: List[pd.DataFrame],
        estimator_name: str
    ) -> pd.DataFrame:
        """Aggregate per-class metrics across CV folds (mean ± std)."""
        if not per_class_list:
            return pd.DataFrame()
        
        # Stack all per-class metrics
        all_per_class = pd.concat(per_class_list, axis=1)
        
        # Compute mean and std
        mean_per_class = all_per_class.mean(axis=1)
        std_per_class = all_per_class.std(axis=1)
        
        # Create formatted version
        formatted = pd.DataFrame({
            estimator_name: [
                f"{mean:.4f} ± {std:.4f}" if not np.isnan(std) else f"{mean:.4f}"
                for mean, std in zip(mean_per_class, std_per_class)
            ]
        }, index=mean_per_class.index)
        formatted.index.name = 'Metrics'
        
        return formatted
    
    def _save_strategy_results(
        self,
        strategy_name: str,
        results: Dict,
        label_names: List[str]
    ) -> None:
        """
        Save results for a single strategy immediately.
        
        This ensures results are saved incrementally as each strategy completes.
        """
        if not self.save_dir:
            return
        
        # Save overall metrics
        if 'metrics' in results and results['metrics'] is not None:
            metrics_file = self.save_dir / f'annotation_{strategy_name}_metrics.csv'
            results['metrics'].to_csv(metrics_file)
            logger.info(f"Saved {strategy_name} metrics to {metrics_file}")
        
        # Save per-class metrics
        if 'per_class_metrics' in results and results['per_class_metrics'] is not None:
            per_class_file = self.save_dir / f'annotation_{strategy_name}_per_class_metrics.csv'
            results['per_class_metrics'].to_csv(per_class_file)
            logger.info(f"Saved {strategy_name} per-class metrics to {per_class_file}")
        
        # Save confusion matrix if available
        if 'confusion_matrix' in results and results['confusion_matrix'] is not None:
            cm_file = self.save_dir / f'annotation_{strategy_name}_confusion_matrix.csv'
            results['confusion_matrix'].to_csv(cm_file)
            logger.info(f"Saved {strategy_name} confusion matrix to {cm_file}")
        
        # Save classification report if available
        if 'classification_report' in results and results['classification_report'] is not None:
            cls_report_file = self.save_dir / f'annotation_{strategy_name}_classification_report.csv'
            cls_report_df = pd.DataFrame(results['classification_report']).T
            cls_report_df.to_csv(cls_report_file)
            logger.info(f"Saved {strategy_name} classification report to {cls_report_file}")
        
        # Update summary file incrementally
        self._update_summary_file(strategy_name, results)
    
    def _update_summary_file(self, strategy_name: str, results: Dict) -> None:
        """
        Update summary file incrementally with results from a single strategy.
        
        Reads existing summary (if any), adds/updates the new strategy, and saves.
        """
        if not self.save_dir:
            return
        
        summary_file = self.save_dir / 'annotation_summary.csv'
        
        # Extract metrics for this strategy
        if 'metrics' not in results or results['metrics'] is None:
            return
        
        metrics = results['metrics']
        if not isinstance(metrics, pd.DataFrame) or len(metrics) == 0:
            return
        
        # Extract mean values (handle formatted strings)
        row = {'strategy': strategy_name}
        for metric_name in metrics.index:
            value = metrics.loc[metric_name, metrics.columns[0]]
            if isinstance(value, str):
                # Extract mean from "mean ± std" format
                try:
                    mean_val = float(value.split(' ± ')[0])
                except:
                    mean_val = np.nan
            else:
                mean_val = value
            row[metric_name] = mean_val
        
        # Read existing summary or create new
        if summary_file.exists():
            try:
                summary_df = pd.read_csv(summary_file)
                # Remove existing row for this strategy if it exists
                summary_df = summary_df[summary_df['strategy'] != strategy_name]
            except Exception as e:
                logger.warning(f"Error reading existing summary file: {e}. Creating new summary.")
                summary_df = pd.DataFrame()
        else:
            summary_df = pd.DataFrame()
        
        # Add new row
        new_row_df = pd.DataFrame([row])
        if len(summary_df) == 0:
            summary_df = new_row_df
        else:
            summary_df = pd.concat([summary_df, new_row_df], ignore_index=True)
        
        # Save updated summary
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Updated summary file with {strategy_name} results: {summary_file}")
    
    def _save_summary(self, all_results: Dict, label_names: List[str]) -> None:
        """
        Create/update summary file that aggregates metrics across all strategies.
        
        This is called after all strategies complete to ensure final summary is correct.
        Note: Summary is already updated incrementally, but this ensures consistency.
        """
        if not self.save_dir:
            return
        
        summary_rows = []
        for strategy_name, results in all_results.items():
            if 'metrics' in results and results['metrics'] is not None:
                metrics = results['metrics']
                if isinstance(metrics, pd.DataFrame) and len(metrics) > 0:
                    # Extract mean values (handle formatted strings)
                    row = {'strategy': strategy_name}
                    for metric_name in metrics.index:
                        value = metrics.loc[metric_name, metrics.columns[0]]
                        if isinstance(value, str):
                            # Extract mean from "mean ± std" format
                            try:
                                mean_val = float(value.split(' ± ')[0])
                            except:
                                mean_val = np.nan
                        else:
                            mean_val = value
                        row[metric_name] = mean_val
                    summary_rows.append(row)
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_file = self.save_dir / 'annotation_summary.csv'
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"Final summary saved to {summary_file}")
