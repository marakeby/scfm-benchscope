"""
Batch effects evaluation module.

Evaluates how well embeddings remove batch effects while preserving biological signal.

Metric math (concise reference)
------------------------------
1) BRAS (Batch Removal Adapted Silhouette)
   - Treats BATCHES as clusters (not cell types). For each cell type, for each cell:
     a = mean distance to cells in the SAME batch
     b = mean distance to cells in OTHER batches (scib uses mean_other or furthest)
     silhouette_i = (b - a) / max(a, b)   ∈ [-1, 1]
   - Interpretation: positive s → cell closer to own batch (bad mixing); negative s → closer to
     other batches (good mixing). So higher |s| = more batch structure.
   - BRAS returns mean over cells of (1 - |silhouette|), so higher BRAS = better mixing.
   - Works with one cell type: it just computes batch-vs-batch separation on that one group.

2) CiLISI_batch (cell-type iLISI for batch)
   - For each cell type, take the subset of cells. For each cell i in that subset:
     - Look at its k nearest neighbors (in embedding space).
     - p_b = fraction of those neighbors in batch b (weighted by neighbor graph).
     - Simpson index = sum_b (p_b)².  LISI_i = 1 / Simpson.
   - So LISI_i = "effective number of batches in the neighborhood" (1 = one batch, n_batches = even mix).
   - Raw range: LISI ∈ [1, n_batches_ct]. We normalize per cell type:
     norm_ct = (mean_LISI_ct - 1) / (n_batches_ct - 1) ∈ [0, 1],
     then take cell-count-weighted mean over cell types → CiLISI_batch ∈ [0, 1].
"""
from pathlib import Path
from typing import Dict, Optional
from anndata import AnnData
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from matplotlib import pyplot as plt
from utils.logs_ import get_logger

logger = get_logger()

# Optional: scib_metrics provides BRAS (batch removal score)
try:
    import scib_metrics  # type: ignore
except ImportError:
    scib_metrics = None  # type: ignore

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData


def compute_lisi_pure_python(
    adata: AnnData,
    obs_key: str,
    n_neighbors: int = 90,
    use_rep: str | None = None,
) -> np.ndarray:
    """
    Compute Local Inverse Simpson's Index (LISI) using pure Python.

    LISI measures the effective number of categories (e.g. batches)
    in each cell's local neighborhood. Higher values indicate better mixing.

    This implementation avoids scib's compiled LISI (GLIBC issues) and
    uses Scanpy's neighbor graph.
    """
    # Ensure neighbors exist
    if "connectivities" not in adata.obsp:
        if use_rep is None:
            raise ValueError(
                "Neighbors not found. Provide use_rep or precompute neighbors."
            )
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)

    conn = adata.obsp["connectivities"]
    if not sp.issparse(conn):
        raise ValueError("Expected sparse connectivities matrix.")

    # Encode labels as integers
    labels = adata.obs[obs_key].to_numpy()
    _, label_indices = np.unique(labels, return_inverse=True)
    n_labels = label_indices.max() + 1

    n_cells = adata.n_obs
    lisi_scores = np.empty(n_cells, dtype=float)

    # Iterate over rows (cells)
    for i in range(n_cells):
        row = conn.getrow(i)
        if row.nnz == 0:
            lisi_scores[i] = 1.0
            continue

        weights = row.data
        neighbors = row.indices

        # Normalize weights
        weights = weights / weights.sum()

        # Weighted label proportions
        proportions = np.bincount(
            label_indices[neighbors],
            weights=weights,
            minlength=n_labels,
        )

        simpson = np.sum(proportions ** 2)
        lisi_scores[i] = 1.0 / simpson if simpson > 0 else 1.0

    return lisi_scores

class BatchEffectsEvaluator:
    """
    Evaluates batch effect correction in embeddings.

    Computes only:
    - BRAS (scib_metrics batch removal score)
    - CiLISI_batch (per cell-type iLISI for batch mixing, weighted mean;
      stored normalized to [0,1] as (raw - 1) / (n_batches - 1) for comparability across datasets)
    """
    
    def __init__(
        self,
        adata: AnnData,
        embedding_key: str,
        batch_key: str = 'batch',
        label_key: str = 'label',
        save_dir: str = None,
        auto_subsample: bool = True,
        metric: str = 'euclidean',
        plots_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize batch effects evaluator.
        
        Args:
            adata: AnnData object with embeddings
            embedding_key: Key in adata.obsm containing embeddings
            batch_key: Key in adata.obs containing batch labels
            label_key: Key in adata.obs containing cell type/biological labels
            save_dir: Directory to save results
            auto_subsample: Whether to subsample large datasets (>10k cells)
            metric: Distance metric for neighbor computation ('euclidean' or 'cosine').
                   Use 'cosine' for embeddings on hypersphere (e.g., normalized embeddings).
            plots_dir: Directory to save visualization plots (if None, uses save_dir)
            **kwargs: Additional parameters (e.g., n_neighbors for LISI)
        """
        self.adata = adata
        self.embedding_key = embedding_key
        self.batch_key = batch_key
        self.label_key = label_key
        self.save_dir = Path(save_dir) if save_dir else None
        self.plots_dir = Path(plots_dir) if plots_dir else (self.save_dir if self.save_dir else None)
        self.auto_subsample = auto_subsample
        self.metric = metric
        self.n_neighbors = kwargs.get('n_neighbors', 90)
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.plots_dir:
            self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate batch effects in embeddings.
        
        Returns:
            Dictionary of batch effect metrics.
        """
        logger.info(f'Evaluating batch effects for embedding: {self.embedding_key}')

        # Work on a copy and ensure unique obs/var names (avoids "not valid obs/var names" errors)
        adata_eval = self.adata.copy()

        # Coerce label/batch to string so BRAS and CiLISI never see mixed types or NaN
        for key in (self.label_key, self.batch_key):
            if key in adata_eval.obs:
                col = adata_eval.obs[key]
                adata_eval.obs[key] = (
                    col.fillna('_missing_').astype(str).replace('nan', '_missing_')
                )
        if adata_eval.obs_names.duplicated().any():
            logger.warning('Duplicate obs names detected; making unique.')
            adata_eval.obs_names_make_unique()
        if adata_eval.var_names.duplicated().any():
            logger.warning('Duplicate var names detected; making unique.')
            adata_eval.var_names_make_unique()

        # Subsample if needed
        if self.auto_subsample and adata_eval.shape[0] > 10000:
            logger.info('Subsampling dataset for batch effects evaluation (>10k cells)')
            from utils.sampling import sample_adata
            adata_eval = sample_adata(adata_eval, sample_size=5000, stratify_by=None)
            if adata_eval.obs_names.duplicated().any():
                adata_eval.obs_names_make_unique()

        # Check if multiple batches exist
        n_batches = adata_eval.obs[self.batch_key].nunique()
        if n_batches <= 1:
            logger.warning(f'Only {n_batches} batch(es) found. Batch effects evaluation requires multiple batches.')
            return {'error': 'Insufficient batches for evaluation'}
        
        results_dict: Dict[str, float] = {}

        # 1. BRAS (Batch removal score)
        if scib_metrics is not None:
            logger.info('Computing BRAS...')
            try:
                emb = np.asarray(adata_eval.obsm[self.embedding_key])
                score = scib_metrics.metrics.bras(
                    emb,
                    adata_eval.obs[self.label_key].values,
                    adata_eval.obs[self.batch_key].values,
                )
                results_dict["bras"] = float(score)
            except Exception as e:
                logger.warning(f"BRAS computation failed: {e}")
                results_dict["bras"] = np.nan
        else:
            results_dict["bras"] = np.nan

        # 5c. CiLISI_batch: per cell-type iLISI (batch mixing within cell type), weighted mean.
        # Raw iLISI for a cell type is in [1, n_batches_ct] where n_batches_ct is the number
        # of batches that cell type appears in (NOT the global n_batches). We normalize
        # per cell type to [0, 1]: norm_ct = (raw_ct - 1) / (n_batches_ct - 1), then take
        # the cell-count-weighted mean across cell types.
        logger.info('Computing CiLISI_batch...')
        try:
            cell_type_key = self.label_key
            batch_key = self.batch_key
            weighted_sum: float = 0.0
            total_cells: int = 0
            for ct in adata_eval.obs[cell_type_key].unique():
                tmp = adata_eval[adata_eval.obs[cell_type_key] == ct].copy()
                if tmp.n_obs < 2:
                    continue

                # Number of batches this cell type is sequenced in
                n_batches_ct = tmp.obs[batch_key].nunique()
                if n_batches_ct <= 1:
                    # Only one batch for this cell type — perfect mixing is trivial (score = 0)
                    # but iLISI = 1 = max; skip to avoid 0/0 in normalization
                    continue

                # Compute neighbors on subset with n_neighbors=90 for LISI
                if "neighbors" in tmp.uns:
                    tmp.uns.pop("neighbors", None)
                ilisi_cells = compute_lisi_pure_python(
                    tmp,
                    obs_key=batch_key,
                    n_neighbors=self.n_neighbors,
                    use_rep=self.embedding_key,
                )
                ct_mean_raw = float(np.nanmean(ilisi_cells))

                # Normalize this cell type: 1 (worst) -> 0, n_batches_ct (best) -> 1
                ct_norm = (ct_mean_raw - 1.0) / (float(n_batches_ct) - 1.0)
                ct_norm = float(np.clip(ct_norm, 0.0, 1.0))

                n_ct = tmp.n_obs
                weighted_sum += ct_norm * n_ct
                total_cells += n_ct

            results_dict["CiLISI_batch"] = (
                float(weighted_sum / total_cells) if total_cells > 0 else np.nan
            )
        except Exception as e:
            logger.warning(f"CiLISI_batch computation failed: {e}")
            results_dict["CiLISI_batch"] = np.nan

        # Global score for batch_effects is computed in the summarizer as mean of
        # normalized(bras, CiLISI_batch).

        # Remove NaN values
        results_dict = {k: v for k, v in results_dict.items() if not np.isnan(v)}
        
        # Save results to dataset's uns (unstructured metadata)
        # Use 'baseline_batch_effects' for baseline methods, 'batch_effects' for method-specific
        if self.embedding_key.startswith('X_baseline_'):
            uns_key = 'baseline_batch_effects'
        else:
            uns_key = 'batch_effects'
        
        if uns_key not in self.adata.uns:
            self.adata.uns[uns_key] = {}
        self.adata.uns[uns_key][self.embedding_key] = results_dict
        logger.info(f"Saved batch effects results to adata.uns['{uns_key}']['{self.embedding_key}']")
        
        # Also save to file if save_dir is provided
        # Append to existing file or create new one (don't overwrite previous embeddings)
        if self.save_dir:
            metrics_file = self.save_dir / 'batch_effects_metrics.csv'
            new_row = pd.DataFrame.from_dict({self.embedding_key: results_dict}, orient='index')
            
            if metrics_file.exists():
                try:
                    existing_df = pd.read_csv(metrics_file, index_col=0)
                    # Remove existing row for this embedding if present (update)
                    if self.embedding_key in existing_df.index:
                        existing_df = existing_df.drop(self.embedding_key)
                    # Concatenate with new row
                    combined_df = pd.concat([existing_df, new_row])
                    combined_df.to_csv(metrics_file)
                except Exception as e:
                    logger.warning(f"Error appending to existing file, overwriting: {e}")
                    new_row.to_csv(metrics_file)
            else:
                new_row.to_csv(metrics_file)
            logger.info(f"Saved batch effects results to '{metrics_file}'")
        
        logger.info(f"Batch effects evaluation complete. Metrics: {results_dict}")
        return results_dict
    
    def visualize_batch_effects(
        self,
        pre_embedding_key: Optional[str] = None,
        use_pca_for_pre: bool = True
    ) -> None:
        """
        Create UMAP visualizations: PRE only for pca_qc, POST only for other methods.
        
        - pca_qc: saves only PRE (umap_batch_pre.png, umap_label_pre.png) — it is the pre state.
        - harmony, bbknn, scvi, etc.: saves only POST (umap_batch_post.png, umap_label_post.png).
        
        Args:
            pre_embedding_key: Key for PRE (pca_qc only). Default X_pca or X.
            use_pca_for_pre: If True and 'X_pca' exists, use it for PRE.
        """
        if self.plots_dir is None:
            logger.warning("No plots_dir specified. Skipping batch effects visualization.")
            return
        
        if self.batch_key not in self.adata.obs.columns:
            logger.warning(f"Batch key '{self.batch_key}' not found in adata.obs. Skipping visualization.")
            return
        
        is_pca_qc = "pca_qc" in self.embedding_key.lower()
        
        if self.auto_subsample and self.adata.shape[0] > 10000:
            from utils.sampling import sample_adata
            adata_viz = sample_adata(self.adata, sample_size=5000, stratify_by=None)
            logger.info(f"Subsampled to {adata_viz.shape[0]} cells for visualization")
        else:
            adata_viz = self.adata.copy()
        
        # PRE UMAPs only for pca_qc (canonical pre-integration baseline)
        if is_pca_qc:
            if pre_embedding_key is None:
                pre_embedding_key = "X_pca" if (use_pca_for_pre and "X_pca" in self.adata.obsm) else "X"
                logger.info(f"Using {pre_embedding_key} for PRE embedding visualization")
            adata_umap = self._compute_umap_for_embedding(adata_viz, pre_embedding_key)
            if adata_umap is not None:
                logger.info("Creating PRE embedding visualizations (pca_qc only)")
                self._save_umap_plot(
                    adata_umap=adata_umap,
                    color_key=self.batch_key,
                    title=f"PRE embedding - colored by {self.batch_key}",
                    filename="umap_batch_pre.png",
                )
                if self.label_key in adata_umap.obs.columns:
                    self._save_umap_plot(
                        adata_umap=adata_umap,
                        color_key=self.label_key,
                        title=f"PRE embedding - colored by {self.label_key}",
                        filename="umap_label_pre.png",
                    )
        
        # POST UMAPs only for integration methods (not for pca_qc — that is the pre state)
        if not is_pca_qc and self.embedding_key in adata_viz.obsm:
            logger.info(f"Creating POST embedding visualizations (key: {self.embedding_key})")
            adata_umap_post = self._compute_umap_for_embedding(adata_viz, self.embedding_key)
            if adata_umap_post is not None:
                self._save_umap_plot(
                    adata_umap=adata_umap_post,
                    color_key=self.batch_key,
                    title=f"POST embedding ({self.embedding_key}) - colored by {self.batch_key}",
                    filename="umap_batch_post.png",
                )
                if self.label_key in adata_umap_post.obs.columns:
                    self._save_umap_plot(
                        adata_umap=adata_umap_post,
                        color_key=self.label_key,
                        title=f"POST embedding ({self.embedding_key}) - colored by {self.label_key}",
                        filename="umap_label_post.png",
                    )
    
    def _compute_umap_for_embedding(
        self,
        adata: AnnData,
        embedding_key: str
    ) -> Optional[AnnData]:
        """
        Compute UMAP for a given embedding.
        
        Args:
            adata: AnnData object
            embedding_key: Key in adata.obsm or 'X' for expression matrix
        
        Returns:
            AnnData with computed UMAP, or None if embedding not found
        """
        # Create a copy for UMAP computation
        adata_umap = adata.copy()
        
        # Determine use_rep for neighbors computation
        if embedding_key == 'X':
            use_rep = 'X'
        elif embedding_key in adata_umap.obsm:
            use_rep = embedding_key
        else:
            logger.warning(f"Embedding key '{embedding_key}' not found. Skipping visualization.")
            return None
        
        # Compute neighbors and UMAP
        logger.info(f"Computing UMAP for embedding (key: {embedding_key})")
        sc.pp.neighbors(
            adata_umap,
            use_rep=use_rep,
            n_neighbors=15,
            metric=self.metric,
            random_state=0
        )
        sc.tl.umap(adata_umap, random_state=0)
        
        return adata_umap
    
    def _save_umap_plot(
        self,
        adata_umap: AnnData,
        color_key: str,
        title: str,
        filename: str
    ) -> None:
        """
        Save a UMAP plot colored by a specific key.
        
        Args:
            adata_umap: AnnData with precomputed UMAP
            color_key: Key in adata.obs to color by
            title: Plot title
            filename: Output filename
        """
        try:
            fig = sc.pl.umap(
                adata_umap,
                color=color_key,
                show=False,
                wspace=0.4,
                frameon=False,
                return_fig=True,
                title=title
            )
            
            output_path = self.plots_dir / filename
            fig.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved UMAP plot to '{output_path}'")
            
        except Exception as e:
            logger.error(f"Error saving UMAP plot {filename}: {e}", exc_info=True)
    
