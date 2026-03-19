"""
Biological signal evaluation module.

Evaluates how well embeddings preserve biological structure via NMI and ARI
from the scib_metrics Leiden pipeline (pynndescent + Leiden clustering).

Single label (one cell type): NMI and ARI are not meaningful — they compare true
labels to Leiden clusters. With one true label, entropy of true partition is 0;
NMI/ARI become 0 (if Leiden finds multiple clusters) or 1 (if one cluster), neither
measures "biological signal preservation". We skip them (set to NaN) when
label_key has only one unique value.
"""
from pathlib import Path
from typing import Dict
from anndata import AnnData
import numpy as np
import pandas as pd
from utils.logs_ import get_logger

logger = get_logger()

# Optional: scib_metrics provides NMI/ARI with Leiden clustering (pynndescent + Leiden)
try:
    import scib_metrics  # type: ignore
except ImportError:
    scib_metrics = None  # type: ignore


class BiologicalSignalEvaluator:
    """
    Evaluates biological signal preservation in embeddings.

    Computes only NMI and ARI from the scib_metrics Leiden pipeline
    (pynndescent n_neighbors=90 + Leiden), plus avg_bio (mean of nmi and ari scaled to [0, 1]).
    """
    
    def __init__(
        self,
        adata: AnnData,
        embedding_key: str,
        label_key: str = 'label',
        save_dir: str = None,
        auto_subsample: bool = True,
        metric: str = 'euclidean',
        leiden_resolution: float = 0.3,
        n_neighbors: int = 15,
        **kwargs
    ):
        """
        Initialize biological signal evaluator.
        
        Args:
            adata: AnnData object with embeddings
            embedding_key: Key in adata.obsm containing embeddings
            label_key: Key in adata.obs containing cell type/biological labels
            save_dir: Directory to save results
            auto_subsample: Whether to subsample large datasets (>10k cells)
            metric: Distance metric for neighbor computation ('euclidean' or 'cosine')
            leiden_resolution: Resolution parameter for Leiden clustering
            n_neighbors: Number of neighbors for graph construction
            **kwargs: Additional parameters
        """
        self.adata = adata
        self.embedding_key = embedding_key
        self.label_key = label_key
        self.save_dir = Path(save_dir) if save_dir else None
        self.auto_subsample = auto_subsample
        self.metric = metric
        self.leiden_resolution = leiden_resolution
        self.n_neighbors = n_neighbors
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate biological signal preservation in embeddings.
        
        Returns:
            Dictionary of biological signal metrics.
        """
        logger.info(f'Evaluating biological signal for embedding: {self.embedding_key}')
        
        # Subsample if needed
        if self.auto_subsample and self.adata.shape[0] > 10000:
            logger.info('Subsampling dataset for biological signal evaluation (>10k cells)')
            from utils.sampling import sample_adata
            adata_eval = sample_adata(self.adata, sample_size=5000, stratify_by=None)
        else:
            adata_eval = self.adata.copy()

        # Coerce labels to string and drop cells with NaN in embedding or labels (avoids NMI/ARI errors)
        if self.label_key in adata_eval.obs:
            adata_eval.obs[self.label_key] = (
                adata_eval.obs[self.label_key]
                .fillna('_missing_')
                .astype(str)
                .replace('nan', '_missing_')
            )
        emb = np.asarray(adata_eval.obsm[self.embedding_key])
        valid = ~(np.isnan(emb).any(axis=1) | np.isinf(emb).any(axis=1))
        if not valid.all():
            adata_eval = adata_eval[valid].copy()
        results_dict: Dict[str, float] = {}
        if adata_eval.n_obs == 0:
            logger.warning('No valid cells after dropping NaN/Inf in embedding. Skipping NMI/ARI.')
        else:
            # NMI and ARI require at least two distinct labels (cell types) to be meaningful.
            # With one label, they degenerate to 0 or 1 and do not measure biological preservation.
            n_labels = adata_eval.obs[self.label_key].nunique()
            if n_labels <= 1:
                logger.warning(
                    f"Only {n_labels} unique value(s) in '{self.label_key}'. "
                    "Skipping NMI/ARI (not defined for single cell type)."
                )
                results_dict["nmi"] = np.nan
                results_dict["ari"] = np.nan
            elif scib_metrics is not None:
                logger.info('Computing NMI and ARI (Leiden)...')
                try:
                    emb = np.asarray(adata_eval.obsm[self.embedding_key])
                    neigh_result_90 = scib_metrics.nearest_neighbors.pynndescent(
                        emb, n_neighbors=90, random_state=0
                    )
                    nmi_ari_dict = scib_metrics.nmi_ari_cluster_labels_leiden(
                        neigh_result_90,
                        adata_eval.obs[self.label_key].values,
                    )
                    results_dict["nmi"] = float(nmi_ari_dict["nmi"])
                    results_dict["ari"] = float(nmi_ari_dict["ari"])
                except Exception as e:
                    logger.warning(f"NMI/ARI (Leiden) computation failed: {e}")
                    results_dict["nmi"] = np.nan
                    results_dict["ari"] = np.nan
            else:
                results_dict["nmi"] = np.nan
                results_dict["ari"] = np.nan

        # avg_bio: mean of nmi and ari scaled to [0, 1] (1=best)
        scaled = []
        nmi = results_dict.get("nmi")
        if not np.isnan(nmi):
            scaled.append(float(nmi))
        ari = results_dict.get("ari")
        if not np.isnan(ari):
            scaled.append((float(ari) + 1.0) / 2.0)  # [-1, 1] -> [0, 1]
        if scaled:
            results_dict["avg_bio"] = float(np.mean(scaled))
        
        # Remove NaN values
        results_dict = {k: v for k, v in results_dict.items() if not np.isnan(v)}
        
        # Save results to dataset's uns (unstructured metadata)
        if 'biological_signal' not in self.adata.uns:
            self.adata.uns['biological_signal'] = {}
        self.adata.uns['biological_signal'][self.embedding_key] = results_dict
        logger.info(f"Saved biological signal results to adata.uns['biological_signal']['{self.embedding_key}']")
        
        # Also save to file if save_dir is provided
        # Append to existing file or create new one (don't overwrite previous embeddings)
        if self.save_dir:
            metrics_file = self.save_dir / 'biological_signal_metrics.csv'
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
            logger.info(f"Saved biological signal results to '{metrics_file}'")
        
        logger.info(f"Biological signal evaluation complete. Metrics: {results_dict}")
        return results_dict
