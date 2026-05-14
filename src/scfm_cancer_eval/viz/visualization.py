from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import anndata as ad
import scanpy as sc
import logging
logger = logging.getLogger('ml_logger')
from scfm_cancer_eval.utils.logs_ import get_logger

from scfm_cancer_eval.utils.sampling import sample_adata
logger = get_logger()

class EmbeddingVisualizer:
    """Class for visualizing embeddings with optional batch and label coloring."""
    def __init__(self, embedding, obs, save_dir=".", auto_subsample=True, sample_size=10000):
        
        self.auto_subsample=auto_subsample
        self.embedding = embedding
        self.sample_size = sample_size
        self.obs = obs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _embedding_copy(embedding):
        """
        Return a defensive copy so scanpy/anndata cannot mutate caller-owned data.
        Supports numpy arrays, scipy sparse matrices, and torch tensors.
        """
        # torch.Tensor -> numpy
        if hasattr(embedding, "detach") and hasattr(embedding, "cpu") and hasattr(embedding, "numpy"):
            embedding = embedding.detach().cpu().numpy()

        # scipy sparse matrices provide .copy()
        if hasattr(embedding, "tocsc") and hasattr(embedding, "tocsr") and hasattr(embedding, "copy"):
            return embedding.copy()

        # numpy / array-like
        try:
            import numpy as np
            return np.array(embedding, copy=True)
        except Exception:
            # best-effort fallback
            return embedding.copy() if hasattr(embedding, "copy") else embedding

    def plot(self):
        adata = ad.AnnData(X=self._embedding_copy(self.embedding))
        adata.obs = self.obs.copy()
        logger.info(f'Visualizing embeddings {adata.X.shape}')
        logger.info(f'Calc neighbors using X')

        if self.auto_subsample:
            if adata.shape[0]>self.sample_size:
                logger.info(f'Subsampling adata to {self.sample_size} cells for visualization')
                adata = sample_adata(adata, sample_size=self.sample_size, stratify_by='batch')
                logger.info(f'Subsampled adata to {adata.X.shape} cells')
    
            
        sc.pp.neighbors(adata, use_rep='X')
        logger.info(f'Calc UMAP')
        sc.tl.umap(adata)

        if 'batch' in adata.obs.columns:
            embeddings_fig = sc.pl.umap(adata, color='batch', show=False, wspace=0.4, frameon=False, return_fig=True)
            embeddings_fig.savefig(self.save_dir / 'embedding_batch.png', dpi=200, bbox_inches='tight')
            logger.info("Saved UMAP plot colored by batch to 'figures/umap_batch.png'")

        if 'label' in adata.obs.columns:
            embeddings_fig = sc.pl.umap(adata, color='label', show=False, wspace=0.4, frameon=False, return_fig=True)
            embeddings_fig.savefig(self.save_dir / 'embedding_label.png', dpi=200, bbox_inches='tight')
            logger.info("Saved UMAP plot colored by label to 'figures/umap_label.png'")

