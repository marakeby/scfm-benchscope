import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from os.path import join
import os
import random
import sys
from scipy import sparse
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

# scib-metrics silhouette / isolated_labels use JAX pairwise distances; full cohorts can
# request GPU allocations on the order of hundreds of GB. Cap cells before benchmark.
MAX_CELLS_FOR_BENCHMARK = 10_000


def densify_obsm_embeddings(adata: ad.AnnData, keys: list[str]) -> None:
    """scib-metrics (JAX) needs dense `AnnData.X`; sparse obsm becomes sparse `.X`."""
    for k in keys:
        if k not in adata.obsm:
            continue
        x = adata.obsm[k]
        if sparse.issparse(x):
            adata.obsm[k] = np.asarray(x.toarray(), dtype=np.float64)
        else:
            adata.obsm[k] = np.asarray(x, dtype=np.float64)


# sys.path.insert(0, '/home/jupyter/scFM_eval_')
sys.path.insert(0, '..')
from evaluation.eval import EmbeddingEvaluator
base_dir = '/home/haitham/mnt/__output_v3/exp'


def get_embd_key(model_name):

    if 'pca' in model_name :
        key = f'X_pca'    
        
    if 'hvg' in model_name :
        key = f'X_hvg'    
        
    if 'scvi' in model_name :
        key = f'X_scVI'
    if 'scgpt' in model_name :
        key = f'X_scGPT'
    if 'Geneformer' in model_name :
        key = f'X_geneformer'
    if 'gf' in model_name :
        key = f'X_geneformer'
        
    if 'scfoundation' in model_name:
        key = f'X_scfoundation'
    
    if 'cellplm' in model_name:
        key = f'X_CellPLM'
    
    if 'scimilarity' in model_name:
        key = f'X_scimilarity'
    return key

f = '/home/haitham/mnt/__output_v3/exp/hvg/seurat_4096/brca_cell_type_hvg_seurat_4096_brca_cell_type'

embedding_key = get_embd_key(f)
embedding_key

data_fname= join(f,'data.h5ad' )
embs = ad.read_h5ad(data_fname)

if embs.n_obs > MAX_CELLS_FOR_BENCHMARK:
    sc.pp.subsample(
        embs,
        n_obs=MAX_CELLS_FOR_BENCHMARK,
        copy=False,
        random_state=0,
    )

densify_obsm_embeddings(embs, [embedding_key])

bm = Benchmarker(
    embs,
    batch_key="donor_id",
    label_key="cell_types",
    # If you still OOM on GPU, either lower MAX_CELLS_FOR_BENCHMARK or disable the two
    # metrics that run JAX silhouette on all cells: isolated_labels, silhouette_label.
    bio_conservation_metrics=BioConservation(),
    batch_correction_metrics=BatchCorrection(),
    embedding_obsm_keys=[embedding_key],
    n_jobs=6,
)
bm.benchmark()