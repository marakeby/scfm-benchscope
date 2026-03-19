# cellplm_extractor.py

import scanpy as sc
import numpy as np
import anndata as ad
from CellPLM.pipeline.cell_type_annotation import CellTypeAnnotationPipeline
from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline
from utils.logs_ import get_logger
from features.extractor import EmbeddingExtractor
import mygene

def convert_symbols_to_ensembl(adata: ad.AnnData, symbol_col: str = None, new_col: str = "ensembl_id") -> ad.AnnData:
    """
    Convert gene symbols in an AnnData object to Ensembl IDs using mygene.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    symbol_col : str, optional
        Column in `adata.var` that contains gene symbols. 
        If None, uses `adata.var_names`.
    new_col : str, optional
        Column name to store Ensembl IDs. Default is 'ensembl_id'.
    
    Returns
    -------
    AnnData
        AnnData object with new column `new_col` containing Ensembl IDs.
    """
    mg = mygene.MyGeneInfo()
    
    # Extract symbols
    if symbol_col is None:
        symbols = adata.var_names.tolist()
    else:
        symbols = adata.var[symbol_col].tolist()
    
    # Query mygene for mapping
    query_res = mg.querymany(
        symbols,
        scopes="symbol",
        fields="ensembl.gene",
        species="human"
    )
    
    # Convert results into mapping dict
    mapping = {}
    for r in query_res:
        if "notfound" in r and r["notfound"]:
            continue
        if "ensembl" in r:
            if isinstance(r["ensembl"], list):
                mapping[r["query"]] = r["ensembl"][0]["gene"]  # take first
            else:
                mapping[r["query"]] = r["ensembl"]["gene"]
    
    # Map back to AnnData
    ensembl_ids = [mapping.get(sym, None) for sym in symbols]
    adata.var[new_col] = ensembl_ids
    # print('ensembl_ids', ensembl_ids)
    return adata


class CellPLMExtractor(EmbeddingExtractor):
    """
    Extract single-cell embeddings using a pretrained CellPLM model.

    Steps:
    1. Preprocess AnnData (normalization, optional filtering).
    2. Load pretrained CellPLM via CellTypeAnnotationPipeline.
    3. Use pipeline.extract_features to generate cell embeddings.
    """
    def __init__(self, params):
        super().__init__(params)
        self.log = get_logger()
        self.log.info(f'CellPLMExtractor ({self.params})')
        pretrain_prefix = self.params['pretrain_prefix']
        pretrain_directory = self.params['pretrain_directory']
    # def __init__(self, pretrain_prefix: str, pretrain_directory: str):
        self.pipeline = CellEmbeddingPipeline(
            pretrain_prefix=pretrain_prefix,
            pretrain_directory=pretrain_directory
        )

    def preprocess(self, adata: sc.AnnData) -> sc.AnnData:
        sc.pp.normalize_total(adata, target_sum=1e4)
        # sc.pp.log1p(adata)
        # adata = convert_symbols_to_ensembl(adata)
        return adata

    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        """
        Returns:
            embeddings: NumPy array (n_cells, embedding_dim)
        """
        DEVICE = 'cuda'
        embedding = self.pipeline.predict(adata, # An AnnData object
                device=DEVICE) # Specify a gpu or cpu for model inference

        # adata.obsm['emb'] = embedding.cpu().numpy()

        # embeddings = self.pipeline.transform(adata)
        return embedding.cpu().numpy()
    
    def fit_transform(self, data_loader):
        self.data_loader = data_loader
        adata = data_loader.adata
        adata = self.preprocess( adata)
        embeddings = self.extract_embeddings(adata)
        self.data_loader.adata.obsm['X_CellPLM'] = embeddings
        return embeddings

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract embeddings using CellPLM")
    parser.add_argument("--h5ad", required=True, help="Path to .h5ad input file")
    parser.add_argument("--pretrain_prefix", required=True, help="Prefix identifying checkpoint")
    parser.add_argument("--pretrain_dir", required=True, help="Directory holding pretrained checkpoints")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for inference if available")
    parser.add_argument("--output", default="cellplm_embeddings.npy", help="Output path for embeddings (.npy)")
    # args
