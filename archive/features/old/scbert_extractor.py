
# scbert_extractor.py

import os
import torch
import scanpy as sc
import numpy as np
from scBERT.performer_pytorch.performer_pytorch import PerformerLM  # adjust as needed
# from scBERT.preprocess import preprocess_anndata  # adjust as needed

class scBERTExtractor:
    def __init__(self, checkpoint_path, config_kwargs=None):
        self.model = self._load_model(checkpoint_path, config_kwargs)

    def _load_model(self, checkpoint_path, config_kwargs):
        config = config_kwargs or dict(num_tokens=7, dim=200, depth=6, heads=10, max_seq_len=16906, gene2vec_path='../data/gene2vec_16906.npy')
        model = PerformerLM(**config)
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model

    def preprocess(self, h5ad_path, outdir='processed'):
        adata = sc.read_h5ad(h5ad_path)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        # processed_path = os.path.join(outdir, 'preprocessed.h5ad')
        # adata.write_h5ad(processed_path)
        return adata

    def extract_embeddings(self, adata, method='cls'):
        X = torch.tensor(adata.X.A if hasattr(adata.X, 'A') else adata.X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X, return_encodings=True)  # hypothetical interface
            token_embs = outputs
        if method == 'cls':
            return token_embs[:, -1, :].numpy()
        elif method == 'mean':
            return token_embs.mean(dim=1).numpy()
        elif method == 'sum':
            return token_embs.sum(dim=1).numpy()
        else:
            raise ValueError(f"Unknown method: {method}")
