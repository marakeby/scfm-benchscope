# cellplm_extractor.py

import json
import os
import re

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from CellPLM.model import OmicsFormer

import CellPLM.pipeline as _cellplm_pipeline
from utils.logs_ import get_logger
from features.extractor import EmbeddingExtractor


def _load_pretrain_map_location(
    pretrain_prefix: str,
    overwrite_config: dict | None = None,
    pretrain_directory: str = "./ckpt",
):
    """Mirror of CellPLM.pipeline.load_pretrain with ``map_location=cpu`` for ``torch.load``.

    Upstream loads checkpoints without ``map_location``, so CUDA-saved weights raise
    ``RuntimeError`` when ``torch.cuda.is_available()`` is false. Loading to CPU first
    is safe; ``CellEmbeddingPipeline.predict`` then moves the model with ``.to(device)``.
    """
    config_path = os.path.join(pretrain_directory, f"{pretrain_prefix}.config.json")
    ckpt_path = os.path.join(pretrain_directory, f"{pretrain_prefix}.best.ckpt")
    with open(config_path, "r") as openfile:
        config = json.load(openfile)
    if overwrite_config:
        config.update(overwrite_config)
    model = OmicsFormer(**config)
    pretrained_model_dict = torch.load(
        ckpt_path, map_location=torch.device("cpu")
    )["model_state_dict"]
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_model_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


_cellplm_pipeline.load_pretrain = _load_pretrain_map_location

from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline


def _ensembl_stable_gene_id(s: str) -> str | None:
    """Return version-stripped ID if ``s`` looks like an Ensembl gene stable ID (ENS…G…)."""
    if not isinstance(s, str) or not s.strip():
        return None
    u = s.strip().upper().split(".", 1)[0]
    if re.match(r"^ENS[A-Z]{0,10}G\d{6,}$", u):
        return u
    return None


_ENSEMBL_VAR = re.compile(r"^ENS[A-Z]{0,10}G\d{6,}(\.\d+)?$", re.IGNORECASE)


def _var_index_majority_ensembl(var_names) -> bool:
    """True if most ``var_names`` look like Ensembl gene stable IDs (human/mouse/…)."""
    s = pd.Index(var_names).astype(str)
    if len(s) == 0:
        return False
    return bool(s.str.match(_ENSEMBL_VAR, na=False).mean() > 0.5)


def _adata_ensembl_var_index_from_column(
    adata: sc.AnnData, col: str, log
) -> sc.AnnData | None:
    """If ``adata.var[col]`` has Ensembl IDs, use them as ``var_names``."""
    if col not in adata.var.columns:
        return None
    series = adata.var[col].astype(str)
    mask = series.map(lambda x: _ensembl_stable_gene_id(str(x)) is not None).fillna(False)
    if not bool(mask.any()):
        return None
    out = adata[:, mask].copy()
    out.var_names = series[mask].str.split(".").str[0].values
    out.var_names_make_unique()
    log.info("CellPLM: using existing %s column for Ensembl gene IDs (%d genes).", col, out.n_vars)
    return out


def _adata_with_ensembl_var_index(adata: sc.AnnData, params: dict, log) -> sc.AnnData:
    """Subset to genes with Ensembl IDs and set ``var_names`` from ``ensembl_id_col``."""
    ens_col = params.get("ensembl_id_col", "ensembl_id")
    from_col = _adata_ensembl_var_index_from_column(adata, ens_col, log)
    if from_col is not None:
        return from_col

    raise ValueError(
        "CellPLM: no Ensembl gene IDs in AnnData.var[%r]. "
        "Use embedding.params gene_name_id_dict (see H5ADLoader.map_ensembl) so symbols are "
        "mapped before inference, or set var_names / ensembl_id_col to Ensembl IDs, or set "
        "ensembl_auto_conversion: false if var_names are already Ensembl."
        % (ens_col,)
    )


class CellPLMExtractor(EmbeddingExtractor):
    MODELS_PATH_KEYS = frozenset({"pretrain_directory", "gene_name_id_dict"})

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
        return adata

    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        """
        Returns:
            embeddings: NumPy array (n_cells, embedding_dim)
        """
        use_gpu = bool(self.params.get("use_gpu", True))
        if use_gpu and torch.cuda.is_available():
            device: str | torch.device = "cuda"
        else:
            device = "cpu"

        # CellPLM's built-in symbol→Ensembl uses a fragile mygene dataframe path; we never
        # use it. When var_names are not in the model vocab, require Ensembl in var or map first.
        auto = bool(self.params.get("ensembl_auto_conversion", True))
        adata_work = adata.copy()
        model_genes = self.pipeline.model.gene_set
        in_vocab = pd.Index(adata_work.var_names.astype(str)).isin(model_genes)
        ensembl_auto_cellplm = auto
        if (not bool(in_vocab.any())) and auto:
            if _var_index_majority_ensembl(adata_work.var_names):
                ensembl_auto_cellplm = False
            else:
                self.log.info(
                    "CellPLM: var_names not in model vocab — using Ensembl from var columns "
                    "(e.g. after map_ensembl)."
                )
                adata_work = _adata_with_ensembl_var_index(
                    adata_work, self.params, self.log
                )
                ensembl_auto_cellplm = False

        embedding = self.pipeline.predict(
            adata_work,
            device=device,
            ensembl_auto_conversion=ensembl_auto_cellplm,
        )
        return embedding.cpu().numpy()
    
    def fit_transform(self, data_loader):
        self.data_loader = data_loader
        mapping = self.params.get("gene_name_id_dict")
        if mapping and hasattr(data_loader, "map_ensembl"):
            self.log.info("CellPLM: gene_name_id_dict set — mapping via data_loader.map_ensembl")
            data_loader.map_ensembl(mapping)
        adata = data_loader.adata
        adata = self.preprocess(adata)
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
