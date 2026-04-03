"""
Embedding extraction with the Nicheformer foundation model (theislab/nicheformer).

Requires the ``nicheformer`` Pixi environment (``pixi run -e nicheformer ...``).

Expected params (under ``embedding.params``):

- ``checkpoint_path``: Lightning checkpoint (``.ckpt``) for the pretrained ``Nicheformer`` module.
- ``technology_mean_path``: path to a ``.npy`` vector of per-gene scaling means (same length and
  gene order as ``adata.var`` — use the means file that matches your assay, see
  ``nicheformer/data/model_means/`` in the upstream repo).

Optional:

- ``batch_size`` (default 32), ``num_workers`` (default 0),
  ``max_seq_len`` (4096), ``aux_tokens`` (30), ``chunk_size`` (1000),
  ``embedding_layer`` (-1 = last transformer layer),
  ``use_gpu`` (default True when CUDA is available),
  ``precision`` (32 or ``"bf16-mixed"``; passed to Lightning Trainer if you extend this),
  ``split_value`` (default ``"train"``): value used for ``adata.obs["nicheformer_split"]`` when
  that column is missing (upstream ``NicheformerDataset`` filters on this column).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from features.extractor import EmbeddingExtractor
from utils.logs_ import get_logger


def _prepare_adata_for_nicheformer(adata, split_value: str):
    """Ensure ``nicheformer_split`` exists (required by ``NicheformerDataset``)."""
    adata = adata.copy()
    if "nicheformer_split" not in adata.obs.columns:
        adata.obs["nicheformer_split"] = split_value
    return adata


class NicheformerExtractor(EmbeddingExtractor):
    """Run Nicheformer pretrained weights and write cell embeddings to ``obsm``."""

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.log = get_logger()
        self.log.info("NicheformerExtractor (%s)", self.params)

    def fit_transform(self, data_loader):
        from nicheformer.data.dataset import NicheformerDataset
        from nicheformer.models import Nicheformer

        required = ("checkpoint_path", "technology_mean_path")
        for k in required:
            if k not in self.params:
                raise ValueError(f"Nicheformer extractor requires embedding.params.{k}")

        ckpt = self.params["checkpoint_path"]
        tech_path = self.params["technology_mean_path"]
        technology_mean = np.load(tech_path)
        batch_size = int(self.params.get("batch_size", 32))
        num_workers = int(self.params.get("num_workers", 0))
        max_seq_len = int(self.params.get("max_seq_len", 4096))
        aux_tokens = int(self.params.get("aux_tokens", 30))
        chunk_size = int(self.params.get("chunk_size", 1000))
        embedding_layer = int(self.params.get("embedding_layer", -1))
        split_value = str(self.params.get("split_value", "train"))
        use_gpu = bool(self.params.get("use_gpu", True))

        pl.seed_everything(int(self.params.get("seed", 42)))

        adata = _prepare_adata_for_nicheformer(data_loader.adata, split_value=split_value)
        data_loader.adata = adata

        if technology_mean.shape[0] != adata.n_vars:
            raise ValueError(
                f"technology_mean length {technology_mean.shape[0]} != n_vars {adata.n_vars}. "
                "Use a mean vector matching gene order and count in ``adata``."
            )

        dataset = NicheformerDataset(
            adata=adata,
            technology_mean=technology_mean,
            split=split_value,
            max_seq_len=max_seq_len,
            aux_tokens=aux_tokens,
            chunk_size=chunk_size,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        map_location = torch.device("cpu")
        model = Nicheformer.load_from_checkpoint(ckpt, map_location=map_location)
        model.eval()

        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = model.to(device)

        embeddings: list[np.ndarray] = []
        self.log.info("Nicheformer: extracting embeddings (%s cells, device=%s)", adata.n_obs, device)

        with torch.no_grad():
            for batch in dataloader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                emb = model.get_embeddings(batch=batch, layer=embedding_layer)
                embeddings.append(emb.detach().cpu().numpy())

        out = np.concatenate(embeddings, axis=0)
        data_loader.adata.obsm[self.output_key] = out
        return out
