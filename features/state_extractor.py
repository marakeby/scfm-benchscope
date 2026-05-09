"""
STATE (arc-state) cell embedding extractor.

Uses ``state.emb.Inference`` + ``create_dataloader`` with in-memory ``AnnData`` from
``H5ADLoader``, matching the library's ``encode_adata`` concatenation of cell and
dataset embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from features.extractor import EmbeddingExtractor
from utils.logs_ import get_logger


class StateExtractor(EmbeddingExtractor):
    """
    Params (under ``embedding.params``):

    - ``checkpoint_path`` (str, required): PyTorch Lightning checkpoint for ``StateEmbeddingModel``.
    - ``config_path`` (str, optional): OmegaConf YAML to pass to ``Inference`` before loading
      (use when the checkpoint does not embed ``cfg_yaml`` and you must supply training config).
    - ``batch_size`` (int, optional): Overrides ``cfg.model.batch_size`` for inference.
    - ``num_workers`` (int, default 1): DataLoader workers. Kept ``>= 1`` because ``arc-state``
      sets ``persistent_workers=True`` (PyTorch requires ``num_workers > 0``).
    - ``dataset_name`` (str, default ``"scfm_eval"``): Name passed as ``adata_name`` to the dataloader.
    - ``gene_column`` (str | None, default ``"auto"``): ``"auto"`` uses STATE's overlap-based
      detection; otherwise a column name in ``adata.var`` (e.g. ``"gene_name"``), or ``None`` for index.
    - ``concat_dataset_embedding`` (bool, default True): If True, concatenate dataset embedding
      to cell embedding along the last axis (same as ``Inference.encode_adata``).
    - ``protein_embeddings_path`` (str, optional): ``gene_symbol_to_embedding`` ``.pt`` file
      (Hugging Face ``arcinstitute/SE-600M`` ships ``protein_embeddings.pt`` next to the checkpoint).
      If omitted, ``<checkpoint_dir>/protein_embeddings.pt`` is used when that file exists.
      Required when the embedded config still points at machine-specific paths such as
      ``/data/auxillary/...`` that do not exist locally.
    """

    MODELS_PATH_KEYS = frozenset(
        {"checkpoint_path", "config_path", "protein_embeddings_path"}
    )

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.log = get_logger()
        self.log.info("StateExtractor params: %s", self.params)

    @staticmethod
    def _convert_to_csr(adata):
        from scipy.sparse import csr_matrix, issparse

        if issparse(adata.X) and not isinstance(adata.X, csr_matrix):
            adata.X = csr_matrix(adata.X)
        return adata

    def _resolve_protein_embeddings(self, checkpoint: str) -> Any | None:
        """Load ESM2 gene embeddings so ``Inference.load_model`` skips cfg ``/data/...`` paths."""
        import torch

        explicit = (self.params.get("protein_embeddings_path") or "").strip()
        if explicit:
            p = Path(explicit).expanduser().resolve()
            if not p.is_file():
                raise FileNotFoundError(
                    f"STATE protein_embeddings_path does not exist: {explicit!r} "
                    "(download arcinstitute/SE-600M and point to protein_embeddings.pt, "
                    "or place that file next to your .ckpt)."
                )
            self.log.info("STATE: loading protein embeddings from %s", p)
            return torch.load(p, map_location="cpu", weights_only=False)

        candidate = Path(checkpoint).resolve().parent / "protein_embeddings.pt"
        if candidate.is_file():
            self.log.info("STATE: using protein embeddings next to checkpoint: %s", candidate)
            return torch.load(candidate, map_location="cpu", weights_only=False)

        self.log.warning(
            "STATE: no protein_embeddings.pt beside checkpoint (%s). "
            "If load fails with a missing /data/... path, download arcinstitute/SE-600M "
            "or set embedding.params.protein_embeddings_path.",
            candidate.parent,
        )
        return None

    def fit_transform(self, data_loader: Any) -> np.ndarray:
        from state.emb import Inference
        from state.emb.data import create_dataloader
        from state.emb.utils import get_precision_config

        import torch

        checkpoint = self.params["checkpoint_path"]
        config_path: Optional[str] = self.params.get("config_path")
        batch_size = self.params.get("batch_size")
        num_workers = int(self.params.get("num_workers", 1))
        dataset_name = self.params.get("dataset_name", "scfm_eval")
        gene_column_param = self.params.get("gene_column", "auto")
        concat_ds = bool(self.params.get("concat_dataset_embedding", True))

        cfg = OmegaConf.load(config_path) if config_path else None
        protein_embeds = self._resolve_protein_embeddings(checkpoint)
        inf = Inference(cfg=cfg, protein_embeds=protein_embeds)
        try:
            inf.load_model(checkpoint)
        except FileNotFoundError as e:
            if protein_embeds is None:
                raise FileNotFoundError(
                    "STATE could not load protein embeddings: the checkpoint config points at "
                    "paths from the original training environment (e.g. /data/auxillary/...). "
                    "Download arcinstitute/SE-600M from Hugging Face so `protein_embeddings.pt` is "
                    "beside your .ckpt, or set embedding.params.protein_embeddings_path to that file."
                ) from e
            raise

        adata = data_loader.adata
        adata = self._convert_to_csr(adata)

        dataloader_cfg = inf._vci_conf
        if batch_size is not None:
            try:
                dataloader_cfg = OmegaConf.create(OmegaConf.to_container(dataloader_cfg, resolve=True))
                if not hasattr(dataloader_cfg, "model"):
                    dataloader_cfg["model"] = {}
                dataloader_cfg.model.batch_size = int(batch_size)
            except Exception as e:
                self.log.warning("Could not override batch_size: %s", e)

        if gene_column_param == "auto":
            gene_column = inf._auto_detect_gene_column(adata)
        else:
            gene_column = gene_column_param

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        precision = get_precision_config(device_type=device_type)

        workers = max(1, num_workers)

        dl = create_dataloader(
            dataloader_cfg,
            workers=workers,
            adata=adata,
            adata_name=dataset_name,
            shuffle=False,
            protein_embeds=inf.protein_embeds,
            precision=precision,
            gene_column=gene_column,
        )

        all_cell: list[np.ndarray] = []
        all_ds: list[np.ndarray] = []
        for embeddings, ds_embeddings in tqdm(
            inf.encode(dl), total=len(dl), desc="STATE encode"
        ):
            all_cell.append(embeddings.astype(np.float32))
            if ds_embeddings is not None:
                all_ds.append(ds_embeddings.astype(np.float32))

        out = np.concatenate(all_cell, axis=0)
        if concat_ds and len(all_ds) > 0:
            ds_cat = np.concatenate(all_ds, axis=0)
            out = np.concatenate([out, ds_cat], axis=-1)

        out_key = self.get_output_key()
        data_loader.adata.obsm[out_key] = out
        return out
