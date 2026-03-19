#!/usr/bin/env python
"""Standalone script to run Omnicell embedding in a subprocess.

Run with PYTHONPATH set so cell_types is importable, e.g.:
  PYTHONPATH=<repo_root>:<package_dir> OMNICELL_DATA_DIR=<...> \\
  python run_omnicell_embed.py --input in.h5ad --output out.npy \\
  --checkpoint_path ... --base_dir ... --config obs --gene_types feature_name --batch_size 4096

Expects adata.var to already have gene identifiers in var.index (e.g. gene_symbol)
and normalized names (no "SYMBOL (ID)"); the parent process prepares adata before writing.

Compatible with cell-types encoder API: uses OmnicellBase from tasks.core.base,
and computes per-cell embeddings via encoder(x, return_cell_embeddings=True) (encoders.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch


def _get_cell_embeddings(
    base: "OmnicellBase",
    X: np.ndarray,
    shared_gene_ids: list[int],
    batch_size: int,
) -> np.ndarray:
    """Compute per-cell encoder embeddings using encoder(x, return_cell_embeddings=True).

    Encoder expects input shape [B, S, G] with S=1 for per-cell (one cell per set).
    Returns [B, latent_dim] per-cell embeddings.
    """
    device = base.device
    encoder = base.encoder
    n_cells, _ = X.shape
    X_shared = X[:, shared_gene_ids].astype(np.float32)
    embeddings_list = []

    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        batch_x = X_shared[start:end]
        # [batch_size, G] -> [batch_size, 1, G] for encoder
        x_tensor = torch.from_numpy(batch_x).to(device).unsqueeze(1)
        with torch.no_grad():
            _lat, cell_emb_raw, _ = encoder(x_tensor, return_cell_embeddings=True)
        # cell_emb_raw: [B, 1, latent_dim] -> [B, latent_dim]
        emb = cell_emb_raw.squeeze(1).cpu().numpy()
        embeddings_list.append(emb)

    return np.concatenate(embeddings_list, axis=0).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Omnicell embedding (subprocess entrypoint)")
    parser.add_argument("--input", required=True, help="Input .h5ad path")
    parser.add_argument("--output", required=True, help="Output .npy path")
    parser.add_argument("--checkpoint_path", required=True, help="Omnicell checkpoint .pt")
    parser.add_argument("--base_dir", required=True, help="cell_types package dir (cell-types/cell_types)")
    parser.add_argument("--config", default="obs", help="Cell-types config name")
    parser.add_argument("--gene_types", default="feature_name", choices=["feature_name", "feature_id"])
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--load_avg", action="store_true")
    args = parser.parse_args()

    base_path = Path(args.base_dir).resolve()
    if (base_path / "cell_types").is_dir():
        repo_root = base_path
        package_dir = base_path / "cell_types"
    elif base_path.name == "cell_types":
        repo_root = base_path.parent
        package_dir = base_path
    else:
        repo_root = base_path.parent
        package_dir = base_path

    # Ensure cell_types and hydra_utils (under bulk_prediction) are importable
    path_add = [str(repo_root), str(package_dir)]
    if (package_dir / "bulk_prediction").is_dir():
        path_add.append(str(package_dir / "bulk_prediction"))
    for p in path_add:
        if p not in sys.path:
            sys.path.insert(0, p)

    from cell_types.tasks.core.base import OmnicellBase

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    omnicell_base = OmnicellBase(
        config=args.config,
        checkpoint=args.checkpoint_path,
        pert_config=None,
        pert_checkpoint=None,
        device=device,
        load_avg=args.load_avg,
        base_dir=str(package_dir),
    )

    adata = ad.read_h5ad(args.input)
    if hasattr(adata, "filename") and adata.isbacked:
        adata = adata.to_memory()
    adata = adata.copy()

    # used_genes: from generator (set at training load) or from checkpoint for inference
    used_genes = getattr(omnicell_base.generator.model, "used_genes", None)
    if used_genes is None:
        try:
            ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
            used_genes = ckpt.get("used_genes")
        except Exception:
            pass
    if used_genes is None:
        print("Omnicell: used_genes not on generator and not in checkpoint; map_genes will not filter by used_genes.", file=sys.stderr)

    adata_mapped = omnicell_base.map_genes(adata, args.gene_types, used_genes=used_genes)
    X = adata_mapped.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    # Per-cell embeddings via encoder(..., return_cell_embeddings=True)
    embeddings = _get_cell_embeddings(
        omnicell_base,
        X,
        omnicell_base.shared_gene_ids,
        args.batch_size,
    )
    np.save(args.output, embeddings.astype(np.float32))
    return 0


if __name__ == "__main__":
    sys.exit(main())
