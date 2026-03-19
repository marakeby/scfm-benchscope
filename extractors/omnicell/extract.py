#!/usr/bin/env python
"""Omnicell (cell-types) encoder embedding extraction.

Uses the Omnicell encoder from the cell-types repo to extract cell embeddings.
Requires cell-types to be available (e.g. same env as scfm_meta) and
OMNICELL_DATA_DIR set to a dir containing protocol_embeddings/ (global_gene_mapping_v2_new.parquet,
gene_manager_serialized.pkl); the cell-types package reads these at import.
For best gene matching, run ensure_both_gene_identifiers(normalize=True) on
the loader before extraction so adata.var has gene_symbol (and gene_id).

Usage:
    python extract.py --input data.h5ad --output embeddings.npy \\
        --checkpoint_path /path/to/omnicell_checkpoint_epoch27.pt \\
        --base_dir /path/to/cell-types/cell_types
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import scanpy as sc

# Base extractor
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from base_extract import (
    BaseExtractor,
    create_argument_parser,
    normalize_adata_var_gene_names,
    run_extraction,
)

logger = logging.getLogger(__name__)


class OmnicellExtractor(BaseExtractor):
    """Omnicell (cell-types) encoder extractor.

    Loads the Omnicell encoder + generator from a checkpoint, maps input genes
    to the model's vocabulary (global gene list + used_genes), and returns
    per-cell encoder embeddings.

    **Required input alignment**
    - Gene identifiers are taken from ``adata.var_names`` for mapping to the
      cell-types global gene list (parquet with ``feature_id`` and ``feature_name``).
    - ``gene_types``: ``"feature_name"`` = gene symbols (e.g. HGNC);
      ``"feature_id"`` = Ensembl IDs. Must match the global list column.
    - If ``gene_types == "feature_name"`` and ``adata.var`` has ``gene_symbol``,
      the extractor uses ``gene_symbol`` as the mapping source (recommended:
      run ``ensure_both_gene_identifiers(normalize=True)`` before extraction).
    - If ``gene_types == "feature_id"`` and ``adata.var`` has ``gene_id``,
      the extractor uses ``gene_id`` for mapping.
    - Gene names like ``"SYMBOL (ENTREZ)"`` are normalized to ``"SYMBOL"``
      before lookup so they match the global list.
    - ``OMNICELL_BASE_DIR``: path to the cell-types repo root (e.g. ``.../cell-types``)
      or the package dir (``.../cell-types/cell_types``). Either works.
    - ``OMNICELL_DATA_DIR`` (required at runtime): path to the dir that contains
      ``protocol_embeddings/`` with ``global_gene_mapping_v2_new.parquet`` and
      ``gene_manager_serialized.pkl``. Set in the environment.
    - ``OMNICELL_PYTHON`` (optional): path to the Python executable to use for the
      embedding subprocess (e.g. your cell-types–specific env). If unset, the
      current process Python is used.

    The extractor runs the cell-types model in a **subprocess** with the correct
    PYTHONPATH and cwd so that cell-types' internal imports work without modifying
    sys.path in the main process or adding __init__.py to the cell-types repo.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        config: str = "obs",
        base_dir: str | None = None,
        batch_size: int = 4096,
        device: str = "auto",
        gene_types: str = "feature_name",
        load_avg: bool = False,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Registry calls extractor_class(params); first positional is then the config dict
        if isinstance(checkpoint_path, dict):
            params = checkpoint_path
            checkpoint_path = None
        if params is not None:
            super().__init__(params=params)
            checkpoint_path = self.params.get("checkpoint_path", checkpoint_path)
            config = self.params.get("config", config)
            base_dir = self.params.get("base_dir", base_dir)
            batch_size = self.params.get("batch_size", batch_size)
            device = self.params.get("device", device)
            gene_types = self.params.get("gene_types", gene_types)
            load_avg = self.params.get("load_avg", load_avg)
        else:
            super().__init__(
                checkpoint_path=checkpoint_path,
                config=config,
                base_dir=base_dir,
                batch_size=batch_size,
                device=device,
                gene_types=gene_types,
                load_avg=load_avg,
                **kwargs,
            )

        self.checkpoint_path = checkpoint_path
        self.config = config
        self.base_dir = Path(base_dir) if base_dir else None
        self.batch_size = batch_size
        self.device = device
        self.gene_types = gene_types
        self.load_avg = load_avg

        if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(
                f"Omnicell checkpoint not found: {self.checkpoint_path}"
            )
        if not self.base_dir or not self.base_dir.is_dir():
            raise FileNotFoundError(
                f"Omnicell base_dir (cell_types package) not found: {self.base_dir}"
            )

        logger.info(
            "Omnicell extractor: config=%s, checkpoint=%s, base_dir=%s",
            self.config,
            self.checkpoint_path,
            self.base_dir,
        )

    @property
    def model_name(self) -> str:
        return "omnicell"

    @property
    def embedding_dim(self) -> int:
        return -1  # Known only after extraction (encoder latent_dim)

    def load_model(self) -> None:
        """No-op: model is loaded in the subprocess."""
        pass

    def _repo_paths(self) -> tuple[Path, Path]:
        """Return (repo_root, package_dir) for cell-types."""
        base = self.base_dir.resolve()
        if (base / "cell_types").is_dir():
            return base, base / "cell_types"
        if base.name == "cell_types":
            return base.parent, base
        return base.parent, base

    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        repo_root, package_dir = self._repo_paths()
        # Prepare adata in this process (gene alignment + normalization)
        if hasattr(adata, "filename") and adata.isbacked:
            adata = adata.to_memory()
        adata = adata.copy()
        if self.gene_types == "feature_name" and "gene_symbol" in adata.var.columns:
            adata.var.index = adata.var["gene_symbol"].astype(str).values
        elif self.gene_types == "feature_id" and "gene_id" in adata.var.columns:
            adata.var.index = adata.var["gene_id"].astype(str).values
        normalize_adata_var_gene_names(adata)

        helper = Path(__file__).resolve().parent / "run_omnicell_embed.py"
        env = os.environ.copy()
        bulk_dir = package_dir / "bulk_prediction"
        path_parts = [str(repo_root), str(package_dir)]
        if bulk_dir.is_dir():
            path_parts.append(str(bulk_dir))  # so hydra_utils (bulk_prediction/hydra_utils) is found
        env["PYTHONPATH"] = os.pathsep.join(path_parts + [env.get("PYTHONPATH", "")])
        if "OMNICELL_DATA_DIR" not in env:
            logger.warning("OMNICELL_DATA_DIR not set; cell-types may fail to load gene manager")

        with tempfile.TemporaryDirectory(prefix="omnicell_embed_") as tmp:
            input_h5ad = Path(tmp) / "adata.h5ad"
            output_npy = Path(tmp) / "embeddings.npy"
            adata.write_h5ad(input_h5ad)

            # Allow overriding the Python used for the subprocess so we can
            # point explicitly to the Omnicell / cell-types env (e.g. MORPH).
            # Falls back to the current interpreter if OMNICELL_PYTHON is unset.
            python_exec = env.get("OMNICELL_PYTHON") or sys.executable

            cmd = [
                python_exec,
                str(helper),
                "--input",
                str(input_h5ad),
                "--output",
                str(output_npy),
                "--checkpoint_path",
                str(self.checkpoint_path),
                "--base_dir",
                str(package_dir),
                "--config",
                self.config,
                "--gene_types",
                self.gene_types,
                "--batch_size",
                str(self.batch_size),
            ]
            if self.load_avg:
                cmd.append("--load_avg")
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(repo_root))
            if result.returncode != 0:
                logger.error("Omnicell subprocess stderr: %s", result.stderr)
                raise RuntimeError(f"Omnicell embedding subprocess failed with code {result.returncode}")
            embeddings = np.load(output_npy)
        logger.info("Omnicell embeddings shape: %s, dtype %s", embeddings.shape, embeddings.dtype)
        return embeddings.astype(np.float32)


def main() -> None:
    parser = create_argument_parser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to Omnicell checkpoint .pt file",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Path to cell_types package directory (cell-types/cell_types)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="obs",
        help="Cell-types config name (default: obs)",
    )
    parser.add_argument(
        "--gene_types",
        type=str,
        default="feature_name",
        choices=["feature_name", "feature_id"],
        help="Gene identifier type in adata.var_names (default: feature_name)",
    )
    parser.add_argument(
        "--load_avg",
        action="store_true",
        help="Load averaged weights (only if checkpoint has avg_encoder_state_dict)",
    )
    args = parser.parse_args()

    extractor = OmnicellExtractor(
        checkpoint_path=args.checkpoint_path,
        base_dir=args.base_dir,
        config=args.config,
        batch_size=args.batch_size,
        device=args.device,
        gene_types=args.gene_types,
        load_avg=args.load_avg,
    )
    run_extraction(
        extractor,
        args.input,
        args.output,
        save_metadata=args.save_metadata,
    )


if __name__ == "__main__":
    main()
