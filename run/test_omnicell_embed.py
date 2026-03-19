#!/usr/bin/env python
"""One-off test: load a dataset from an Omnicell YAML and embed with Omnicell.

Run from repo root with env set (DATASETS_PATH, MODEL_CHECKPOINTS or OMNICELL_CHECKPOINTS_BASE, OMNICELL_BASE_DIR, etc.):
    cd /path/to/scFM_eval && python run/test_omnicell_embed.py
    cd /path/to/scFM_eval && python run/test_omnicell_embed.py --config yaml/batch_bio_integration/omnicell/donor_id/gtex_v9_donor.yaml
"""

from __future__ import annotations

import argparse
import importlib 
import logging
import os
import sys
from pathlib import Path

# Repo root on path (same as run_exp)
_dir = Path(__file__).resolve().parent.parent
if str(_dir) not in sys.path:
    sys.path.insert(0, str(_dir))

from run.utils import get_configs
from setup_path import PARAMS_PATH
from utils.data_state import DataState, ensure_state, get_data_state, get_state_summary

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Omnicell embedding on one dataset from YAML")
    parser.add_argument(
        "--config",
        type=str,
        default="batch_bio_integration/omnicell/combined/harmony_figure5_combined.yaml",
        help="Config path relative to PARAMS_PATH (yaml/...) or absolute",
    )
    args = parser.parse_args()

    # Resolve config path
    if os.path.isabs(args.config):
        config_path = args.config
    else:
        rel = args.config.replace("\\", os.sep).lstrip(os.sep)
        if rel.startswith("yaml" + os.sep):
            rel = rel[len("yaml") + 1 :].lstrip(os.sep)
        config_path = os.path.join(PARAMS_PATH, rel)
    if not os.path.isfile(config_path):
        logger.error("Config not found: %s", config_path)
        return 1

    # Load config
    run_id, data_config, qc_config, preproc_config, hvg_config, feat_config, _ = get_configs(
        config_path
    )
    if feat_config.get("method") != "omnicell":
        logger.warning("Config embedding.method is %s; expected omnicell", feat_config.get("method"))

    # Loader
    loader_module = data_config["module"]
    loader_class_name = data_config["class"]
    LoaderClass = getattr(importlib.import_module(loader_module), loader_class_name)
    loader = LoaderClass(data_config)
    logger.info("Loading data from %s", getattr(loader, "path", "?"))
    loader.load()
    adata = loader.adata
    logger.info("Loaded adata shape: %s", adata.shape)

    # Gene IDs (required for Omnicell mapping)
    loader.ensure_both_gene_identifiers(normalize=True)
    logger.info("ensure_both_gene_identifiers done; gene_symbol in var: %s", "gene_symbol" in adata.var.columns)

    # Data state: Omnicell expects LOG1P
    current = get_data_state(adata)
    target_state = DataState.LOG1P
    target_sum = preproc_config.get("target_sum", 10000)
    if current != target_state:
        logger.info("Ensuring data state %s (current: %s)", target_state.value, current.value)
        ensure_state(adata, target_state, target_sum=target_sum)
    else:
        logger.info("Data already in %s", get_state_summary(adata))

    # Extract embeddings
    from extractors import get_extractor

    params = feat_config.get("params", {}).copy()
    extractor = get_extractor("omnicell", params)
    embeddings = extractor.fit_transform(loader)
    logger.info("Embeddings shape: %s, dtype: %s", embeddings.shape, embeddings.dtype)

    # Sanity checks
    assert embeddings.shape[0] == adata.n_obs, (
        f"Embedding rows {embeddings.shape[0]} != adata.n_obs {adata.n_obs}"
    )
    assert embeddings.shape[1] > 0, "Embedding dim should be > 0"
    logger.info("Test passed: Omnicell embedded %d cells -> dim %d", embeddings.shape[0], embeddings.shape[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
