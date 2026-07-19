"""Display names and model-family grouping for published metric tables."""

from __future__ import annotations

MODEL_NAME_MAP: dict[str, str] = {
    "hvg": "HVG",
    "pca": "PCA",
    "scgpt": "scGPT",
    "scgpt_cancer": "scGPT [cancer]",
    "scvi": "scVI",
    "scvi_donor_id": "scVI",
    "scfoundation": "scFoundation",
    "scimilarity": "SCimiarity",
    "cellplm": "CellPLM",
    "gf-6L-30M-i2048": "GF-V1",
    "gf-6L-30M-i2048_continue": "GF-V1 [continue]",
    "continue_geneformer_V1-10M-i2048_continue": "GF-V1 [continue]",
    "Geneformer-V2-104M_CLcancer": "GF-V2 [cancer]",
    "Geneformer-V2-104M": "GF-V2",
    "full_geneformer_V2-104M-i4096": "GF-V2 [Full]",
    "Geneformer-V2-104M_continue": "GF-V2 [continue]",
    "continue_geneformer_V2-104M-i4096_continue": "GF-V2 [continue]",
    "Geneformer-V2-316M": "GF-V2-Deep",
    "gf-6L-30M-i2048_finetune": "GF-V1 [finetune]",
    "Geneformer-V2-104M_finetune": "GF-V2 [finetune]",
    "geneformer_V2-104M-i4096_finetune": "GF-V2 [finetune]",
    "geneformer_V1-10M-i2048_finetune": "GF-V1 [finetune]",
    "hvg_seurat_4096": "HVG",
    "state_se600m_epoch16": "STATE",
    "scfoundation_brca_cancer_cells": "scFoundation",
    "geneformer_V2-104M_CLcancer-i4096": "GF-V2 [cancer]",
    "geneformer_V2-316M-i4096": "GF-V2-Deep",
    "geneformer_V1-10M-i2048_continue_brca_cell_type": "GF-V1 [continue]",
    "geneformer_V1-10M-i2048_continue": "GF-V1 [continue]",
    "geneformer_V1-10M-i2048": "GF-V1",
    "geneformer_V2-104M-i4096_continue_brca_cell_type": "GF-V2 [continue]",
    "geneformer_V2-104M-i4096_continue": "GF-V2 [continue]",
    "geneformer_V2-104M-i4096": "GF-V2",
    "scgpt_cancer-i2048": "scGPT [cancer]",
    "scgpt_human-i2048": "scGPT",
    "cellplm_85M-20231027": "CellPLM",
    "scimilarity_v1.1": "SCimiarity",
    "pca_n100": "PCA [100]",
    "pca_n50": "PCA [50]",
    "pca_n20": "PCA [20]",
    "scconcept_corpus30m": "scConcept",
    "nicheformer_nicheformer": "Nicheformer",
}

EXPERIMENT_NAME_MAP: dict[str, str] = {
    "pre_post": "Treatment Naive vs Anti PD1",
    "brca_full_pre_post": "Treatment Naive vs Anti PD1",
    "brca_pre_post": "Treatment Naive vs Anti PD1",
    "chemo": "Treatment Naive vs Neoadjuvant Chemo",
    "brca_full_chemo": "Treatment Naive vs Neoadjuvant Chemo",
    "brca_chemo": "Treatment Naive vs Neoadjuvant Chemo",
    "luad2": "Treatment Naive vs TKI treated",
    "luad_tki": "Treatment Naive vs TKI treated",
    "luad1": "Early stage vs Late stage",
    "outcome": "T-cell exhaustion",
    "brca_full_outcome": "T-cell exhaustion",
    "brca_outcome": "T-cell exhaustion",
    "subtype": "ER+ vs TNBC",
    "brca_full_subtype": "ER+ vs TNBC",
    "brca_subtype": "ER+ vs TNBC",
    "brca_cell_type": "BRCA Cell Type",
    "brca_cell_type_continue": "BRCA Cell Type",
    "melanoma_response": "IO Response",
    "crc_mmr": "MMRd vs MMRp",
}


def map_groups(model_id: str) -> str:
    """Broad model family used by the classification/embedding dashboards."""
    exp = model_id.lower()
    if "gf" in exp or "geneformer" in exp:
        return "Geneformer"
    if "scgpt" in exp:
        return "scGPT"
    if any(x in exp for x in ("hvg", "pca", "scvi")):
        return "Baseline"
    return "Other"
