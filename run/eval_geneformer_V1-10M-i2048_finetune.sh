# Geneformer V1 10M fine-tune (40G-tuned MIL) — pixi env: geneformer
# Usage: bash run/eval_geneformer_V1-10M-i2048_finetune.sh (from repo root)
cd "$(dirname "$0")/.."

# Classification tasks --------------------
# BRCA
pixi run -e geneformer scfm-eval exp/geneformer/V1-10M-i2048_finetune_40g/brca_subtype.yaml
pixi run -e geneformer scfm-eval exp/geneformer/V1-10M-i2048_finetune_40g/brca_chemo.yaml
pixi run -e geneformer scfm-eval exp/geneformer/V1-10M-i2048_finetune_40g/brca_outcome.yaml
pixi run -e geneformer scfm-eval exp/geneformer/V1-10M-i2048_finetune_40g/brca_pre_post.yaml
# LUAD
pixi run -e geneformer scfm-eval exp/geneformer/V1-10M-i2048_finetune_40g/luad_tki.yaml

# CRC
pixi run -e geneformer scfm-eval exp/geneformer/V1-10M-i2048_finetune_40g/crc_mmr.yaml
# Melanoma
pixi run -e geneformer scfm-eval exp/geneformer/V1-10M-i2048_finetune_40g/melanoma_response.yaml

# LUAD
# pixi run -e geneformer scfm-eval exp/geneformer/V1-10M-i2048_finetune_40g/luad_cancer_stage.yaml