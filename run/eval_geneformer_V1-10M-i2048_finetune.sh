# Geneformer V1 10M fine-tune (classification head) — pixi env: geneformer
# Usage: bash run/eval_geneformer_V1-10M-i2048_finetune.sh (from repo root)
cd "$(dirname "$0")/.."

#Classification tasks --------------------
#BRCA
# pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V1-10M-i2048_finetune/brca_subtype.yaml
# pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V1-10M-i2048_finetune/brca_chemo.yaml
# pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V1-10M-i2048_finetune/brca_outcome.yaml
# pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V1-10M-i2048_finetune/brca_pre_post.yaml
#LUAD
# pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V1-10M-i2048_finetune/luad_tki.yaml
#CRC
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V1-10M-i2048_finetune/crc_mmr.yaml
#Melanoma
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V1-10M-i2048_finetune/melanoma_response.yaml
