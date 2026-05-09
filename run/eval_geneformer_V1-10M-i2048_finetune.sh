# Geneformer V1 10M fine-tune (classification head) — pixi env: geneformer
# Usage: bash run/eval_geneformer_V1-10M-i2048_finetune.sh (from repo root)
cd "$(dirname "$0")"
pixi run -e geneformer python run_exp.py exp/geneformer/V1-10M-i2048_finetune/brca_subtype.yaml
pixi run -e geneformer python run_exp.py exp/geneformer/V1-10M-i2048_finetune/brca_cell_type.yaml
pixi run -e geneformer python run_exp.py exp/geneformer/V1-10M-i2048_finetune/brca_chemo.yaml
pixi run -e geneformer python run_exp.py exp/geneformer/V1-10M-i2048_finetune/brca_outcome.yaml
pixi run -e geneformer python run_exp.py exp/geneformer/V1-10M-i2048_finetune/brca_pre_post.yaml
pixi run -e geneformer python run_exp.py exp/geneformer/V1-10M-i2048_finetune/luad_tki.yaml
pixi run -e geneformer python run_exp.py exp/geneformer/V1-10M-i2048_finetune/luad_cancer_stage.yaml
