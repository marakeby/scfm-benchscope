# Geneformer V2 104M fine-tune (classification head) — pixi env: geneformer
# Usage: bash run/eval_geneformer_V2-104M-i4096_finetune.sh (from repo root)
cd "$(dirname "$0")"
pixi run -e geneformer python run_exp.py exp/geneformer/V2-104M-i4096_finetune/brca_subtype.yaml
pixi run -e geneformer python run_exp.py exp/geneformer/V2-104M-i4096_finetune/brca_cell_type.yaml
pixi run -e geneformer python run_exp.py exp/geneformer/V2-104M-i4096_finetune/brca_chemo.yaml
pixi run -e geneformer python run_exp.py exp/geneformer/V2-104M-i4096_finetune/brca_outcome.yaml
pixi run -e geneformer python run_exp.py exp/geneformer/V2-104M-i4096_finetune/brca_pre_post.yaml
pixi run -e geneformer python run_exp.py exp/geneformer/V2-104M-i4096_finetune/luad_tki.yaml
pixi run -e geneformer python run_exp.py exp/geneformer/V2-104M-i4096_finetune/luad_cancer_stage.yaml
