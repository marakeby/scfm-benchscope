# Geneformer V2 104M fine-tune (classification head) — pixi env: geneformer
# Usage: bash run/eval_geneformer_V2-104M-i4096_finetune.sh (from repo root)
cd "$(dirname "$0")/.." 


#Embedding tasks --------------------
# pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V2-104M-i4096_finetune/brca_cell_type.yaml

#Classification tasks --------------------
#BRCA
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V2-104M-i4096_finetune/brca_subtype.yaml 
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V2-104M-i4096_finetune/brca_chemo.yaml 
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V2-104M-i4096_finetune/brca_outcome.yaml 
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V2-104M-i4096_finetune/brca_pre_post.yaml 
#LUAD
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V2-104M-i4096_finetune/luad_tki.yaml 
#CRC
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V2-104M-i4096_finetune/crc_mmr.yaml
#Melanoma
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V2-104M-i4096_finetune/melanoma_response.yaml 