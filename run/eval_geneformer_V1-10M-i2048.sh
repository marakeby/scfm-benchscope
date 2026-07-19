# Geneformer V1 10M (2048) — pixi env: geneformer
# Usage: bash run/eval_geneformer_V1-10M-i2048.sh (from repo root)

cd "$(dirname "$0")/.."


#Embedding tasks --------------------
pixi run -e geneformer scfm-eval exp/geneformer/V1-10M-i2048/brca_cell_type.yaml 
pixi run -e geneformer scfm-eval exp/geneformer/V1-10M-i2048/brca_cell_type_continue.yaml 

#Classification tasks --------------------
#BRCA   
pixi run -e geneformer scfm-eval exp/geneformer/V1-10M-i2048/brca_subtype.yaml 
pixi run -e geneformer scfm-eval exp/geneformer/V1-10M-i2048/brca_chemo.yaml 
pixi run -e geneformer scfm-eval exp/geneformer/V1-10M-i2048/brca_outcome.yaml 
pixi run -e geneformer scfm-eval exp/geneformer/V1-10M-i2048/brca_pre_post.yaml 


#LUAD
pixi run -e geneformer scfm-eval exp/geneformer/V1-10M-i2048/luad_tki.yaml 

#CRC
pixi run -e geneformer scfm-eval exp/geneformer/V1-10M-i2048/crc_mmr.yaml 

#Melanoma   
pixi run -e geneformer scfm-eval exp/geneformer/V1-10M-i2048/melanoma_response.yaml 

#counts
# pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V1-10M-i2048/brca_subtype_counts.yaml 
# pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V1-10M-i2048/brca_pre_post_counts.yaml 
# pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V1-10M-i2048/brca_chemo.yaml 
# pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V1-10M-i2048/brca_outcome.yaml 