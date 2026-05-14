# scVI — pixi env: scvi
# Usage: bash run/eval_scvi.sh (from repo root) or: cd run && bash eval_scvi.sh
cd "$(dirname "$0")/.."

#Embedding tasks --------------------
pixi run -e scvi python -m scfm_cancer_eval.run.run_exp exp/scvi/default/brca_cell_type.yaml 

#Classification tasks --------------------
# BRCA
pixi run -e scvi python -m scfm_cancer_eval.run.run_exp exp/scvi/default/brca_subtype.yaml 
pixi run -e scvi python -m scfm_cancer_eval.run.run_exp exp/scvi/default/brca_chemo.yaml 
pixi run -e scvi python -m scfm_cancer_eval.run.run_exp exp/scvi/default/brca_outcome.yaml 
pixi run -e scvi python -m scfm_cancer_eval.run.run_exp exp/scvi/default/brca_pre_post.yaml 
# LUAD
pixi run -e scvi python -m scfm_cancer_eval.run.run_exp exp/scvi/default/luad_tki.yaml 
# CRC
pixi run -e scvi python -m scfm_cancer_eval.run.run_exp exp/scvi/default/crc_mmr.yaml 
# Melanoma
pixi run -e scvi python -m scfm_cancer_eval.run.run_exp exp/scvi/default/melanoma_response.yaml