# scFoundation — pixi env: scf (install Biomap stack per project README)
# Usage: bash run/eval_scfoundation.sh (from repo root) or: cd run && bash eval_scfoundation.sh
cd "$(dirname "$0")/.."


#Embedding tasks --------------------
pixi run -e scf python -m scfm_cancer_eval.run.run_exp exp/scfoundation/brca_cell_type.yaml 

#Classification tasks --------------------      
#BRCA
pixi run -e scf python -m scfm_cancer_eval.run.run_exp exp/scfoundation/brca_subtype.yaml 
pixi run -e scf python -m scfm_cancer_eval.run.run_exp exp/scfoundation/brca_chemo.yaml 
pixi run -e scf python -m scfm_cancer_eval.run.run_exp exp/scfoundation/brca_outcome.yaml 
pixi run -e scf python -m scfm_cancer_eval.run.run_exp exp/scfoundation/brca_pre_post.yaml 
# LUAD
pixi run -e scf python -m scfm_cancer_eval.run.run_exp exp/scfoundation/luad_tki.yaml
# CRC
pixi run -e scf python -m scfm_cancer_eval.run.run_exp exp/scfoundation/crc_mmr.yaml 
#Melanoma
pixi run -e scf python -m scfm_cancer_eval.run.run_exp exp/scfoundation/melanoma_response.yaml 