# scSimilarity — pixi env: scimilarity
# Usage: bash run/eval_scimilarity.sh (from repo root) or: cd run && bash eval_scimilarity.sh
cd "$(dirname "$0")/.."

#Embedding tasks --------------------
pixi run -e scimilarity python -m scfm_cancer_eval.run.run_exp exp/scimilarity/v1.1/brca_cell_type.yaml 

#Classification tasks --------------------  
#BRCA
pixi run -e scimilarity python -m scfm_cancer_eval.run.run_exp exp/scimilarity/v1.1/brca_subtype.yaml 
pixi run -e scimilarity python -m scfm_cancer_eval.run.run_exp exp/scimilarity/v1.1/brca_chemo.yaml 
pixi run -e scimilarity python -m scfm_cancer_eval.run.run_exp exp/scimilarity/v1.1/brca_outcome.yaml 
pixi run -e scimilarity python -m scfm_cancer_eval.run.run_exp exp/scimilarity/v1.1/brca_pre_post.yaml 
#LUAD
pixi run -e scimilarity python -m scfm_cancer_eval.run.run_exp exp/scimilarity/v1.1/luad_tki.yaml 
#CRC
pixi run -e scimilarity python -m scfm_cancer_eval.run.run_exp exp/scimilarity/v1.1/crc_mmr.yaml 
#Melanoma
pixi run -e scimilarity python -m scfm_cancer_eval.run.run_exp exp/scimilarity/v1.1/melanoma_response.yaml

