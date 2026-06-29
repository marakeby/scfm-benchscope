# PCA (n_components=100) — default pixi env
# Usage: bash run/eval_pca.sh   (from repo root) or: cd run && bash eval_pca.sh
cd "$(dirname "$0")/.."

#embedding tasks --------------------
pixi run python -m scfm_cancer_eval.run.run_exp exp/pca/n100/brca_cell_type.yaml 

#classification tasks --------------------
#BRCA
pixi run python -m scfm_cancer_eval.run.run_exp exp/pca/n100/brca_subtype.yaml 
pixi run python -m scfm_cancer_eval.run.run_exp exp/pca/n100/brca_chemo.yaml 
pixi run python -m scfm_cancer_eval.run.run_exp exp/pca/n100/brca_outcome.yaml 
pixi run python -m scfm_cancer_eval.run.run_exp exp/pca/n100/brca_pre_post.yaml 
#LUAD

pixi run python -m scfm_cancer_eval.run.run_exp exp/pca/n100/luad_tki.yaml 
pixi run python -m scfm_cancer_eval.run.run_exp exp/pca/n100/luad_cancer_stage.yaml 
#CRC
pixi run python -m scfm_cancer_eval.run.run_exp exp/pca/n100/crc_mmr.yaml   
#Melanoma
pixi run python -m scfm_cancer_eval.run.run_exp exp/pca/n100/melanoma_response.yaml 