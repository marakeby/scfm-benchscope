# PCA (n_components=100) — default pixi env
# Usage: bash run/eval_pca.sh   (from repo root) or: cd run && bash eval_pca.sh
cd "$(dirname "$0")"
#embedding tasks --------------------
pixi run python run_exp.py exp/pca/n100/brca_cell_type.yaml
#classification tasks --------------------
#BRCA
pixi run python run_exp.py exp/pca/n100/brca_subtype.yaml
pixi run python run_exp.py exp/pca/n100/brca_chemo.yaml
pixi run python run_exp.py exp/pca/n100/brca_outcome.yaml
pixi run python run_exp.py exp/pca/n100/brca_pre_post.yaml
#LUAD

# pixi run python run_exp.py exp/pca/n100/luad_tki.yaml
# pixi run python run_exp.py exp/pca/n100/luad_cancer_stage.yaml

# pixi run python run_exp.py exp/pca/n50/brca_cell_type.yaml


# pixi run python run_exp.py exp/pca/n20/brca_cell_type.yaml

# pixi run python run_exp.py exp/pca/n100/crc_mmr.yaml   
# pixi run python run_exp.py exp/pca/n100/melanoma_response.yaml