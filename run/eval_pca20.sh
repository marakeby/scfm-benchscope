# PCA (n_components=20) — default pixi env
# Usage: bash run/eval_pca20.sh   (from repo root) or: cd run && bash eval_pca20.sh
cd "$(dirname "$0")/.."

#embedding tasks --------------------
pixi run -- scfm-eval exp/pca/n20/brca_cell_type.yaml

#classification tasks --------------------
#BRCA
pixi run -- scfm-eval exp/pca/n20/brca_subtype.yaml
pixi run -- scfm-eval exp/pca/n20/brca_chemo.yaml
pixi run -- scfm-eval exp/pca/n20/brca_outcome.yaml
pixi run -- scfm-eval exp/pca/n20/brca_pre_post.yaml

#LUAD
pixi run -- scfm-eval exp/pca/n20/luad_tki.yaml
#CRC
pixi run -- scfm-eval exp/pca/n20/crc_mmr.yaml
#Melanoma
pixi run -- scfm-eval exp/pca/n20/melanoma_response.yaml
