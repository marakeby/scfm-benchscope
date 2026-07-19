# PCA (n_components=100) — default pixi env
# Usage: bash run/eval_pca.sh   (from repo root) or: cd run && bash eval_pca.sh
cd "$(dirname "$0")/.."


#embedding tasks --------------------
pixi run scfm-eval exp/pca/n50/brca_cell_type.yaml 

#classification tasks --------------------
#BRCA
pixi run scfm-eval exp/pca/n50/brca_subtype.yaml 
pixi run scfm-eval exp/pca/n50/brca_chemo.yaml 
pixi run scfm-eval exp/pca/n50/brca_outcome.yaml 
pixi run scfm-eval exp/pca/n50/brca_pre_post.yaml 
#LUAD

pixi run scfm-eval exp/pca/n50/luad_tki.yaml 
#CRC
pixi run scfm-eval exp/pca/n50/crc_mmr.yaml    
#Melanoma
pixi run scfm-eval exp/pca/n50/melanoma_response.yaml