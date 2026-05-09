# HVG (Seurat) — default pixi env
# Usage: bash run/eval_hvg.sh (from repo root) or: cd run && bash eval_hvg.sh
cd "$(dirname "$0")"
#Embedding tasks --------------------
# BrCA cell type + Pancancer cell type 
# pixi run python run_exp.py exp/hvg/seurat_4096/brca_cell_type.yaml
# pixi run python run_exp.py exp/hvg/seurat_4096/pancancer_cell_type_counts.yaml

#Classification tasks --------------------
# pixi run python run_exp.py exp/hvg/seurat_4096/brca_subtype.yaml
pixi run python run_exp.py exp/hvg/seurat_4096/brca_chemo.yaml
# pixi run python run_exp.py exp/hvg/seurat_4096/brca_outcome.yaml
# pixi run python run_exp.py exp/hvg/seurat_4096/brca_pre_post.yaml

#LUAD
# pixi run python run_exp.py exp/hvg/seurat_4096/luad_tki.yaml

# pixi run python run_exp.py exp/hvg/seurat_4096/crc_mmr.yaml
# pixi run python run_exp.py exp/hvg/seurat_4096/melanoma_response.yaml

# pixi run python run_exp.py exp/hvg/seurat_4096/luad_cancer_stage.yaml
# pixi run python run_exp.py exp/hvg/seurat_4096/brca_cell_type_counts.yaml

                           