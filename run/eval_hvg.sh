# HVG (Seurat) — default pixi env
# Usage: bash run/eval_hvg.sh (from repo root) or: cd run && bash eval_hvg.sh
cd "$(dirname "$0")/.."
#Embedding tasks --------------------
pixi run python -m scfm-eval exp/hvg/seurat_4096/brca_cell_type.yaml

#Classification tasks --------------------
pixi run scfm-eval exp/hvg/seurat_4096/brca_subtype.yaml
pixi run scfm-eval exp/hvg/seurat_4096/brca_chemo.yaml
pixi run scfm-eval exp/hvg/seurat_4096/brca_outcome.yaml
pixi run scfm-eval exp/hvg/seurat_4096/brca_pre_post.yaml

#LUAD
pixi run scfm-eval exp/hvg/seurat_4096/luad_tki.yaml
#CRC
pixi run scfm-eval exp/hvg/seurat_4096/crc_mmr.yaml
#Melanoma
pixi run scfm-eval exp/hvg/seurat_4096/melanoma_response.yaml



                           