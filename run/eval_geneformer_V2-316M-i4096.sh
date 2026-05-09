# Geneformer V2 316M (4096) — pixi env: geneformer
# Usage: bash run/eval_geneformer_V2-316M-i4096.sh (from repo root)
cd "$(dirname "$0")"
#Embedding tasks --------------------
pixi run -e geneformer python run_exp.py exp/geneformer/V2-316M-i4096/brca_cell_type_full.yaml

#Classification tasks --------------------
# pixi run -e geneformer python run_exp.py exp/geneformer/V2-316M-i4096/brca_subtype.yaml
# pixi run -e geneformer python run_exp.py exp/geneformer/V2-316M-i4096/brca_chemo.yaml
# pixi run -e geneformer python run_exp.py exp/geneformer/V2-316M-i4096/brca_outcome.yaml
# pixi run -e geneformer python run_exp.py exp/geneformer/V2-316M-i4096/brca_pre_post.yaml

# pixi run -e geneformer python run_exp.py exp/geneformer/V2-316M-i4096/luad_tki.yaml
# pixi run -e geneformer python run_exp.py exp/geneformer/V2-316M-i4096/luad_cancer_stage.yaml

# pixi run -e geneformer python run_exp.py exp/geneformer/V2-316M-i4096/crc_mmr.yaml
# pixi run -e geneformer python run_exp.py exp/geneformer/V2-316M-i4096/melanoma_response.yaml