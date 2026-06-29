# Geneformer V2 104M (4096) — pixi env: geneformer
# Usage: bash run/eval_geneformer_V2-104M-i4096.sh (from repo root)
cd "$(dirname "$0")/.."
#Embedding tasks --------------------
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V2-104M-i4096/brca_cell_type.yaml
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V2-104M-i4096/brca_cell_type_continue.yaml

#Classification tasks --------------------
#BRCA
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V2-104M-i4096/brca_subtype.yaml
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V2-104M-i4096/brca_chemo.yaml
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V2-104M-i4096/brca_outcome.yaml
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V2-104M-i4096/brca_pre_post.yaml

#LUAD
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V2-104M-i4096/luad_tki.yaml
# pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V2-104M-i4096/luad_cancer_stage.yaml
#CRC
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V2-104M-i4096/crc_mmr.yaml
#Melanoma
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp exp/geneformer/V2-104M-i4096/melanoma_response.yaml

