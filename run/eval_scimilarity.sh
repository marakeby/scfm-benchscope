# scSimilarity — pixi env: scimilarity
# Usage: bash run/eval_scimilarity.sh (from repo root) or: cd run && bash eval_scimilarity.sh
cd "$(dirname "$0")"
pixi run -e scimilarity python run_exp.py exp/scimilarity/v1.1/brca_subtype.yaml
pixi run -e scimilarity python run_exp.py exp/scimilarity/v1.1/brca_cell_type.yaml
pixi run -e scimilarity python run_exp.py exp/scimilarity/v1.1/brca_chemo.yaml
pixi run -e scimilarity python run_exp.py exp/scimilarity/v1.1/brca_outcome.yaml
pixi run -e scimilarity python run_exp.py exp/scimilarity/v1.1/brca_pre_post.yaml
pixi run -e scimilarity python run_exp.py exp/scimilarity/v1.1/luad_tki.yaml
pixi run -e scimilarity python run_exp.py exp/scimilarity/v1.1/luad_cancer_stage.yaml
pixi run -e scimilarity python run_exp.py exp/scimilarity/v1.1/crc_mmr.yaml
pixi run -e scimilarity python run_exp.py exp/scimilarity/v1.1/melanoma_response.yaml