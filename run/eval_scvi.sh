# scVI — pixi env: scvi
# Usage: bash run/eval_scvi.sh (from repo root) or: cd run && bash eval_scvi.sh
cd "$(dirname "$0")"
#embedding tasks --------------------

pixi run -e scvi python run_exp.py exp/scvi/default/brca_cell_type.yaml
# classification tasks --------------------
# BRCA
pixi run -e scvi python run_exp.py exp/scvi/default/brca_subtype.yaml
pixi run -e scvi python run_exp.py exp/scvi/default/brca_chemo.yaml
pixi run -e scvi python run_exp.py exp/scvi/default/brca_outcome.yaml
pixi run -e scvi python run_exp.py exp/scvi/default/brca_pre_post.yaml
# LUAD
# pixi run -e scvi python run_exp.py exp/scvi/default/luad_tki.yaml
# pixi run -e scvi python run_exp.py exp/scvi/default/luad_cancer_stage.yaml
# pixi run -e scvi python run_exp.py exp/scvi/default/crc_mmr.yaml
# pixi run -e scvi python run_exp.py exp/scvi/default/melanoma_response.yaml