# CellPLM — pixi env: cellplm
# Usage: bash run/eval_cellplm.sh (from repo root) or: cd run && bash eval_cellplm.sh
cd "$(dirname "$0")"
# pixi run -e cellplm python run_exp.py exp/cellplm/85M-20231027/brca_subtype.yaml
# pixi run -e cellplm python run_exp.py exp/cellplm/85M-20231027/brca_cell_type.yaml
# pixi run -e cellplm python run_exp.py exp/cellplm/85M-20231027/brca_chemo.yaml
# pixi run -e cellplm python run_exp.py exp/cellplm/85M-20231027/brca_outcome.yaml
# pixi run -e cellplm python run_exp.py exp/cellplm/85M-20231027/brca_pre_post.yaml
# pixi run -e cellplm python run_exp.py exp/cellplm/85M-20231027/luad_tki.yaml
# pixi run -e cellplm python run_exp.py exp/cellplm/85M-20231027/luad_cancer_stage.yaml

pixi run -e cellplm python run_exp.py exp/cellplm/85M-20231027/crc_mmr.yaml
# pixi run -e cellplm python run_exp.py exp/cellplm/85M-20231027/melanoma_response.yaml
