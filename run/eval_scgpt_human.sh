# scGPT (human checkpoint) — pixi env: scgpt
# Usage: bash run/eval_scgpt_human.sh (from repo root) or: cd run && bash eval_scgpt_human.sh
cd "$(dirname "$0")"

#Embedding tasks --------------------
pixi run -e scgpt python run_exp.py exp/scgpt/human-i2048/brca_cell_type.yaml

#Classification tasks --------------------
pixi run -e scgpt python run_exp.py exp/scgpt/human-i2048/brca_subtype.yaml
pixi run -e scgpt python run_exp.py exp/scgpt/human-i2048/brca_chemo.yaml
pixi run -e scgpt python run_exp.py exp/scgpt/human-i2048/brca_outcome.yaml
pixi run -e scgpt python run_exp.py exp/scgpt/human-i2048/brca_pre_post.yaml

# pixi run -e scgpt python run_exp.py exp/scgpt/human-i2048/luad_tki.yaml
# pixi run -e scgpt python run_exp.py exp/scgpt/human-i2048/luad_cancer_stage.yaml
# pixi run -e scgpt python run_exp.py exp/scgpt/human-i2048/crc_mmr.yaml
# pixi run -e scgpt python run_exp.py exp/scgpt/human-i2048/melanoma_response.yaml