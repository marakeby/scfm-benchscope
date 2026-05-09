# Nicheformer — pixi env: nicheformer (isolated stack)
# Usage: bash run/eval_nicheformer.sh (from repo root) or: cd run && bash eval_nicheformer.sh
cd "$(dirname "$0")"
# pixi run -e nicheformer python run_exp.py exp/nicheformer/nicheformer/brca_subtype.yaml
# pixi run -e nicheformer python run_exp.py exp/nicheformer/nicheformer/brca_cell_type.yaml
# pixi run -e nicheformer python run_exp.py exp/nicheformer/nicheformer/brca_chemo.yaml
# pixi run -e nicheformer python run_exp.py exp/nicheformer/nicheformer/brca_outcome.yaml
# pixi run -e nicheformer python run_exp.py exp/nicheformer/nicheformer/brca_pre_post.yaml
# pixi run -e nicheformer python run_exp.py exp/nicheformer/nicheformer/luad_tki.yaml
# pixi run -e nicheformer python run_exp.py exp/nicheformer/nicheformer/luad_cancer_stage.yaml

# pixi run -e nicheformer python run_exp.py exp/nicheformer/nicheformer/crc_mmr.yaml 
pixi run -e nicheformer python run_exp.py exp/nicheformer/nicheformer/melanoma_response.yaml