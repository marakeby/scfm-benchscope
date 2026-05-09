# STATE — pixi env: state
# Usage: bash run/eval_state.sh (from repo root) or: cd run && bash eval_state.sh
cd "$(dirname "$0")"
# pixi run -e state python run_exp.py exp/state/se600m_epoch16/brca_subtype.yaml
# pixi run -e state python run_exp.py exp/state/se600m_epoch16/brca_cell_type.yaml
# pixi run -e state python run_exp.py exp/state/se600m_epoch16/brca_chemo.yaml
# pixi run -e state python run_exp.py exp/state/se600m_epoch16/brca_outcome.yaml
# pixi run -e state python run_exp.py exp/state/se600m_epoch16/brca_pre_post.yaml
# pixi run -e state python run_exp.py exp/state/se600m_epoch16/luad_tki.yaml
# pixi run -e state python run_exp.py exp/state/se600m_epoch16/luad_cancer_stage.yaml
pixi run -e state python run_exp.py exp/state/se600m_epoch16/crc_mmr.yaml
pixi run -e state python run_exp.py exp/state/se600m_epoch16/melanoma_response.yaml