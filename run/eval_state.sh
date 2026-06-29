# STATE — pixi env: state
# Usage: bash run/eval_state.sh (from repo root) or: cd run && bash eval_state.sh
cd "$(dirname "$0")/.."

#Embedding tasks --------------------           
pixi run -e state python -m scfm_cancer_eval.run.run_exp exp/state/se600m_epoch16/brca_cell_type.yaml 

#Classification tasks --------------------
#BRCA   
pixi run -e state python -m scfm_cancer_eval.run.run_exp exp/state/se600m_epoch16/brca_subtype.yaml 
pixi run -e state python -m scfm_cancer_eval.run.run_exp exp/state/se600m_epoch16/brca_chemo.yaml 
pixi run -e state python -m scfm_cancer_eval.run.run_exp exp/state/se600m_epoch16/brca_outcome.yaml
pixi run -e state python -m scfm_cancer_eval.run.run_exp exp/state/se600m_epoch16/brca_pre_post.yaml 
#LUAD
pixi run -e state python -m scfm_cancer_eval.run.run_exp exp/state/se600m_epoch16/luad_tki.yaml 
#CRC
pixi run -e state python -m scfm_cancer_eval.run.run_exp exp/state/se600m_epoch16/crc_mmr.yaml 
#Melanoma
pixi run -e state python -m scfm_cancer_eval.run.run_exp exp/state/se600m_epoch16/melanoma_response.yaml 