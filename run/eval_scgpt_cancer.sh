# scGPT (cancer checkpoint) — pixi env: scgpt
# Usage: bash run/eval_scgpt_cancer.sh (from repo root) or: cd run && bash eval_scgpt_cancer.sh
cd "$(dirname "$0")/.."


#Embedding tasks --------------------
pixi run -e scgpt python -m scfm_cancer_eval.run.run_exp exp/scgpt/cancer-i2048/brca_cell_type.yaml 
#Classification tasks --------------------
#BRCA
pixi run -e scgpt python -m scfm_cancer_eval.run.run_exp exp/scgpt/cancer-i2048/brca_subtype.yaml 
pixi run -e scgpt python -m scfm_cancer_eval.run.run_exp exp/scgpt/cancer-i2048/brca_chemo.yaml 
pixi run -e scgpt python -m scfm_cancer_eval.run.run_exp exp/scgpt/cancer-i2048/brca_outcome.yaml
pixi run -e scgpt python -m scfm_cancer_eval.run.run_exp exp/scgpt/cancer-i2048/brca_pre_post.yaml 
#LUAD
pixi run -e scgpt python -m scfm_cancer_eval.run.run_exp exp/scgpt/cancer-i2048/luad_tki.yaml 
# pixi run -e scgpt python -m scfm_cancer_eval.run.run_exp exp/scgpt/cancer-i2048/luad_cancer_stage.yaml 
#CRC
pixi run -e scgpt python -m scfm_cancer_eval.run.run_exp exp/scgpt/cancer-i2048/crc_mmr.yaml 
#Melanoma
pixi run -e scgpt python -m scfm_cancer_eval.run.run_exp exp/scgpt/cancer-i2048/melanoma_response.yaml 
