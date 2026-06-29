# scConcept — pixi env: scconcept (isolated stack; see scripts/pixi_pythonpath_src.sh).
# Usage: bash run/eval_scconcept.sh (from repo root) or: cd run && bash eval_scconcept.sh
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"


#Embedding tasks --------------------
pixi run -e scconcept python -m scfm_cancer_eval.run.run_exp exp/scconcept/corpus30m/brca_cell_type.yaml 
#Classification tasks --------------------
#BRCA
pixi run -e scconcept python -m scfm_cancer_eval.run.run_exp exp/scconcept/corpus30m/brca_subtype.yaml 
pixi run -e scconcept python -m scfm_cancer_eval.run.run_exp exp/scconcept/corpus30m/brca_chemo.yaml 
pixi run -e scconcept python -m scfm_cancer_eval.run.run_exp exp/scconcept/corpus30m/brca_outcome.yaml 
pixi run -e scconcept python -m scfm_cancer_eval.run.run_exp exp/scconcept/corpus30m/brca_pre_post.yaml 
# LUAD
pixi run -e scconcept python -m scfm_cancer_eval.run.run_exp exp/scconcept/corpus30m/luad_tki.yaml 
# CRC
pixi run -e scconcept python -m scfm_cancer_eval.run.run_exp exp/scconcept/corpus30m/crc_mmr.yaml 
# Melanoma
pixi run -e scconcept python -m scfm_cancer_eval.run.run_exp exp/scconcept/corpus30m/melanoma_response.yaml 
