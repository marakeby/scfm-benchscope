# scConcept — pixi env: scconcept (isolated stack)
# Usage: bash run/eval_scconcept.sh (from repo root) or: cd run && bash eval_scconcept.sh
cd "$(dirname "$0")"
# pixi run -e scconcept python run_exp.py exp/scconcept/corpus30m/brca_subtype.yaml
# pixi run -e scconcept python run_exp.py exp/scconcept/corpus30m/brca_cell_type.yaml
# pixi run -e scconcept python run_exp.py exp/scconcept/corpus30m/brca_chemo.yaml
# pixi run -e scconcept python run_exp.py exp/scconcept/corpus30m/brca_outcome.yaml
# pixi run -e scconcept python run_exp.py exp/scconcept/corpus30m/brca_pre_post.yaml
# pixi run -e scconcept python run_exp.py exp/scconcept/corpus30m/luad_tki.yaml
# pixi run -e scconcept python run_exp.py exp/scconcept/corpus30m/luad_cancer_stage.yaml

# pixi run -e scconcept python run_exp.py exp/scconcept/corpus30m/crc_mmr.yaml
pixi run -e scconcept python run_exp.py exp/scconcept/corpus30m/melanoma_response.yaml
