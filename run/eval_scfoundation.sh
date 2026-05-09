# scFoundation — pixi env: scf (install Biomap stack per project README)
# Usage: bash run/eval_scfoundation.sh (from repo root) or: cd run && bash eval_scfoundation.sh
cd "$(dirname "$0")"
# pixi run -e scf python run_exp.py exp/scfoundation/brca_cancer_cells/brca_subtype.yaml
# pixi run -e scf python run_exp.py exp/scfoundation/brca_cancer_cells/brca_cell_type.yaml
# pixi run -e scf python run_exp.py exp/scfoundation/brca_cancer_cells/brca_chemo.yaml
# pixi run -e scf python run_exp.py exp/scfoundation/brca_cancer_cells/brca_outcome.yaml
# pixi run -e scf python run_exp.py exp/scfoundation/brca_cancer_cells/brca_pre_post.yaml
# pixi run -e scf python run_exp.py exp/scfoundation/brca_cancer_cells/luad_tki.yaml
# pixi run -e scf python run_exp.py exp/scfoundation/brca_cancer_cells/luad_cancer_stage.yaml

pixi run -e scf python run_exp.py exp/scfoundation/brca_cancer_cells/crc_mmr.yaml
# pixi run -e scf python run_exp.py exp/scfoundation/melanoma_response.yaml