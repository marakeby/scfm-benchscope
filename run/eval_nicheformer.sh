# Nicheformer — pixi env: nicheformer (isolated stack; see scripts/pixi_pythonpath_src.sh).
# Usage: bash run/eval_nicheformer.sh (from repo root) or: cd run && bash eval_nicheformer.sh
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONWARNINGS="ignore::DeprecationWarning:pandas${PYTHONWARNINGS:+,${PYTHONWARNINGS}}"
#Embedding tasks --------------------       
pixi run -e nicheformer python -m scfm_cancer_eval.run.run_exp exp/nicheformer/nicheformer/brca_cell_type.yaml 
#Classification tasks --------------------
#BRCA
pixi run -e nicheformer python -m scfm_cancer_eval.run.run_exp exp/nicheformer/nicheformer/brca_subtype.yaml 
pixi run -e nicheformer python -m scfm_cancer_eval.run.run_exp exp/nicheformer/nicheformer/brca_chemo.yaml 
pixi run -e nicheformer python -m scfm_cancer_eval.run.run_exp exp/nicheformer/nicheformer/brca_outcome.yaml 
pixi run -e nicheformer python -m scfm_cancer_eval.run.run_exp exp/nicheformer/nicheformer/brca_pre_post.yaml 
#LUAD
pixi run -e nicheformer python -m scfm_cancer_eval.run.run_exp exp/nicheformer/nicheformer/luad_tki.yaml 
#CRC
pixi run -e nicheformer python -m scfm_cancer_eval.run.run_exp exp/nicheformer/nicheformer/crc_mmr.yaml 
#Melanoma
pixi run -e nicheformer python -m scfm_cancer_eval.run.run_exp exp/nicheformer/nicheformer/melanoma_response.yaml 