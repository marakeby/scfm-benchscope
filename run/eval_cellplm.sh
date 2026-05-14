# CellPLM — pixi env: cellplm
# Usage: bash run/eval_cellplm.sh (from repo root) or: cd run && bash eval_cellplm.sh
#
# By default this script does not pass --max-cells / --max-cells-stratify, so all loaded cells
# are kept unless dataset YAML sets max_cells (or you export MAX_CELLS / MAX_CELLS_STRATIFY).
# Example subsampled run: MAX_CELLS=2000 MAX_CELLS_STRATIFY=batch bash run/eval_cellplm.sh
# For unstratified subsampling use run_exp directly: ... --max-cells 2000 --max-cells-stratify ''
cd "$(dirname "$0")/.."
# ${VAR-default}: empty default so subset flags are omitted unless you export these.
MAX_CELLS="${MAX_CELLS-}"
MAX_CELLS_STRATIFY="${MAX_CELLS_STRATIFY-}"

SUBSET_ARGS=()
[[ -n "$MAX_CELLS" ]] && SUBSET_ARGS+=(--max-cells "$MAX_CELLS")
[[ -n "$MAX_CELLS_STRATIFY" ]] && SUBSET_ARGS+=(--max-cells-stratify "$MAX_CELLS_STRATIFY")

#Embedding tasks --------------------
pixi run -e cellplm python -m scfm_cancer_eval.run.run_exp exp/cellplm/85M-20231027/brca_cell_type.yaml "${SUBSET_ARGS[@]}"
#Classification tasks --------------------
#BRCA
pixi run -e cellplm python -m scfm_cancer_eval.run.run_exp exp/cellplm/85M-20231027/brca_subtype.yaml "${SUBSET_ARGS[@]}"
pixi run -e cellplm python -m scfm_cancer_eval.run.run_exp exp/cellplm/85M-20231027/brca_chemo.yaml "${SUBSET_ARGS[@]}"
pixi run -e cellplm python -m scfm_cancer_eval.run.run_exp exp/cellplm/85M-20231027/brca_outcome.yaml "${SUBSET_ARGS[@]}"
pixi run -e cellplm python -m scfm_cancer_eval.run.run_exp exp/cellplm/85M-20231027/brca_pre_post.yaml "${SUBSET_ARGS[@]}"
#LUAD
pixi run -e cellplm python -m scfm_cancer_eval.run.run_exp exp/cellplm/85M-20231027/luad_tki.yaml "${SUBSET_ARGS[@]}"
#CRC
pixi run -e cellplm python -m scfm_cancer_eval.run.run_exp exp/cellplm/85M-20231027/crc_mmr.yaml "${SUBSET_ARGS[@]}"        
#Melanoma
pixi run -e cellplm python -m scfm_cancer_eval.run.run_exp exp/cellplm/85M-20231027/melanoma_response.yaml "${SUBSET_ARGS[@]}"
