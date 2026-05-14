#!/usr/bin/env bash
# Isolated Pixi envs (nicheformer, scconcept, …) use no-default-feature and do not install
# scfm-cancer-eval from pyproject (often incompatible pins vs upstream models). Put this repo
# on PYTHONPATH so `python -m scfm_cancer_eval...` resolves. Pixi sets PIXI_PROJECT_ROOT.
if [[ -n "${PIXI_PROJECT_ROOT:-}" ]]; then
  export PYTHONPATH="${PIXI_PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
fi

# Nicheformer pins pandas 1.5.3, which still uses np.find_common_type (deprecated in NumPy 1.25+).
# Nicheformer also pins numpy==1.26.4, so we cannot cap NumPy or upgrade pandas here; suppress
# DeprecationWarning only for the pandas package (regex matches module names starting with "pandas").
if [[ -z "${PYTHONWARNINGS:-}" ]]; then
  export PYTHONWARNINGS="ignore::DeprecationWarning:pandas"
else
  export PYTHONWARNINGS="ignore::DeprecationWarning:pandas,${PYTHONWARNINGS}"
fi
