# #!/bin/sh
# export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 

# --------------------------------LUAD1---------------------------------------------

python -m scfm_cancer_eval.run.run_exp luad1/hvg.yaml
python -m scfm_cancer_eval.run.run_exp luad1/pca.yaml
python -m scfm_cancer_eval.run.run_exp luad1/scgpt.yaml
python -m scfm_cancer_eval.run.run_exp luad1/scgpt_cancer.yaml
python -m scfm_cancer_eval.run.run_exp luad1/scvi.yaml


# --------------------------------LUAD2---------------------------------------------

python -m scfm_cancer_eval.run.run_exp luad2/hvg.yaml
python -m scfm_cancer_eval.run.run_exp luad2/pca.yaml
python -m scfm_cancer_eval.run.run_exp luad2/scvi.yaml
python -m scfm_cancer_eval.run.run_exp luad2/scgpt.yaml
python -m scfm_cancer_eval.run.run_exp luad2/scgpt_cancer.yaml

