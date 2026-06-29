# #!/bin/sh
# export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 

# --------------------------------LUAD1---------------------------------------------

#GF
python -m scfm_cancer_eval.run.run_exp luad1/gf-6L-30M-i2048.yaml
python -m scfm_cancer_eval.run.run_exp luad1/gf-6L-30M-i2048_no_batch.yaml
python -m scfm_cancer_eval.run.run_exp luad1/Geneformer-V2-104M.yaml
python -m scfm_cancer_eval.run.run_exp luad1/Geneformer-V2-104M_CLcancer.yaml
python -m scfm_cancer_eval.run.run_exp luad1/Geneformer-V2-316M.yaml

#Other
python -m scfm_cancer_eval.run.run_exp luad1/scfoundation.yaml
python -m scfm_cancer_eval.run.run_exp luad1/scimilarity.yaml
python -m scfm_cancer_eval.run.run_exp luad1/cellplm.yaml


# finetune
python -m scfm_cancer_eval.run.run_exp luad1/gf-6L-30M-i2048_finetune.yaml
python -m scfm_cancer_eval.run.run_exp luad1/Geneformer-V2-104M_finetune.yaml




# --------------------------------LUAD2---------------------------------------------

python -m scfm_cancer_eval.run.run_exp luad2/gf-6L-30M-i2048.yaml
python -m scfm_cancer_eval.run.run_exp luad2/Geneformer-V2-104M.yaml
python -m scfm_cancer_eval.run.run_exp luad2/Geneformer-V2-104M_CLcancer.yaml
python -m scfm_cancer_eval.run.run_exp luad2/Geneformer-V2-316M.yaml

#Other
python -m scfm_cancer_eval.run.run_exp luad2/scfoundation.yaml
python -m scfm_cancer_eval.run.run_exp luad2/scimilarity.yaml
python -m scfm_cancer_eval.run.run_exp luad2/cellplm.yaml

# finetune
python -m scfm_cancer_eval.run.run_exp luad2/gf-6L-30M-i2048_finetune.yaml
python -m scfm_cancer_eval.run.run_exp luad2/Geneformer-V2-104M_finetune.yaml