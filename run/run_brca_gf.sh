# #!/bin/sh
# export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 

# --------------------------------cohort chemo vs naive( Cancer cells)--------------------------------------------

python -m scfm_cancer_eval.run.run_exp brca_full/chemo/gf-6L-30M-i2048.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/chemo/Geneformer-V2-104M.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/chemo/Geneformer-V2-104M_CLcancer.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/chemo/Geneformer-V2-316M.yaml

python -m scfm_cancer_eval.run.run_exp brca_full/chemo/scfoundation.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/chemo/scimilarity.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/chemo/cellplm.yaml

# #finetune
python -m scfm_cancer_eval.run.run_exp brca_full/chemo/gf-6L-30M-i2048_finetune.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/chemo/Geneformer-V2-104M_finetune.yaml


# --------------------------------Pre vs Post ---------------------------------------------

python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/gf-6L-30M-i2048.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/Geneformer-V2-104M.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/Geneformer-V2-104M_CLcancer.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/Geneformer-V2-316M.yaml

python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/scfoundation.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/scimilarity.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/cellplm.yaml

# finetune
python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/gf-6L-30M-i2048_finetune.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/Geneformer-V2-104M_finetune.yaml


# --------------------------------outcome E vs NE (Tcells)---------------------------------------------


python -m scfm_cancer_eval.run.run_exp brca_full/outcome/gf-6L-30M-i2048.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/outcome/Geneformer-V2-104M.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/outcome/Geneformer-V2-104M_CLcancer.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/outcome/Geneformer-V2-316M.yaml

python -m scfm_cancer_eval.run.run_exp brca_full/outcome/scfoundation.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/outcome/scimilarity.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/outcome/cellplm.yaml

# finetune
python -m scfm_cancer_eval.run.run_exp brca_full/outcome/gf-6L-30M-i2048_finetune.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/outcome/Geneformer-V2-104M_finetune.yaml


# --------------------------------subtype (ER+ vs TNBC)---------------------------------------------


python -m scfm_cancer_eval.run.run_exp brca_full/subtype/gf-6L-30M-i2048.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/subtype/Geneformer-V2-104M.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/subtype/Geneformer-V2-104M_CLcancer.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/subtype/Geneformer-V2-316M.yaml

python -m scfm_cancer_eval.run.run_exp brca_full/subtype/scfoundation.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/subtype/scimilarity.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/subtype/cellplm.yaml

# # finetune
python -m scfm_cancer_eval.run.run_exp brca_full/outcome/gf-6L-30M-i2048_finetune.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/outcome/Geneformer-V2-104M_finetune.yaml



# -------------------------------- cell types ---------------------------------------------

python -m scfm_cancer_eval.run.run_exp brca_full/cell_type/hvg.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/cell_type/pca.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/cell_type/scvi.yaml

python -m scfm_cancer_eval.run.run_exp brca_full/cell_type/scgpt.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/cell_type/scgpt_cancer.yaml

python -m scfm_cancer_eval.run.run_exp brca_full/cell_type/gf-6L-30M-i2048.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/cell_type/Geneformer-V2-104M.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/cell_type/Geneformer-V2-104M_CLcancer.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/cell_type/Geneformer-V2-316M.yaml


python -m scfm_cancer_eval.run.run_exp brca_full/cell_type/scfoundation.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/cell_type/scimilarity.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/cell_type/cellplm.yaml

# # Continual Training
python -m scfm_cancer_eval.run.run_exp brca_full/cell_type/gf-6L-30M-i2048_continue.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/cell_type/Geneformer-V2-104M_continue.yaml
