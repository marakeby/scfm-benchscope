# #!/bin/sh
# export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 

#--------------------------------Pre vs Post ---------------------------------------------

python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/hvg.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/pca.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/scvi.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/scgpt.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/scgpt_cancer.yaml


#--------------------------------subtype (ER+ vs TNBC)---------------------------------------------

python -m scfm_cancer_eval.run.run_exp brca_full/subtype/hvg.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/subtype/pca.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/subtype/scvi.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/subtype/scgpt.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/subtype/scgpt_cancer.yaml


#--------------------------------outcome E vs NE (Tcells)---------------------------------------------

python -m scfm_cancer_eval.run.run_exp brca_full/outcome/hvg.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/outcome/pca.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/outcome/scvi.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/outcome/scgpt.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/outcome/scgpt_cancer.yaml


#--------------------------------cohort chemo vs naive( Cancer cells)---------------------------------------------

python -m scfm_cancer_eval.run.run_exp brca_full/chemo/hvg.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/chemo/pca.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/chemo/scvi.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/chemo/scgpt.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/chemo/scgpt_cancer.yaml


#-------------------------------- cell types ---------------------------------------------
#go from here
python -m scfm_cancer_eval.run.run_exp brca_full/cell_type/hvg.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/cell_type/pca.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/cell_type/scvi.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/cell_type/scgpt.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/cell_type/scgpt_cancer.yaml

#-------------------------------- ALL Cells ---------------------------------------------

python -m scfm_cancer_eval.run.run_exp brca_full/all_cells/hvg.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/all_cells/pca.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/all_cells/scvi.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/all_cells/scgpt.yaml
python -m scfm_cancer_eval.run.run_exp brca_full/all_cells/scgpt_cancer.yaml

