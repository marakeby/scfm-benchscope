# pixi run python -m scfm_cancer_eval.run.run_exp brca_full/subtype/hvg.yaml

#Done
pixi run python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/hvg.yaml
pixi run -e scvi python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/scvi.yaml
pixi run python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/pca.yaml 

#scgpt
pixi run -e scgpt python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/scgpt.yaml
pixi run -e scgpt python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/scgpt_cancer.yaml


#geneformer
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/gf-6L-30M-i2048.yaml
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/Geneformer-V2-104M.yaml
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/Geneformer-V2-104M_CLcancer.yaml
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/Geneformer-V2-316M.yaml
pixi run -e geneformer python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/Geneformer-V2-104M_finetune.yaml


#others
pixi run -e scf python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/scfoundation.yaml
pixi run -e scimilarity python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/scimilarity.yaml
pixi run -e cellplm python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/cellplm.yaml

#new models
pixi run -e scconcept python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/scconcept.yaml
pixi run -e nicheformer python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/nicheformer.yaml
pixi run -e state python -m scfm_cancer_eval.run.run_exp brca_full/pre_post/state.yaml

#progress



#error
