# pixi run python run_exp.py brca_full/subtype/hvg.yaml

#Done
pixi run python run_exp.py brca_full/pre_post/hvg.yaml
pixi run -e scvi python run_exp.py brca_full/pre_post/scvi.yaml
pixi run python run_exp.py brca_full/pre_post/pca.yaml 

#scgpt
pixi run -e scgpt python run_exp.py brca_full/pre_post/scgpt.yaml
pixi run -e scgpt python run_exp.py brca_full/pre_post/scgpt_cancer.yaml


#geneformer
pixi run -e geneformer python run_exp.py brca_full/pre_post/gf-6L-30M-i2048.yaml
pixi run -e geneformer python run_exp.py brca_full/pre_post/Geneformer-V2-104M.yaml
pixi run -e geneformer python run_exp.py brca_full/pre_post/Geneformer-V2-104M_CLcancer.yaml
pixi run -e geneformer python run_exp.py brca_full/pre_post/Geneformer-V2-316M.yaml
pixi run -e geneformer python run_exp.py brca_full/pre_post/Geneformer-V2-104M_finetune.yaml


#others
pixi run -e scf python run_exp.py brca_full/pre_post/scfoundation.yaml
pixi run -e scimilarity python run_exp.py brca_full/pre_post/scimilarity.yaml
pixi run -e cellplm python run_exp.py brca_full/pre_post/cellplm.yaml

#new models
pixi run -e scconcept python run_exp.py brca_full/pre_post/scconcept.yaml
pixi run -e nicheformer python run_exp.py brca_full/pre_post/nicheformer.yaml
pixi run -e state python run_exp.py brca_full/pre_post/state.yaml

#progress



#error
