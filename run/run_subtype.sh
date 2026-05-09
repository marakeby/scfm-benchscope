# pixi run python run_exp.py brca_full/subtype/hvg.yaml

#Done
# pixi run python run_exp.py brca_full/subtype/hvg.yaml
# pixi run -e scvi python run_exp.py brca_full/subtype/scvi.yaml
# pixi run python run_exp.py brca_full/subtype/pca.yaml 

#scgpt
# pixi run -e scgpt python run_exp.py brca_full/subtype/scgpt.yaml
# pixi run -e scgpt python run_exp.py brca_full/subtype/scgpt_cancer.yaml


#geneformer
pixi run -e geneformer python run_exp.py brca_full/subtype/gf-6L-30M-i2048.yaml #error
# pixi run -e geneformer python run_exp.py brca_full/subtype/Geneformer-V2-104M.yaml
# pixi run -e geneformer python run_exp.py brca_full/subtype/Geneformer-V2-104M_CLcancer.yaml
# pixi run -e geneformer python run_exp.py brca_full/subtype/Geneformer-V2-316M.yaml
# pixi run -e geneformer python run_exp.py brca_full/subtype/Geneformer-V2-104M_finetune.yaml #error


#others
# pixi run -e scf python run_exp.py brca_full/subtype/scfoundation.yaml
# pixi run -e scimilarity python run_exp.py brca_full/subtype/scimilarity.yaml #error
# pixi run -e cellplm python run_exp.py brca_full/subtype/cellplm.yaml

#new models
# pixi run -e scconcept python run_exp.py brca_full/subtype/scconcept.yaml #error
# pixi run -e nicheformer python run_exp.py brca_full/subtype/nicheformer.yaml #error
# pixi run -e state python run_exp.py brca_full/subtype/state.yaml #error

#progress



#error
