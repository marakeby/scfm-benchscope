# ============================================================
# scFM suite container (multi-env) on your CUDA 12.8 torch base
# ============================================================
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MAMBA_ROOT_PREFIX=/opt/micromamba

# ---------------------------
# 1) OS dependencies
# ---------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates \
    build-essential pkg-config \
    cmake ninja-build \
    libffi-dev libssl-dev \
    libgl1 \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------
# 2) Install micromamba (ROBUST) - NO "shell init"
# ---------------------------
RUN mkdir -p /tmp/micromamba \
 && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /tmp/micromamba \
 && mv /tmp/micromamba/bin/micromamba /usr/local/bin/micromamba \
 && chmod +x /usr/local/bin/micromamba \
 && /usr/local/bin/micromamba --version

ENV PATH=${MAMBA_ROOT_PREFIX}/bin:${PATH}

# Configure channels
RUN micromamba config append channels conda-forge \
 && micromamba config set channel_priority strict

# ---------------------------
# 3) Common torch wheel config
# ---------------------------
ARG TORCH_VERSION=2.7.1
ARG CUDA_WHL=cu128
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/${CUDA_WHL}
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------------------------
# 4) Envs
# ---------------------------

# ---- cellPLM (python 3.9 recommended by repo) + metrics stack ----
RUN micromamba create -y -n cellplm python=3.9 pip \
 && micromamba run -n cellplm pip install -U pip setuptools wheel \
 && micromamba run -n cellplm pip install torch==${TORCH_VERSION}+${CUDA_WHL} \
 && micromamba run -n cellplm pip install cellplm \
 && micromamba install -y -n cellplm scib=1.1.5 igraph leidenalg \
 && micromamba run -n cellplm pip install harmonypy bbknn umap-learn pynndescent lisi \
 && micromamba clean -a -y

# ---- scvi-tools ----
RUN micromamba create -y -n scvi python=3.11 pip \
 && micromamba run -n scvi pip install -U pip setuptools wheel \
 && micromamba run -n scvi pip install torch==${TORCH_VERSION}+${CUDA_WHL} \
 && micromamba run -n scvi pip install scvi-tools \
 && micromamba clean -a -y

# ---- Geneformer (requirements from HF repo; pins transformers==4.46) ----
RUN micromamba create -y -n geneformer python=3.11 pip \
 && micromamba run -n geneformer pip install -U pip setuptools wheel \
 && micromamba run -n geneformer pip install torch==${TORCH_VERSION}+${CUDA_WHL} \
 && micromamba run -n geneformer pip install \
      "transformers==4.46" \
      "datasets>=2.12" \
      "anndata>=0.9" "scanpy>=1.9" "loompy>=3.0" \
      "numpy>=1.23" "pandas>=2.0" "scipy>=1.10" "scikit_learn>=1.2" \
      "matplotlib>=3.7" "seaborn>=0.12" "tensorboard>=2.15" "tqdm>=4.65" \
      "bitsandbytes>=0.45.5" "peft>=0.11.1" \
      "ray>=2.6" "optuna>=3.6" "optuna-integration>=3.6" \
 && micromamba clean -a -y

# ---- scGPT (install package; skip flash-attn by default) ----
RUN micromamba create -y -n scgpt python=3.11 pip \
 && micromamba run -n scgpt pip install -U pip setuptools wheel \
 && micromamba run -n scgpt pip install torch==${TORCH_VERSION}+${CUDA_WHL} \
 && micromamba run -n scgpt pip install scgpt \
 && micromamba clean -a -y

# ---- scimilarity ----
RUN micromamba create -y -n scimilarity python=3.11 pip \
 && micromamba run -n scimilarity pip install -U pip setuptools wheel \
 && micromamba run -n scimilarity pip install torch==${TORCH_VERSION}+${CUDA_WHL} \
 && micromamba run -n scimilarity pip install scimilarity \
 && micromamba clean -a -y

# ---- nicheformer ----
RUN micromamba create -y -n nicheformer python=3.11 pip \
 && micromamba run -n nicheformer pip install -U pip setuptools wheel \
 && micromamba run -n nicheformer pip install torch==${TORCH_VERSION}+${CUDA_WHL} \
 && micromamba run -n nicheformer pip install git+https://github.com/theislab/nicheformer.git \
 && micromamba clean -a -y

# ---- scConcept (flash-attn can be brittle; isolated env) ----
RUN micromamba create -y -n scconcept python=3.11 pip \
 && micromamba run -n scconcept pip install -U pip setuptools wheel \
 && micromamba run -n scconcept pip install torch==${TORCH_VERSION}+${CUDA_WHL} \
 && micromamba run -n scconcept pip install --no-build-isolation "flash-attn==2.7.*" \
 && micromamba run -n scconcept pip install git+https://github.com/theislab/lamin_dataloader.git \
 && micromamba run -n scconcept pip install git+https://github.com/theislab/scConcept.git@main \
 && micromamba clean -a -y

# ---- STATE (arc-state) ----
RUN micromamba create -y -n state python=3.11 pip \
 && micromamba run -n state pip install -U pip setuptools wheel \
 && micromamba run -n state pip install torch==${TORCH_VERSION}+${CUDA_WHL} \
 && micromamba run -n state pip install arc-state \
 && micromamba clean -a -y

# ---- scFoundation ----
RUN micromamba create -y -n scfoundation python=3.11 pip \
 && micromamba run -n scfoundation pip install -U pip setuptools wheel \
 && micromamba run -n scfoundation pip install torch==${TORCH_VERSION}+${CUDA_WHL} \
 && micromamba run -n scfoundation pip install git+https://github.com/biomap-research/scFoundation.git \
 && micromamba clean -a -y

WORKDIR /workspace
CMD ["/bin/bash"]