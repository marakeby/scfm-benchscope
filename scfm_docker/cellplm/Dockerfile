# Base image with torch 2.7.1 + CUDA 12.8 + Ubuntu 22.04
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies (common for scanpy + plotting + igraph build wheels fallback)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    build-essential pkg-config \
    libgl1 \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip toolchain
RUN python -m pip install --upgrade pip setuptools wheel

# ---- Core single-cell stack ----
RUN pip install \
    numpy scipy pandas \
    scikit-learn \
    anndata scanpy

# ---- CellPLM ----
RUN pip install cellplm

# ---- Metrics / clustering + integration benchmarking ----
# scIB + clustering tools
RUN pip install \
    scib==1.1.5 \
    igraph \
    leidenalg

# Extra evaluation / integration deps commonly used with scIB
RUN pip install \
    harmonypy \
    bbknn \
    umap-learn \
    pynndescent \
    lisi

# Optional: notebooks
RUN pip install \
    jupyterlab ipykernel \
    matplotlib seaborn

RUN python -m ipykernel install --user --name cellplm --display-name "Python (cellplm)"

WORKDIR /workspace
CMD ["bash"]