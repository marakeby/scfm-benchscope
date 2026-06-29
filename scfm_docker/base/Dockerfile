FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH=/opt/venv

# -------------------------------------------------
# System dependencies
# -------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates \
    build-essential pkg-config \
    # common plotting / scanpy deps
    libgl1 \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# Python virtual environment
# -------------------------------------------------
RUN python -m venv ${VENV_PATH} \
 && ${VENV_PATH}/bin/pip install --upgrade pip setuptools wheel

ENV PATH="${VENV_PATH}/bin:${PATH}"

# -------------------------------------------------
# Core scientific + benchmarking utilities (shared)
# -------------------------------------------------
RUN pip install \
    numpy \
    scipy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm \
    joblib \
    pyyaml

# faiss-cpu is useful for kNN / retrieval benchmarking without CUDA complexity
RUN pip install faiss-cpu

# -------------------------------------------------
# Single-cell ecosystem (shared)
# -------------------------------------------------
RUN pip install \
    anndata \
    scanpy

# -------------------------------------------------
# Metrics / clustering / evaluation (shared)
# -------------------------------------------------
RUN pip install \
    scib==1.1.5 \
    scib-metrics \
    igraph \
    leidenalg \
    harmonypy \
    bbknn \
    umap-learn \
    pynndescent

# -------------------------------------------------
# Sanity checks (fail build early if something breaks)
# -------------------------------------------------
RUN python - <<'PY'
import numpy, pandas, scipy, sklearn
import anndata, scanpy
import scib, igraph, leidenalg
import umap, pynndescent
import joblib, yaml, tqdm
import faiss
import scib_metrics
import scib, igraph, leidenalg
print("Base stack OK")
PY



WORKDIR /workspace
CMD ["bash"]