FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH=/opt/venv \
    # helps compilation speed/consistency
    MAX_JOBS=8

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    git curl ca-certificates \
    build-essential pkg-config \
    cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv ${VENV_PATH} \
 && ${VENV_PATH}/bin/pip install -U pip setuptools wheel
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Torch cu128
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128
RUN pip install torch==2.7.1+cu128

# ---- IMPORTANT: build-time deps needed by flash-attn metadata/build ----
RUN pip install \
    numpy==1.26.4 \
    psutil==6.0.0 \
    packaging==24.1 \
    ninja==1.11.1.1 \
    cmake==3.30.3

# Install flash-attn (scConcept requires flash-attn==2.7.*)
RUN pip install --no-build-isolation "flash-attn==2.7.*"

# scConcept itself
RUN pip install git+https://github.com/theislab/lamin_dataloader.git
RUN pip install git+https://github.com/theislab/scConcept.git@main

# Smoke test
RUN python - <<'PY'
import torch
import numpy, psutil
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda, "gpu?", torch.cuda.is_available())
print("numpy:", numpy.__version__, "psutil:", psutil.__version__)
PY

WORKDIR /workspace
CMD ["bash"]