FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH=/opt/venv \
    MAX_JOBS=8

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates \
    build-essential pkg-config \
    cmake ninja-build \
 && rm -rf /var/lib/apt/lists/*

# Isolated venv (prevents polluting base python)
RUN python -m venv ${VENV_PATH} \
 && ${VENV_PATH}/bin/pip install -U pip setuptools wheel
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Build-time deps needed by flash-attn setup/metadata
RUN pip install \
    numpy==1.26.4 \
    psutil==6.0.0 \
    packaging==24.1

# scConcept requires flash-attn==2.7.*
RUN pip install --no-build-isolation "flash-attn==2.7.*"

# Install scConcept
RUN pip install git+https://github.com/theislab/lamin_dataloader.git
RUN pip install git+https://github.com/theislab/scConcept.git@main

# Smoke test (don’t assume import name)
RUN python - <<'PY'
import torch
import numpy, psutil
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda, "gpu?", torch.cuda.is_available())
print("numpy:", numpy.__version__, "psutil:", psutil.__version__)
PY

WORKDIR /workspace
CMD ["bash"]