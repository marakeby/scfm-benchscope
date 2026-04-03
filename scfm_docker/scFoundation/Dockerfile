# CUDA-enabled PyTorch runtime (good default for inference + embeddings)
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for scanpy/h5py/scipy, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    build-essential pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy repo into image (you can also volume-mount instead)
COPY . /workspace

# Upgrade pip tooling
RUN python -m pip install --upgrade pip setuptools wheel

# Python deps (scanpy pulls anndata/h5py/etc.)
# local-attention is typically published as "local-attention" on pip
RUN pip install --no-cache-dir \
    numpy pandas scipy tqdm pyyaml einops \
    scanpy \
    local-attention

# (Optional) if you hit issues with h5py wheels, uncomment:
# RUN pip install --no-cache-dir --upgrade "h5py>=3.10"

CMD ["bash"]