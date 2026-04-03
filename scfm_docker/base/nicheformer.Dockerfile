FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NICHE_VENV=/opt/nicheformer-venv

# Install system deps needed for git installs (and common scientific packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl ca-certificates \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Create isolated venv
RUN python -m venv ${NICHE_VENV} \
 && ${NICHE_VENV}/bin/pip install --upgrade pip setuptools wheel

# Pin zarr to v2.x before anything pulls zarr v3
RUN ${NICHE_VENV}/bin/pip install \
    "numpy<2" \
    "zarr<3" \
    "anndata==0.10.8"

# Install nicheformer
RUN ${NICHE_VENV}/bin/pip install "git+https://github.com/theislab/nicheformer.git"

# Smoke test
RUN ${NICHE_VENV}/bin/python -c "import nicheformer; print('nicheformer ok')"

WORKDIR /workspace
CMD ["bash"]