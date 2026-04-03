FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

# System deps (build tools + small set of libs commonly needed by scientific wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Pip sanity
RUN python -m pip install --upgrade pip wheel setuptools

# Install python deps
COPY requirements.scimilarity.txt /tmp/requirements.scimilarity.txt
RUN pip install --no-cache-dir -r /tmp/requirements.scimilarity.txt

# Install scimilarity (from PyPI by default)
RUN pip install --no-cache-dir scimilarity

# Smoke tests
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
RUN python -c "import scimilarity; print('scimilarity import OK')"

CMD ["/bin/bash"]