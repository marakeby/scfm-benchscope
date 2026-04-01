# GPU-oriented image: CUDA runtime + Pixi + locked envs + optional Geneformer (HF clone).
#
# Build:  docker build -t scfm-eval .
# Run:    docker run --gpus all -v "$PWD":/workspace -w /workspace scfm-eval \
#           pixi run -e default run-exp yaml/examples/mock_subtype_eval.yaml
#
# Install NVIDIA Container Toolkit on the host so `--gpus all` works.
# Align CUDA major with the host driver if you see CUDA loader errors.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

RUN curl -fsSL https://pixi.sh/install.sh | bash

ENV PATH="/root/.pixi/bin:$PATH"

WORKDIR /workspace

# Cache Pixi solve layer
COPY pixi.toml pixi.lock ./
RUN pixi install --frozen

COPY . .

RUN bash scripts/install_packages.sh || echo "install_packages skipped — run inside container: pixi run install-packages"

CMD ["pixi", "run", "-e", "default", "run-exp", "yaml/examples/mock_subtype_eval.yaml"]
