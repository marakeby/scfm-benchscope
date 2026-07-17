# Canonical GPU-capable runtime. The default image installs the core environment.
# Build another locked model environment with:
#   docker build --build-arg SCFM_PIXI_ENV=geneformer -t scfm-eval:geneformer .

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG PIXI_VERSION=0.66.0
ARG SCFM_PIXI_ENV=default

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIXI_HOME=/opt/pixi \
    PATH="/opt/pixi/bin:${PATH}" \
    SCFM_DATA_PATH=/data \
    SCFM_MODELS_PATH=/models \
    SCFM_OUTPUT_PATH=/output \
    SCFM_PIXI_ENV=${SCFM_PIXI_ENV}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

RUN curl -fsSL https://pixi.sh/install.sh \
    | PIXI_VERSION="v${PIXI_VERSION}" PIXI_HOME="${PIXI_HOME}" PIXI_NO_PATH_UPDATE=1 bash

WORKDIR /opt/scfm-eval

# The local editable package is part of the Pixi lock, so its metadata and source
# must be present before a frozen installation can be materialized.
COPY pyproject.toml README.md LICENSE pixi.toml pixi.lock ./
COPY src ./src
RUN pixi install --frozen -e "${SCFM_PIXI_ENV}" \
    && pixi clean cache -y

COPY . .

VOLUME ["/data", "/models", "/output"]

ENTRYPOINT ["bash", "/opt/scfm-eval/scripts/docker_entrypoint.sh"]
CMD ["--help"]
