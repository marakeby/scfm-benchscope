FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN python -m pip install --upgrade pip setuptools wheel

# (Optional) Install your pinned deps first (faster rebuilds if repo changes)
COPY requirements.geneformer.txt /tmp/requirements.geneformer.txt
RUN pip install --no-cache-dir -r /tmp/requirements.geneformer.txt

# Geneformer install per docs: git-lfs + clone + pip install .
RUN git lfs install \
 && git clone https://huggingface.co/ctheodoris/Geneformer /opt/Geneformer \
 && pip install --no-cache-dir /opt/Geneformer

# Smoke test
RUN python -c "import geneformer; print('geneformer import OK')"

CMD ["/bin/bash"]