FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Make pip sane + keep pkg_resources available
RUN python -m pip install --upgrade pip wheel \
 && python -m pip install --no-cache-dir "setuptools<81"

# Install torchtext matching torch in the base image
RUN conda install -y -c pytorch torchtext=0.16.* && conda clean -afy

# Install scGPT runtime deps
COPY requirements.scgpt.txt /tmp/requirements.scgpt.txt
RUN pip install --no-cache-dir -r /tmp/requirements.scgpt.txt

# Smoke tests at build time
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
RUN python -c "import torchtext; print('torchtext', torchtext.__version__)"
RUN python -c "import scgpt; print('scgpt import OK')"
RUN python -c "import scvi; print('scvi-tools', scvi.__version__)"

CMD ["/bin/bash"]