FROM scfm-base:cu128

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Geneformer docs require git-lfs
RUN apt-get update && apt-get install -y --no-install-recommends \
    git-lfs \
 && rm -rf /var/lib/apt/lists/*

RUN git lfs install

# Clone official Geneformer source
RUN git clone https://huggingface.co/ctheodoris/Geneformer /opt/Geneformer
WORKDIR /opt/Geneformer

# Install Geneformer dependencies explicitly first.
# The project requirements file currently pins transformers,
# and the project has also documented breakage from newer transformers versions.
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt \
 && pip install --force-reinstall "transformers==4.49.0" \
 && pip install -e .

# Smoke test
RUN python -c "import geneformer; print('geneformer ok')"

WORKDIR /workspace
CMD ["bash"]