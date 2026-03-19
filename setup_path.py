"""Central path configuration. Loads .env from repo root so all paths can use env vars."""

import os
from os.path import join, realpath, dirname
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

BASE_PATH = dirname(realpath(__file__))

# Paths (override via .env or environment)
PARAMS_PATH = os.environ.get("SCFM_EVAL_PARAMS_PATH", join(BASE_PATH, "yaml"))
RUN_PATH = join(BASE_PATH, "run")

# Dataset path: .h5ad paths in YAML are relative to this
DATA_PATH = os.environ["DATASETS_PATH"]
# Model checkpoints: each model has its own subdir (STACK, scGPT, omnicell, etc.)
CHECKPOINTS_BASE = Path(os.environ["MODEL_CHECKPOINTS"])

# All results saved to this central location
OUTPUT_PATH = Path(os.environ["OUTPUT_PATH"])
EMBEDDINGS_PATH = OUTPUT_PATH / "embeddings"

# Custom temp directory on network storage (avoids filling local /tmp)
# This is used for large temporary files (h5ad, embeddings, etc.)
TEMP_PATH = OUTPUT_PATH / ".temp"
TEMP_PATH.mkdir(parents=True, exist_ok=True)