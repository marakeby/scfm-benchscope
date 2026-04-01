#!/usr/bin/env bash
# (3) Download or sync experiment datasets under DATA_ROOT (match setup_path.py + yaml dataset paths).
#
# Config:
#   cp scripts/config/data_download.env.example scripts/config/data_download.env
#   $EDITOR scripts/config/data_download.env
#   bash scripts/download_data.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${DATA_DOWNLOAD_CONFIG:-$ROOT/scripts/config/data_download.env}"
if [[ -f "$CONFIG" ]]; then
  echo "Loading $CONFIG"
  set -a
  # shellcheck source=/dev/null
  source "$CONFIG"
  set +a
else
  echo "No $CONFIG — using defaults. Copy scripts/config/data_download.env.example to data_download.env."
fi

DATA_ROOT="${DATA_ROOT:-$HOME}"
mkdir -p "$DATA_ROOT"

if [[ -n "${GCS_SYNC_URI:-}" ]]; then
  if ! command -v gsutil >/dev/null 2>&1; then
    echo "error: gsutil not found. Install Google Cloud SDK and run: gcloud auth login"
    exit 1
  fi
  DEST="${GCS_SYNC_DEST:-$DATA_ROOT/mnt/DATA/}"
  mkdir -p "$DEST"
  echo "gsutil rsync $GCS_SYNC_URI -> $DEST"
  gsutil -m rsync -r "$GCS_SYNC_URI" "$DEST"
fi

if [[ -n "${DATA_URL:-}" ]]; then
  DATA_DEST="${DATA_DEST:-$DATA_ROOT/mnt/DATA/downloaded.h5ad}"
  mkdir -p "$(dirname "$DATA_DEST")"
  echo "Downloading $DATA_URL -> $DATA_DEST"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 -o "$DATA_DEST" "$DATA_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$DATA_DEST" "$DATA_URL"
  else
    echo "error: need curl or wget"
    exit 1
  fi
fi

if [[ -n "${RSYNC_SRC:-}" ]]; then
  RSYNC_DEST="${RSYNC_DEST:-$DATA_ROOT/mnt/DATA/}"
  mkdir -p "$RSYNC_DEST"
  echo "rsync $RSYNC_SRC -> $RSYNC_DEST"
  rsync -aH --info=progress2 "$RSYNC_SRC" "$RSYNC_DEST"
fi

if [[ -z "${GCS_SYNC_URI:-}" ]] && [[ -z "${DATA_URL:-}" ]] && [[ -z "${RSYNC_SRC:-}" ]]; then
  echo "Nothing to do: set GCS_SYNC_URI, DATA_URL, or RSYNC_SRC in data_download.env"
  exit 0
fi

echo "download_data.sh done. Point setup_path.py DATA_PATH to: $DATA_ROOT"
