#!/usr/bin/env bash
# SeuratDisk for SeuratObject 5+: upstream master still uses defunct GetAssayData(slot=…).
# This commit (PR #198) switches to `layer` — required for SaveH5Seurat / Convert with Seurat 5.
# https://github.com/mojaveazure/seurat-disk/pull/198
#
# Run from repo root:  bash scripts/install_seuratdisk.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# LinearParadox/seurat-disk @ 752413c (PR #198 head when tested)
SEURAT_DISK_REF="752413c0f32b0a3a483da3c24d583b3f699ce189"
TMP="${TMPDIR:-/tmp}/seurat-disk-${SEURAT_DISK_REF}.tar.gz"
curl -fsSL -o "$TMP" "https://codeload.github.com/LinearParadox/seurat-disk/tar.gz/${SEURAT_DISK_REF}"
cd "$ROOT"
exec pixi run -e r-seurat -- R CMD INSTALL "$TMP"
