#!/usr/bin/env bash
# scGPT env: matplotlib/torch wheels use symbols from a newer libstdc++ than some
# host /lib/.../libstdc++.so.6; prefer conda-forge libs in this prefix.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
