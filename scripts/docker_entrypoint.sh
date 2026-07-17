#!/usr/bin/env bash
set -euo pipefail

cd "${SCFM_APP_ROOT:-/opt/scfm-eval}"

if [[ $# -eq 0 ]]; then
  set -- --help
fi

exec pixi run -e "${SCFM_PIXI_ENV:-default}" scfm-eval "$@"
