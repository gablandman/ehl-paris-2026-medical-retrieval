#!/usr/bin/env bash
# [experiment, negative] Skull-strip every volume with HD-BET into a parallel data tree.
# HD-BET install (once): uv pip install --python "$PY" HD-BET
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO/core:$REPO/mind:$REPO/experiments:${PYTHONPATH:-}"
PY="${PY:-python}"
DATA="${DATA_ROOT:-${ROOT:-$HOME/medretrieval}/data}"
SR="${STRIP_ROOT:-${ROOT:-$HOME/medretrieval}/data_stripped}"
"$PY" "$REPO/experiments/skullstrip.py" \
  --data-root "$DATA" --out-root "$SR" \
  --stage "${STRIP_STAGE:-$REPO/runs/.strip_stage}" \
  --hdbet "${HDBET:-hd-bet}" --device "${HDBET_DEVICE:-cuda}"
