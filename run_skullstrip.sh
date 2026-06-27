#!/usr/bin/env bash
# Skull-strip every query+gallery volume with HD-BET into a parallel data tree.
#
# Why: BrainIAC was pretrained on skull-stripped, MNI-registered, N4-corrected
# brains. Our raw volumes still carry skull/neck/eyes, so the frozen encoder is
# off-distribution. Stripping non-brain tissue first should bring the volumes
# back on-distribution before embedding.
#
# The output mirrors the originals' relative paths/filenames under STRIP_ROOT,
# so the existing manifests resolve unchanged. To embed the stripped volumes,
# re-run the frozen BrainIAC encoder (run_brainiac.sh, on the brainiac-encoder
# branch) with DATA_ROOT pointed at STRIP_ROOT:
#
#   DATA_ROOT="$STRIP_ROOT" ./run_brainiac.sh
#
# HD-BET install (once):  uv pip install --python "$PYTHON" HD-BET
#
# Override paths with DATA_ROOT / STRIP_ROOT / PYTHON / HDBET env vars.
set -euo pipefail
DR="${DATA_ROOT:-$HOME/medretrieval/data}"
SR="${STRIP_ROOT:-$HOME/medretrieval/data_stripped}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python}"
HDBET="${HDBET:-hd-bet}"

"$PYTHON" "$HERE/skullstrip.py" \
  --data-root "$DR" \
  --out-root "$SR" \
  --stage "${STRIP_STAGE:-$HOME/medretrieval/.strip_stage}" \
  --hdbet "$HDBET" --device "${HDBET_DEVICE:-cuda}"
