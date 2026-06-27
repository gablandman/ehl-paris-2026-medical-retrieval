#!/usr/bin/env bash
# [experiment, negative] Frozen BrainIAC encoder + bijection reranking.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO/core:$REPO/mind:$REPO/experiments:${PYTHONPATH:-}"
PY="${PY:-python}"
DATA="${DATA_ROOT:-${ROOT:-$HOME/medretrieval}/data}"
OUTDIR="${OUTDIR:-$REPO/runs/brainiac_out}"; mkdir -p "$OUTDIR"
"$PY" "$REPO/experiments/brainiac_encode.py" \
  --data-root "$DATA" \
  --query-csv "$DATA/dataset1/val_queries.csv"  --gallery-csv "$DATA/dataset1/val_gallery.csv" \
  --query-csv "$DATA/dataset1/test_queries.csv" --gallery-csv "$DATA/dataset1/test_gallery.csv" \
  --query-csv "$DATA/dataset2/val_queries.csv"  --gallery-csv "$DATA/dataset2/val_gallery.csv" \
  --query-csv "$DATA/dataset2/test_queries.csv" --gallery-csv "$DATA/dataset2/test_gallery.csv" \
  --query-csv "$DATA/dataset3/val_queries.csv"  --gallery-csv "$DATA/dataset3/val_gallery.csv" \
  --query-csv "$DATA/dataset3/test_queries.csv" --gallery-csv "$DATA/dataset3/test_gallery.csv" \
  --out-dir "$OUTDIR" --cache-dir "${CACHE_DIR:-$REPO/runs/.brainiac_cache}" \
  --sinkhorn-tau 10.0 --sinkhorn-iter 50
