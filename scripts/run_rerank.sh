#!/usr/bin/env bash
# Sinkhorn/Hungarian reranking on the SliceCLIP bi-encoder (prior baseline, 0.555).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO/core:$REPO/mind:$REPO/experiments:${PYTHONPATH:-}"
PY="${PY:-python}"
DATA="${DATA_ROOT:-${ROOT:-$HOME/medretrieval}/data}"
OUTDIR="${OUTDIR:-$REPO/runs/rerank_out}"; mkdir -p "$OUTDIR"
"$PY" "$REPO/core/rerank_baseline.py" \
  --data-root "$DATA" \
  --train-pair-csv "$DATA/dataset1/train_pairs.csv" \
  --query-csv "$DATA/dataset1/val_queries.csv"  --gallery-csv "$DATA/dataset1/val_gallery.csv" \
  --query-csv "$DATA/dataset1/test_queries.csv" --gallery-csv "$DATA/dataset1/test_gallery.csv" \
  --query-csv "$DATA/dataset2/val_queries.csv"  --gallery-csv "$DATA/dataset2/val_gallery.csv" \
  --query-csv "$DATA/dataset2/test_queries.csv" --gallery-csv "$DATA/dataset2/test_gallery.csv" \
  --query-csv "$DATA/dataset3/val_queries.csv"  --gallery-csv "$DATA/dataset3/val_gallery.csv" \
  --query-csv "$DATA/dataset3/test_queries.csv" --gallery-csv "$DATA/dataset3/test_gallery.csv" \
  --out-dir "$OUTDIR" --sinkhorn-tau 10.0 --sinkhorn-iter 50
