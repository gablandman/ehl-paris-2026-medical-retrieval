#!/usr/bin/env bash
# [experiment, superseded] SliceCLIP with ds2-style rigid+elastic augmentation.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO/core:$REPO/mind:$REPO/experiments:${PYTHONPATH:-}"
PY="${PY:-python}"
DATA="${DATA_ROOT:-${ROOT:-$HOME/medretrieval}/data}"
OUTDIR="${OUTDIR:-$REPO/runs/ds2aug_out}"; mkdir -p "$OUTDIR"
PYTHONUNBUFFERED=1 "$PY" -u "$REPO/experiments/slice_clip_ds2aug.py" \
  --data-root "$DATA" \
  --train-pair-csv "$DATA/dataset1/train_pairs.csv" \
  --query-csv "$DATA/dataset1/val_queries.csv"  --gallery-csv "$DATA/dataset1/val_gallery.csv" \
  --query-csv "$DATA/dataset1/test_queries.csv" --gallery-csv "$DATA/dataset1/test_gallery.csv" \
  --query-csv "$DATA/dataset2/val_queries.csv"  --gallery-csv "$DATA/dataset2/val_gallery.csv" \
  --query-csv "$DATA/dataset2/test_queries.csv" --gallery-csv "$DATA/dataset2/test_gallery.csv" \
  --query-csv "$DATA/dataset3/val_queries.csv"  --gallery-csv "$DATA/dataset3/val_gallery.csv" \
  --query-csv "$DATA/dataset3/test_queries.csv" --gallery-csv "$DATA/dataset3/test_gallery.csv" \
  --epochs "${EPOCHS:-500}" --holdout 50 --eval-baseline \
  --out "$OUTDIR/ds2aug_submission.csv" --ckpt "$OUTDIR/model.pt"
