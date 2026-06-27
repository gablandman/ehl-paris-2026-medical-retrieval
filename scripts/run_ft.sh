#!/usr/bin/env bash
# [experiment, negative] Fine-tune BrainIAC (adapter head / LoRA). Usage: run_ft.sh <mode> [extra]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO/core:$REPO/mind:$REPO/experiments:${PYTHONPATH:-}"
PY="${PY:-python}"
DATA="${DATA_ROOT:-${ROOT:-$HOME/medretrieval}/data}"
OUTDIR="${OUTDIR:-$REPO/runs/ft_out}"; mkdir -p "$OUTDIR"
MODE="${1:-head}"
"$PY" "$REPO/experiments/brainiac_finetune.py" \
  --mode "$MODE" --data-root "$DATA" \
  --train-pair-csv "$DATA/dataset1/train_pairs.csv" \
  --query-csv "$DATA/dataset1/val_queries.csv"  --gallery-csv "$DATA/dataset1/val_gallery.csv" \
  --query-csv "$DATA/dataset1/test_queries.csv" --gallery-csv "$DATA/dataset1/test_gallery.csv" \
  --query-csv "$DATA/dataset2/val_queries.csv"  --gallery-csv "$DATA/dataset2/val_gallery.csv" \
  --query-csv "$DATA/dataset2/test_queries.csv" --gallery-csv "$DATA/dataset2/test_gallery.csv" \
  --query-csv "$DATA/dataset3/val_queries.csv"  --gallery-csv "$DATA/dataset3/val_gallery.csv" \
  --query-csv "$DATA/dataset3/test_queries.csv" --gallery-csv "$DATA/dataset3/test_gallery.csv" \
  --out-dir "$OUTDIR" "${@:2}"
