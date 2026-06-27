#!/usr/bin/env bash
# Frozen BrainIAC retrieval + bijection reranking.
# Override data location with: DATA_ROOT=/path/to/data ./run_brainiac.sh
set -euo pipefail
DR="${DATA_ROOT:-$HOME/medretrieval/data}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python}"

"$PYTHON" "$HERE/brainiac_encode.py" \
  --data-root "$DR" \
  --query-csv "$DR/dataset1/val_queries.csv"  --gallery-csv "$DR/dataset1/val_gallery.csv" \
  --query-csv "$DR/dataset1/test_queries.csv" --gallery-csv "$DR/dataset1/test_gallery.csv" \
  --query-csv "$DR/dataset2/val_queries.csv"  --gallery-csv "$DR/dataset2/val_gallery.csv" \
  --query-csv "$DR/dataset2/test_queries.csv" --gallery-csv "$DR/dataset2/test_gallery.csv" \
  --query-csv "$DR/dataset3/val_queries.csv"  --gallery-csv "$DR/dataset3/val_gallery.csv" \
  --query-csv "$DR/dataset3/test_queries.csv" --gallery-csv "$DR/dataset3/test_gallery.csv" \
  --out-dir "$HERE/brainiac_out" \
  --cache-dir "$HOME/medretrieval/.brainiac_cache" \
  --sinkhorn-tau 10.0 --sinkhorn-iter 50
