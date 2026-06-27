#!/bin/bash
MODE=$1
DR=$HOME/medretrieval/data
EXTRA=${@:2}
$HOME/venv/bin/python brainiac_finetune.py \
  --mode $MODE \
  --data-root "$DR" \
  --train-pair-csv "$DR/dataset1/train_pairs.csv" \
  --query-csv "$DR/dataset1/val_queries.csv"  --gallery-csv "$DR/dataset1/val_gallery.csv" \
  --query-csv "$DR/dataset1/test_queries.csv" --gallery-csv "$DR/dataset1/test_gallery.csv" \
  --query-csv "$DR/dataset2/val_queries.csv"  --gallery-csv "$DR/dataset2/val_gallery.csv" \
  --query-csv "$DR/dataset2/test_queries.csv" --gallery-csv "$DR/dataset2/test_gallery.csv" \
  --query-csv "$DR/dataset3/val_queries.csv"  --gallery-csv "$DR/dataset3/val_gallery.csv" \
  --query-csv "$DR/dataset3/test_queries.csv" --gallery-csv "$DR/dataset3/test_gallery.csv" \
  --out-dir "$HOME/medretrieval/ft_out" $EXTRA
