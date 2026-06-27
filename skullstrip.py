#!/usr/bin/env python
"""Skull-strip every query+gallery volume referenced by the retrieval CSVs.

Reads the 12 manifest CSVs, resolves each image path against --data-root (the
manifests list .nii.gz but the released volumes are uncompressed .nii, so we
fall back like the baseline does), runs HD-BET brain extraction once over a
flat staging folder (one model load), then writes the stripped volume into a
parallel tree under --out-root mirroring the original relative paths/filenames.
Existing CSVs therefore resolve unchanged against --out-root.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path


def resolve_image_path(data_root: Path, rel: str) -> Path:
    p = (data_root / rel)
    if p.exists():
        return p
    t = str(p)
    alt = Path(t[:-3]) if t.endswith(".nii.gz") else Path(t + ".gz")
    return alt if alt.exists() else p


def read_rows(csv_path: Path, id_col: str, img_col: str):
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            yield row[id_col], row[img_col]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--hdbet", type=str, default="hd-bet")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    out_root = args.out_root.resolve()
    stage_in = args.stage / "in"
    stage_out = args.stage / "out"
    stage_in.mkdir(parents=True, exist_ok=True)
    stage_out.mkdir(parents=True, exist_ok=True)

    csvs = []
    for ds in ("dataset1", "dataset2", "dataset3"):
        for split in ("val", "test"):
            csvs.append((data_root / ds / f"{split}_queries.csv", "query_id", "query_image"))
            csvs.append((data_root / ds / f"{split}_gallery.csv", "target_id", "target_image"))

    # Map: stage basename (safe, unique) -> (source path, dest path in mirror tree)
    plan: dict[str, tuple[Path, Path]] = {}
    for csv_path, id_col, img_col in csvs:
        for _id, rel in read_rows(csv_path, id_col, img_col):
            src = resolve_image_path(data_root, rel)
            if not src.exists():
                raise FileNotFoundError(f"missing source: {src} (from {csv_path})")
            # mirror dest keeps the manifest's relative path verbatim
            dest = out_root / rel
            # safe unique stage name from relative path
            key = rel.replace("/", "__")
            if not key.endswith(".nii.gz"):
                key = key + (".gz" if key.endswith(".nii") else ".nii.gz")
            plan[key] = (src, dest)

    print(f"unique volumes to strip: {len(plan)}", flush=True)

    # Stage inputs: HD-BET wants .nii.gz names; copy (contents may be plain .nii,
    # nibabel reads either). Skip if dest already done (resume support).
    todo = 0
    for key, (src, dest) in plan.items():
        if dest.exists() and dest.stat().st_size > 0:
            continue
        staged = stage_in / key
        if not (staged.exists() and staged.stat().st_size > 0):
            shutil.copy2(src, staged)
        todo += 1
    print(f"staged for stripping (not yet done): {todo}", flush=True)

    if todo > 0:
        cmd = [args.hdbet, "-i", str(stage_in), "-o", str(stage_out),
               "-device", args.device, "--disable_tta"]
        print("running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    # Distribute outputs to mirror tree.
    written = 0
    missing = []
    for key, (src, dest) in plan.items():
        if dest.exists() and dest.stat().st_size > 0:
            written += 1
            continue
        produced = stage_out / key  # HD-BET writes same basename as -o folder
        if not (produced.exists() and produced.stat().st_size > 0):
            missing.append(key)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, dest)
        written += 1

    print(f"written to mirror tree: {written}/{len(plan)}", flush=True)
    if missing:
        print(f"MISSING OUTPUTS ({len(missing)}): {missing[:10]}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
