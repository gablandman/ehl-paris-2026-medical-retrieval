from __future__ import annotations
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "monai>=1.5.0",
#   "torch>=2.7.0",
#   "numpy>=2.0",
#   "nibabel>=5.3",
#   "scipy>=1.13",
#   "tqdm>=4.67",
# ]
# ///

"""
SliceCLIP + dataset2-style geometric augmentation (GPU-batched).

WHY
---
The leaderboard score is the mean of dataset1/2/3 MRR. We are strong on
dataset1 (clean, registered, the only labelled set) but weak on dataset2/3.
The organisers tell us exactly how dataset2 was built:

    "validation and test images have random rigid rotation/translation and
     non-linear deformations applied. Query and target images in a correct
     pair are deformed INDEPENDENTLY, so they no longer share one geometry."

We reproduce that recipe as a TRAINING-TIME augmentation on the labelled
dataset1 pairs: take a true (query, target) pair and deform the query and the
target with two INDEPENDENT random rigid+elastic transforms, then ask the model
to still match them. This teaches invariance to exactly the distortions
dataset2 introduces, using the real labels we already have. It targets the weak
two-thirds of the score (dataset2, partially dataset3) without spending
submissions to tune.

DESIGN
------
- Deterministic preprocessing (load / RAS / 1mm spacing / scale to [0,1]) is
  cached by MONAI PersistentDataset as the FULL 3D volume (not the 3 slices),
  so the geometric augmentation is applied to the 3D volume BEFORE slice
  extraction -- matching dataset2 (deform in 3D, then the axial slice sees
  reoriented anatomy).
- All preprocessed volumes are uniform (1,240,240,155), so we load them ONCE
  into a GPU tensor bank and do the augmentation BATCHED ON GPU:
      1. 3D rigid: random rotation ~+-15deg + translation ~+-10 voxels via
         affine_grid + grid_sample (faithful: out-of-plane rotation changes
         which anatomy lands in the axial slices, like dataset2),
      2. occupancy-based 3-slice extraction at 35/50/65% depth + resize 96x96
         (identical to the baseline SliceStackd),
      3. 2D non-linear elastic on the 96x96 slice stack via a random low-res
         displacement field upsampled and applied with grid_sample (pragmatic
         in-plane elastic; the 3D rigid already supplies out-of-plane variation).
  Query and target are augmented with INDEPENDENT random parameters.

A synthetic-dataset2 validation harness holds out a fixed subset of the 350
labelled pairs, deforms query and target independently with the same recipe,
and computes MRR offline -- a faithful dataset2 proxy for tuning without
spending submissions.

INFERENCE uses NO augmentation and reproduces the baseline slice extraction
exactly (verified bit-identical), so embeddings are comparable to the existing
pipeline and the Sinkhorn rerank stacks unchanged.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.data import PersistentDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    Spacingd,
)
from tqdm import tqdm

from slice_clip_baseline import (
    CONFIG,
    SliceCLIP,
    collect_prediction_sets,
    load_training_pairs,
    rank_targets,
    write_submission,
)

# ---------------------------------------------------------------------------
# Augmentation config (tunable strength). Defaults roughly match dataset2's
# geometric variation without destroying the matching signal.
#
# TUNING (offline hold-out sweep over --aug-prob, 300 epochs, 50-pair hold-out;
# clean = no-deform MRR ~ dataset1 proxy, synth = independently-deformed MRR ~
# dataset2 proxy):
#     aug_prob | clean | synth   <- always-on deformation over-regularises
#       0.00   | 0.671 | 0.129       and collapses dataset1; a light mix is best.
#       0.25   | 0.792 | 0.307   <- BEST on BOTH proxies (chosen default)
#       0.40   | 0.613 | 0.306
# An early always-on run (aug_prob=1.0) scored 0.401 on the leaderboard vs the
# 0.555 baseline because it crushed dataset1 (ds1 MRR 0.90 -> 0.50). Mixing
# clean+deformed at aug_prob=0.25 keeps dataset1 strong while gaining on ds2/3.
# ---------------------------------------------------------------------------
AUG = {
    "rotate_deg": 15.0,          # +-15 deg about each axis (3D, before slicing)
    "translate_frac": 0.08,      # +- fraction of axis length (~+-10-19 vox)
    "elastic_prob": 0.8,         # in-plane non-linear elastic on the 96x96 stack
    "elastic_ctrl": 5,           # control-grid size for the displacement field
    "elastic_strength": 0.10,    # max displacement as fraction of image (normalised coords)
}

SLICE_POSITIONS = tuple(CONFIG["slice_positions"])
IMAGE_SIZE = int(CONFIG["image_size"])


# ---------------------------------------------------------------------------
# Deterministic preprocessing -> cached full 3D volume.
# ---------------------------------------------------------------------------
def volume_transform() -> Compose:
    return Compose(
        [
            LoadImaged(keys="image", image_only=True),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS", labels=None),
            Spacingd(keys="image", pixdim=CONFIG["spacing_mm"], mode="bilinear"),
            ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
            EnsureTyped(keys="image"),
        ]
    )


def make_volume_dataset(images: dict[str, Path], example_root: Path) -> PersistentDataset:
    cache_dir = example_root / (str(CONFIG["cache_dir"]) + "_vol")
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"image": str(path), "id": image_id} for image_id, path in sorted(images.items())]
    return PersistentDataset(data=rows, transform=volume_transform(), cache_dir=cache_dir)


def _pad_or_crop(v: torch.Tensor, target) -> torch.Tensor:
    """Center pad-with-zeros or crop a (1,H,W,D) volume to `target` (Ht,Wt,Dt).

    Zero-padding is safe: occupancy-based slice selection keys off the non-zero
    z-extent, which padding does not change.
    """
    _, H, W, D = v.shape
    Ht, Wt, Dt = target
    out = torch.zeros((1, Ht, Wt, Dt), dtype=v.dtype)
    # source crop window (centered) and destination paste window (centered)
    def span(s, t):
        if s >= t:
            o = (s - t) // 2
            return slice(o, o + t), slice(0, t)
        else:
            o = (t - s) // 2
            return slice(0, s), slice(o, o + s)
    sh, dh = span(H, Ht)
    sw, dw = span(W, Wt)
    sd, dd = span(D, Dt)
    out[:, dh, dw, dd] = v[:, sh, sw, sd]
    return out


def load_volume_bank(volume_dataset: PersistentDataset, ids: list[str], device):
    """Load preprocessed volumes for `ids` into one (N,1,H,W,D) tensor.

    Volumes differ in shape across datasets (dataset2/3 were reshaped by the
    geometric construction), so we center pad/crop each to a fixed canvas that
    holds the largest volume. This lets the augmentation run BATCHED on GPU.
    """
    id_to_index = {row["id"]: i for i, row in enumerate(volume_dataset.data)}
    raw = []
    maxshape = [0, 0, 0]
    for iid in tqdm(ids, desc="Caching/loading volumes"):
        item = volume_dataset[id_to_index[iid]]
        v = torch.as_tensor(item["image"]).float().cpu()
        if v.ndim == 3:
            v = v[None]
        raw.append(v)
        for k in range(3):
            maxshape[k] = max(maxshape[k], v.shape[k + 1])
    target = tuple(maxshape)
    print(f"volume canvas (pad/crop target): {target}", flush=True)
    vols = [_pad_or_crop(v, target) for v in raw]
    bank = torch.stack(vols).to(device)  # (N,1,H,W,D)
    pos = {iid: i for i, iid in enumerate(ids)}
    return bank, pos


# ---------------------------------------------------------------------------
# GPU-batched augmentation.
# ---------------------------------------------------------------------------
def _occupancy_zslices(vol_batch: torch.Tensor) -> torch.Tensor:
    """vol_batch: (B,1,H,W,D) -> integer z-indices (B,3) at 35/50/65% occupancy.

    Occupancy is measured per-volume on the (deformed) tensor, matching the
    baseline which keys slice positions off the non-zero z-extent.
    """
    B, _, H, W, D = vol_batch.shape
    nz = (vol_batch[:, 0].abs() > 1e-6).any(dim=1).any(dim=1)  # (B, D) any nonzero in slice
    z_idx = torch.zeros(B, len(SLICE_POSITIONS), dtype=torch.long, device=vol_batch.device)
    for b in range(B):
        occ = torch.nonzero(nz[b], as_tuple=False).flatten()
        if len(occ) == 0:
            zmin, zmax = D // 2, D // 2
        else:
            zmin, zmax = int(occ[0]), int(occ[-1])
        for k, p in enumerate(SLICE_POSITIONS):
            z_idx[b, k] = int(round(zmin + p * (zmax - zmin)))
    return z_idx.clamp_(0, D - 1)


def _rand_rigid_grid(B: int, shape, device, gen):
    """Random 3D rigid (rotation+translation) sampling grid for grid_sample.

    Returns an affine_grid of shape (B,H,W,D,3). Rotations are composed about
    the three axes; translation is in normalised [-1,1] coords.
    """
    H, W, D = shape
    rot = (torch.rand(B, 3, device=device, generator=gen) * 2 - 1) * (AUG["rotate_deg"] * np.pi / 180.0)
    # translation in normalised coords (full extent maps to [-1,1] => *2)
    trans = (torch.rand(B, 3, device=device, generator=gen) * 2 - 1) * (AUG["translate_frac"] * 2.0)

    cz, sz = torch.cos(rot[:, 0]), torch.sin(rot[:, 0])
    cy, sy = torch.cos(rot[:, 1]), torch.sin(rot[:, 1])
    cx, sx = torch.cos(rot[:, 2]), torch.sin(rot[:, 2])
    zero = torch.zeros(B, device=device)
    one = torch.ones(B, device=device)

    def stack3(r0, r1, r2):
        return torch.stack([torch.stack(r0, -1), torch.stack(r1, -1), torch.stack(r2, -1)], 1)

    Rz = stack3([cz, -sz, zero], [sz, cz, zero], [zero, zero, one])
    Ry = stack3([cy, zero, sy], [zero, one, zero], [-sy, zero, cy])
    Rx = stack3([one, zero, zero], [zero, cx, -sx], [zero, sx, cx])
    R = Rz @ Ry @ Rx  # (B,3,3)
    theta = torch.cat([R, trans.unsqueeze(-1)], dim=-1)  # (B,3,4)
    grid = F.affine_grid(theta, (B, 1, H, W, D), align_corners=False)
    return grid


def _rand_elastic_grid(B: int, size: int, device, gen):
    """Random in-plane elastic deformation grid (B,size,size,2) for grid_sample."""
    c = AUG["elastic_ctrl"]
    disp = (torch.rand(B, 2, c, c, device=device, generator=gen) * 2 - 1) * AUG["elastic_strength"]
    disp = F.interpolate(disp, size=(size, size), mode="bilinear", align_corners=True)  # (B,2,size,size)
    base = F.affine_grid(
        torch.eye(2, 3, device=device).unsqueeze(0).repeat(B, 1, 1), (B, 1, size, size), align_corners=False
    )  # (B,size,size,2)
    return base + disp.permute(0, 2, 3, 1)


def augment_batch(vols: torch.Tensor, gen, enabled: bool, aug_prob: float = 1.0) -> torch.Tensor:
    """vols: (B,1,H,W,D) -> (B,3,IMAGE_SIZE,IMAGE_SIZE) augmented slice stacks.

    With enabled=False this reproduces the baseline deterministic slice stack
    bit-for-bit (no rigid, no elastic), so inference matches the original model.

    `aug_prob` (only when enabled) is the PER-ITEM probability of being deformed.
    aug_prob<1 mixes clean and deformed examples so the model stays strong on
    clean (dataset1) inputs while learning invariance for deformed (dataset2)
    ones. Each item draws its own rigid AND its own clean/deform coin, so the
    query and target of a pair are deformed independently.
    """
    B = vols.shape[0]
    if enabled:
        grid = _rand_rigid_grid(B, vols.shape[2:], vols.device, gen)
        warped = F.grid_sample(vols, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        if aug_prob < 1.0:
            do_rigid = (torch.rand(B, device=vols.device, generator=gen) < aug_prob).view(B, 1, 1, 1, 1)
            vols = torch.where(do_rigid, warped, vols)
        else:
            vols = warped

    z_idx = _occupancy_zslices(vols)  # (B,3)
    # gather the three z-slices per volume -> (B,3,H,W); vols[:,0] is (B,H,W,D)
    v0 = vols[:, 0]
    out = []
    for k in range(len(SLICE_POSITIONS)):
        zk = z_idx[:, k].view(B, 1, 1, 1).expand(B, v0.shape[1], v0.shape[2], 1)
        out.append(torch.gather(v0, 3, zk).squeeze(-1))  # (B,H,W)
    stack = torch.stack(out, dim=1)  # (B,3,H,W)
    stack = torch.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
    stack = F.interpolate(stack, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)

    if enabled and AUG["elastic_prob"] > 0:
        do = (torch.rand(B, device=vols.device, generator=gen) < AUG["elastic_prob"]) & (
            torch.rand(B, device=vols.device, generator=gen) < aug_prob
        )
        if do.any():
            egrid = _rand_elastic_grid(B, IMAGE_SIZE, vols.device, gen)
            warped = F.grid_sample(stack, egrid, mode="bilinear", padding_mode="zeros", align_corners=False)
            stack = torch.where(do.view(B, 1, 1, 1), warped, stack)
    return stack


# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------
def train_augmented(model, bank, pos, train_pairs, epochs, augment, device, gen, aug_prob=1.0):
    """In-batch contrastive training; augmentation is GPU-batched per step."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(CONFIG["learning_rate"]))
    loss_fn = nn.CrossEntropyLoss()
    bs = int(CONFIG["batch_size"])
    q_idx = torch.tensor([pos[str(p["query_id"])] for p in train_pairs], device=device)
    t_idx = torch.tensor([pos[str(p["target_id"])] for p in train_pairs], device=device)
    n = len(train_pairs)
    model.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=device, generator=gen)
        total_loss, seen = 0.0, 0
        for s in range(0, n, bs):
            bidx = perm[s : s + bs]
            qv = bank[q_idx[bidx]]
            tv = bank[t_idx[bidx]]
            # independent augmentations for query and target
            q_stack = augment_batch(qv, gen, augment, aug_prob)
            t_stack = augment_batch(tv, gen, augment, aug_prob)
            labels = torch.arange(len(bidx), device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(q_stack, t_stack)
            loss = (loss_fn(logits, labels) + loss_fn(logits.T, labels)) / 2
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(CONFIG["max_grad_norm"]))
            optimizer.step()
            total_loss += float(loss.item()) * len(bidx)
            seen += len(bidx)
        if epoch % 25 == 0 or epoch in (1, epochs):
            print(f"epoch {epoch:04d} loss={total_loss / max(seen,1):.5f}", flush=True)
    return model


@torch.no_grad()
def synthetic_ds2_mrr(model, bank, pos, holdout_pairs, device, n_repeat=5, seed=12345):
    """Deform held-out query/target independently, embed, rank, average MRR."""
    model.eval()
    qids = [str(p["query_id"]) for p in holdout_pairs]
    tids = [str(p["target_id"]) for p in holdout_pairs]
    qv = bank[torch.tensor([pos[q] for q in qids], device=device)]
    tv = bank[torch.tensor([pos[t] for t in tids], device=device)]
    rrs = []
    for rep in range(n_repeat):
        g = torch.Generator(device=device).manual_seed(seed + rep)
        q_stack = augment_batch(qv, g, True)
        t_stack = augment_batch(tv, g, True)
        q_emb = model.encode_query(q_stack)
        t_emb = model.encode_target(t_stack)
        S = (q_emb @ t_emb.T).cpu().numpy()  # (N,N), gold is the diagonal
        order = np.argsort(-S, axis=1)
        for i in range(len(qids)):
            rank = int(np.where(order[i] == i)[0][0]) + 1
            rrs.append(1.0 / rank)
    return float(np.mean(rrs))


@torch.no_grad()
def clean_holdout_mrr(model, bank, pos, holdout_pairs, device):
    """MRR on the held-out pairs with NO deformation -- a dataset1 (clean) proxy.

    Guards against augmentation destroying performance on clean inputs.
    """
    model.eval()
    qids = [str(p["query_id"]) for p in holdout_pairs]
    tids = [str(p["target_id"]) for p in holdout_pairs]
    qv = bank[torch.tensor([pos[q] for q in qids], device=device)]
    tv = bank[torch.tensor([pos[t] for t in tids], device=device)]
    q_emb = model.encode_query(augment_batch(qv, None, enabled=False))
    t_emb = model.encode_target(augment_batch(tv, None, enabled=False))
    S = (q_emb @ t_emb.T).cpu().numpy()
    order = np.argsort(-S, axis=1)
    rrs = [1.0 / (int(np.where(order[i] == i)[0][0]) + 1) for i in range(len(qids))]
    return float(np.mean(rrs))


@torch.no_grad()
def embed_clean(model, volume_dataset, ids, device, encoder):
    """Embed ids with NO augmentation, at each volume's NATIVE shape.

    Inference pools (dataset2/3) have heterogeneous volume shapes, so we slice
    each volume at its own shape (one at a time) -- this reproduces the baseline
    slice extraction exactly (verified bit-identical for uniform volumes) and is
    safe for variable shapes since no 3D batching is needed without augmentation.
    """
    model.eval()
    id_to_index = {row["id"]: i for i, row in enumerate(volume_dataset.data)}
    out = {}
    for iid in tqdm(ids, desc=f"embed-{encoder}"):
        item = volume_dataset[id_to_index[iid]]
        v = torch.as_tensor(item["image"]).float()
        if v.ndim == 3:
            v = v[None]
        v = v[None].to(device)  # (1,1,H,W,D)
        stack = augment_batch(v, None, enabled=False)  # (1,3,96,96)
        emb = (model.encode_query(stack) if encoder == "query" else model.encode_target(stack))
        out[iid] = emb[0].cpu().numpy().astype(np.float32)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SliceCLIP with dataset2-style GPU-batched geometric augmentation.")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--train-pair-csv", type=Path, action="append", default=[])
    p.add_argument("--query-csv", type=Path, action="append", default=[])
    p.add_argument("--gallery-csv", type=Path, action="append", default=[])
    p.add_argument("--out", type=Path, default=Path("ds2aug_submission.csv"))
    p.add_argument("--ckpt", type=Path, default=Path("ds2aug_out/model.pt"))
    p.add_argument("--epochs", type=int, default=int(CONFIG["epochs"]))
    p.add_argument("--holdout", type=int, default=50)
    p.add_argument("--aug-prob", type=float, default=0.25,
                   help="per-item probability of deformation; <1 mixes clean+deformed so "
                        "dataset1 (clean) stays strong while dataset2 (deformed) improves. "
                        "0.25 was the sweep optimum (best on BOTH a clean and a synthetic-ds2 "
                        "hold-out proxy); always-on (1.0) over-regularises and hurts dataset1.")
    p.add_argument("--eval-baseline", action="store_true",
                   help="also train a no-aug model on the same split and report its synthetic-ds2 MRR")
    p.add_argument("--no-train", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(CONFIG["seed"]))
    np.random.seed(int(CONFIG["seed"]))
    torch.manual_seed(int(CONFIG["seed"]))

    example_root = Path(__file__).resolve().parent
    data_root = args.data_root.resolve()
    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(CONFIG["device"]))

    all_pairs = load_training_pairs(data_root, args.train_pair_csv, [], [], [])
    rng = random.Random(int(CONFIG["seed"]))
    indices = list(range(len(all_pairs)))
    rng.shuffle(indices)
    holdout_idx = set(indices[: args.holdout])
    holdout_pairs = [all_pairs[i] for i in sorted(holdout_idx)]
    train_pairs = [all_pairs[i] for i in indices if i not in holdout_idx]
    print(f"pairs: total={len(all_pairs)} train={len(train_pairs)} holdout={len(holdout_pairs)}", flush=True)

    prediction_sets = collect_prediction_sets(data_root, args.query_csv, args.gallery_csv)

    images: dict[str, Path] = {}
    for p in all_pairs:
        images[str(p["query_id"])] = Path(p["query_path"])
        images[str(p["target_id"])] = Path(p["target_path"])
    for ps in prediction_sets:
        images.update(ps["queries"])
        images.update(ps["targets"])

    print(json.dumps({
        "aug": AUG,
        "epochs": args.epochs,
        "num_images": len(images),
        "num_prediction_sets": len(prediction_sets),
        "device": str(device),
    }, indent=2), flush=True)

    volume_dataset = make_volume_dataset(images, example_root)
    # Bank ONLY the labelled dataset1 images used for training+holdout: these are
    # uniform shape, so they batch on GPU for the augmentation. Inference pools
    # (heterogeneous shapes) are embedded per-volume at native shape below.
    train_ids = sorted({str(p["query_id"]) for p in all_pairs} | {str(p["target_id"]) for p in all_pairs})
    bank, pos = load_volume_bank(volume_dataset, train_ids, device)
    print(f"volume bank: {tuple(bank.shape)} on {bank.device}", flush=True)

    if args.eval_baseline:
        print("\n[baseline] training NO-augmentation model on the same split...", flush=True)
        gb = torch.Generator(device=device).manual_seed(int(CONFIG["seed"]))
        base_model = SliceCLIP(int(CONFIG["embedding_dim"])).to(device)
        train_augmented(base_model, bank, pos, train_pairs, args.epochs, augment=False, device=device, gen=gb)
        bm = synthetic_ds2_mrr(base_model, bank, pos, holdout_pairs, device)
        bc = clean_holdout_mrr(base_model, bank, pos, holdout_pairs, device)
        print(f"[baseline] clean-holdout MRR = {bc:.4f}  synthetic-ds2 MRR = {bm:.4f}", flush=True)

    if args.no_train and args.ckpt.exists():
        print(f"\nLoading augmented model from {args.ckpt}", flush=True)
        model = SliceCLIP(int(CONFIG["embedding_dim"])).to(device)
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
    else:
        print(f"\n[augmented] training WITH ds2-style augmentation "
              f"(aug_prob={args.aug_prob}) for {args.epochs} epochs...", flush=True)
        ga = torch.Generator(device=device).manual_seed(int(CONFIG["seed"]))
        model = SliceCLIP(int(CONFIG["embedding_dim"])).to(device)
        train_augmented(model, bank, pos, train_pairs, args.epochs, augment=True,
                        device=device, gen=ga, aug_prob=args.aug_prob)
        torch.save(model.state_dict(), args.ckpt)
        print(f"Saved checkpoint -> {args.ckpt}", flush=True)

    ac = clean_holdout_mrr(model, bank, pos, holdout_pairs, device)
    print(f"[augmented] clean-holdout MRR = {ac:.4f}", flush=True)
    am = synthetic_ds2_mrr(model, bank, pos, holdout_pairs, device)
    print(f"[augmented] synthetic-ds2 MRR = {am:.4f}", flush=True)

    print("\nEmbedding inference pools (no augmentation)...", flush=True)
    inf_ids = sorted({iid for ps in prediction_sets for iid in {**ps["queries"], **ps["targets"]}})
    q_emb_all = embed_clean(model, volume_dataset, inf_ids, device, "query")
    t_emb_all = embed_clean(model, volume_dataset, inf_ids, device, "target")

    rows = []
    for ps in prediction_sets:
        q_emb = {i: q_emb_all[i] for i in ps["queries"]}
        t_emb = {i: t_emb_all[i] for i in ps["targets"]}
        rows.extend(rank_targets(model, q_emb, t_emb))
    write_submission(args.out, rows)
    print(f"Wrote {len(rows)} rows to {args.out}", flush=True)
    np.savez(args.ckpt.parent / "embeddings.npz",
             **{f"q::{k}": v for k, v in q_emb_all.items()},
             **{f"t::{k}": v for k, v in t_emb_all.items()})
    print(f"Saved embeddings -> {args.ckpt.parent / 'embeddings.npz'}", flush=True)


if __name__ == "__main__":
    main()
