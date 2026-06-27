from __future__ import annotations
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "monai>=1.5.0",
#   "torch>=2.7.0",
#   "numpy>=2.0",
#   "nibabel>=5.3",
#   "scipy>=1.11",
#   "tqdm>=4.67",
# ]
# ///

"""
MIND-based modality-invariant similarity for cross-modal brain-MRI retrieval.

This is a TRAINING-FREE retrieval signal. The idea (Heinrich et al., 2012,
"MIND: Modality independent neighbourhood descriptor"): describe each voxel by
the *local self-similarity* of its patch to a fixed neighbourhood. Because the
descriptor is built from within-image patch relationships, it is largely
invariant to the (non-functional) intensity mapping between T1 and T2, plus
robust to noise and bias fields. The query/gallery dissimilarity is then the
mean absolute difference of the two MIND descriptor fields.

Caveat: MIND dissimilarity is a *spatial* comparison; it assumes the two
volumes are roughly registered. That holds for dataset1 (common grid) and
dataset3 (intra-op resampled into the query space), but dataset2 applies an
independent rigid+elastic deformation to query vs target, breaking voxel
correspondence. We add an optional cheap center-of-mass translation pre-align.

Two entry points:

  * ``--mode offline``  -- validate on local proxies built from the 350
    dataset1 train pairs: a held-out aligned proxy (stands in for ds1/ds3) and
    a synthetic-ds2 proxy (held-out targets get an independent rigid+elastic
    deformation). Reports MIND-only and MIND+cosine MRR against the SliceCLIP
    cosine baseline.

  * ``--mode submit``   -- compute MIND dissimilarity for every real
    (dataset, split) query/gallery pool, optionally blend with the cached
    SliceCLIP cosine matrices, apply Sinkhorn reranking, and write a single
    submission CSV.

Reused helpers from the repo baseline:
  * slice_clip_baseline.collect_prediction_sets / load_training_pairs / read_csv
  * rerank_baseline.sinkhorn_log / rank_argmax / rank_sinkhorn / mrr /
    top1_conflict_rate / write_submission
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from slice_clip_baseline import (
    collect_prediction_sets,
    load_training_pairs,
)
from rerank_baseline import (
    mrr,
    rank_argmax,
    rank_sinkhorn,
    top1_conflict_rate,
    write_submission,
)


# --------------------------------------------------------------------------- #
# Volume loading (MONAI, mirrors slice_clip_baseline preprocessing choices)
# --------------------------------------------------------------------------- #

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    Spacingd,
)


def _loader() -> Compose:
    """RAS orient, 1mm spacing, per-volume intensity scaling to [0, 1].

    Deliberately the same deterministic preprocessing as the SliceCLIP
    baseline minus the slice extraction -- MIND wants the whole volume.
    """
    return Compose(
        [
            LoadImaged(keys="image", image_only=True),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS", labels=None),
            Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
            EnsureTyped(keys="image"),
        ]
    )


def load_volume(path: Path, transform: Compose, grid: int, device: torch.device) -> torch.Tensor:
    """Load one volume and resize it to a common ``grid^3`` cube on ``device``.

    Returns a float tensor of shape ``(grid, grid, grid)`` scaled to [0, 1].
    """
    out = transform({"image": str(path)})
    vol = torch.as_tensor(out["image"]).float()  # (1, X, Y, Z)
    vol = torch.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    vol = vol.unsqueeze(0).to(device)  # (1, 1, X, Y, Z)
    vol = F.interpolate(vol, size=(grid, grid, grid), mode="trilinear", align_corners=False)
    return vol[0, 0]


# --------------------------------------------------------------------------- #
# Synthetic deformations (for the ds1-holdout and synthetic-ds2 proxies)
# --------------------------------------------------------------------------- #


def _affine_grid_from_params(shape, angles, trans, device):
    """Build a normalized-coordinate affine sampling grid for a rigid transform."""
    ax, ay, az = angles
    Rx = torch.tensor(
        [[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]],
        dtype=torch.float32,
    )
    Ry = torch.tensor(
        [[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]],
        dtype=torch.float32,
    )
    Rz = torch.tensor(
        [[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]],
        dtype=torch.float32,
    )
    R = Rz @ Ry @ Rx
    theta = torch.zeros(1, 3, 4, dtype=torch.float32)
    theta[0, :, :3] = R
    theta[0, :, 3] = torch.tensor(trans, dtype=torch.float32)
    size = (1, 1, *shape)
    grid = F.affine_grid(theta.to(device), size, align_corners=False)
    return grid


def deform_volume(vol: torch.Tensor, rng: np.random.Generator,
                  rigid_deg: float, elastic_mag: float, elastic_sigma: float) -> torch.Tensor:
    """Apply an independent rigid + elastic deformation (the ds2 corruption model).

    rigid_deg : max rotation per axis (degrees); also drives a small translation.
    elastic_mag : std of the random displacement field, in normalized coords.
    elastic_sigma : Gaussian smoothing (voxels) of the displacement field.
    """
    device = vol.device
    shape = vol.shape  # (G, G, G)
    angles = [np.deg2rad(rng.uniform(-rigid_deg, rigid_deg)) for _ in range(3)]
    trans = [rng.uniform(-0.06, 0.06) for _ in range(3)]
    base = _affine_grid_from_params(shape, angles, trans, device)  # (1, G, G, G, 3)

    if elastic_mag > 0:
        disp = torch.from_numpy(
            rng.normal(0.0, elastic_mag, size=(1, *shape, 3)).astype(np.float32)
        ).to(device)
        # smooth the displacement field with a separable Gaussian
        disp = _gaussian_smooth_field(disp, elastic_sigma)
        grid = base + disp
    else:
        grid = base

    out = F.grid_sample(
        vol.view(1, 1, *shape), grid, mode="bilinear",
        padding_mode="zeros", align_corners=False,
    )
    return out[0, 0]


def _gaussian_smooth_field(field: torch.Tensor, sigma: float) -> torch.Tensor:
    """Smooth a (1, G, G, G, 3) displacement field with a separable 3D Gaussian."""
    if sigma <= 0:
        return field
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=field.device)
    k1 = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    k1 = k1 / k1.sum()
    # field: (1, G, G, G, 3) -> (3, 1, G, G, G)
    x = field[0].permute(3, 0, 1, 2).unsqueeze(1)
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = k1.numel()
        kernel = k1.view(shape)
        pad = [0, 0, 0, 0, 0, 0]
        pad[(2 - axis) * 2] = radius
        pad[(2 - axis) * 2 + 1] = radius
        x = F.pad(x, pad, mode="replicate")
        x = F.conv3d(x, kernel)
    return x.squeeze(1).permute(1, 2, 3, 0).unsqueeze(0)


# --------------------------------------------------------------------------- #
# MIND descriptor
# --------------------------------------------------------------------------- #

# 6-connected search neighbourhood (standard MIND-SSC uses 6; classic MIND uses 6).
_MIND_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]


def _box_kernel(radius: int, device: torch.device) -> torch.Tensor:
    """Normalized box patch kernel of side ``2*radius+1`` for patch-SSD via conv."""
    side = 2 * radius + 1
    k = torch.ones(1, 1, side, side, side, device=device)
    return k / k.numel()


def _shift(vol: torch.Tensor, off) -> torch.Tensor:
    """Shift a (1,1,X,Y,Z) volume by integer offset with zero padding (via roll+mask)."""
    dz = torch.roll(vol, shifts=(off[0], off[1], off[2]), dims=(2, 3, 4))
    # zero the wrapped-around border
    if off[0] > 0:
        dz[:, :, : off[0], :, :] = 0
    elif off[0] < 0:
        dz[:, :, off[0]:, :, :] = 0
    if off[1] > 0:
        dz[:, :, :, : off[1], :] = 0
    elif off[1] < 0:
        dz[:, :, :, off[1]:, :] = 0
    if off[2] > 0:
        dz[:, :, :, :, : off[2]] = 0
    elif off[2] < 0:
        dz[:, :, :, :, off[2]:] = 0
    return dz


def mind_descriptor(vol: torch.Tensor, patch_radius: int = 1, eps: float = 1e-5) -> torch.Tensor:
    """Compute the MIND descriptor field for a single ``(G, G, G)`` volume.

    Returns a tensor of shape ``(C, G, G, G)`` with ``C = len(_MIND_OFFSETS)``
    channels, each in (0, 1]. For every neighbourhood offset r:

        Dp(x, r) = box-convolution of (I - shift_r(I))^2     (Gaussian/box patch SSD)
        V(x)     = mean over r of Dp(x, r)                   (local variance estimate)
        MIND(x, r) = exp(-Dp(x, r) / V(x))

    then each voxel's descriptor vector is normalized so its max channel is 1.
    """
    device = vol.device
    v = vol.view(1, 1, *vol.shape)
    kernel = _box_kernel(patch_radius, device)
    pad = patch_radius

    dps = []
    for off in _MIND_OFFSETS:
        diff2 = (v - _shift(v, off)) ** 2
        dp = F.conv3d(F.pad(diff2, [pad] * 6, mode="replicate"), kernel)
        dps.append(dp)
    Dp = torch.cat(dps, dim=1)  # (1, C, G, G, G)

    var = Dp.mean(dim=1, keepdim=True)
    var = torch.clamp(var, min=eps)
    mind = torch.exp(-Dp / var)
    # normalize per voxel so the largest channel is 1 (standard MIND post-processing)
    mind = mind / torch.clamp(mind.max(dim=1, keepdim=True).values, min=eps)
    return mind[0]  # (C, G, G, G)


def center_of_mass_align(ref: torch.Tensor, mov: torch.Tensor) -> torch.Tensor:
    """Translate ``mov`` so its intensity center of mass matches ``ref`` (cheap pre-align)."""
    def com(x):
        total = x.sum()
        if total <= 0:
            g = x.shape[0]
            return torch.tensor([(g - 1) / 2.0] * 3, device=x.device)
        idx = [torch.arange(s, device=x.device, dtype=torch.float32) for s in x.shape]
        cx = (x.sum(dim=(1, 2)) * idx[0]).sum() / total
        cy = (x.sum(dim=(0, 2)) * idx[1]).sum() / total
        cz = (x.sum(dim=(0, 1)) * idx[2]).sum() / total
        return torch.stack([cx, cy, cz])

    shift = (com(ref) - com(mov)).round().to(torch.int64).tolist()
    out = torch.roll(mov, shifts=tuple(shift), dims=(0, 1, 2))
    return out


# --------------------------------------------------------------------------- #
# MIND dissimilarity matrix
# --------------------------------------------------------------------------- #


def mind_fields(paths: dict[str, Path], transform, grid, device, patch_radius):
    """Compute and cache MIND descriptor fields for a set of id->path images."""
    fields: dict[str, torch.Tensor] = {}
    vols: dict[str, torch.Tensor] = {}
    for img_id, path in paths.items():
        vol = load_volume(path, transform, grid, device)
        vols[img_id] = vol
        fields[img_id] = mind_descriptor(vol, patch_radius=patch_radius)
    return fields, vols


def mind_sim_matrix(q_fields, q_vols, t_fields, t_vols, align=False, patch_radius=1):
    """Negative mean |MIND_q - MIND_t| over voxels+channels => (Q, T) similarity.

    Higher = more similar. With ``align`` we COM-align the target volume to the
    query and recompute its descriptor on the fly (used for the ds2 case).
    """
    qids = sorted(q_fields)
    tids = sorted(t_fields)
    S = np.zeros((len(qids), len(tids)), dtype=np.float32)
    if not align:
        # Vectorize over targets: stack target fields once, flatten descriptors,
        # and compute mean|q - t| for all targets at once. The mean over the
        # flat descriptor is exactly the mean over voxels+channels.
        T = torch.stack([t_fields[t].reshape(-1) for t in tids])  # (Tn, D)
        for i, q in enumerate(qids):
            qf = q_fields[q].reshape(-1)  # (D,)
            d = (T - qf.unsqueeze(0)).abs().mean(dim=1)  # (Tn,)
            S[i] = -d.detach().cpu().numpy().astype(np.float32)
    else:
        # COM-align each target to each query and recompute its descriptor.
        for i, q in enumerate(qids):
            qf = q_fields[q].reshape(-1)
            for j, t in enumerate(tids):
                aligned = center_of_mass_align(q_vols[q], t_vols[t])
                tf = mind_descriptor(aligned, patch_radius=patch_radius).reshape(-1)
                S[i, j] = -float((qf - tf).abs().mean().item())
    return S, qids, tids


# --------------------------------------------------------------------------- #
# Blending helpers
# --------------------------------------------------------------------------- #


def slice_clip_cosine(ckpt: Path, q_paths, t_paths, device):
    """Embed the proxy images with a trained SliceCLIP checkpoint -> cosine (Q,T).

    Reuses the exact SliceCLIP model + MONAI image pipeline from the baseline so
    the offline cosine number is comparable to the real prediction-pool cosine.
    """
    from slice_clip_baseline import (
        CONFIG,
        SliceCLIP,
        embed_images,
        make_image_dataset,
    )

    example_root = Path(__file__).resolve().parent
    model = SliceCLIP(int(CONFIG["embedding_dim"])).to(device)
    model.load_state_dict(torch.load(str(ckpt), map_location=device))

    q_ds = make_image_dataset(q_paths, example_root, augment=False)
    t_ds = make_image_dataset(t_paths, example_root, augment=False)
    q_emb = embed_images(model, q_ds, encoder="query")
    t_emb = embed_images(model, t_ds, encoder="target")

    qids = sorted(q_emb)
    tids = sorted(t_emb)
    Q = np.stack([q_emb[q] for q in qids]).astype(np.float32)
    T = np.stack([t_emb[t] for t in tids]).astype(np.float32)
    return (Q @ T.T).astype(np.float32), qids, tids


def zscore(S: np.ndarray) -> np.ndarray:
    """Z-score a similarity matrix (so MIND and cosine are on comparable scales)."""
    mu = S.mean()
    sd = S.std()
    return (S - mu) / sd if sd > 1e-8 else S - mu


def blend(S_mind: np.ndarray, S_cos: np.ndarray, w_mind: float) -> np.ndarray:
    """Weighted sum of z-scored MIND and cosine similarity matrices."""
    return w_mind * zscore(S_mind) + (1.0 - w_mind) * zscore(S_cos)


# --------------------------------------------------------------------------- #
# Offline validation
# --------------------------------------------------------------------------- #


def build_proxy_cosine(qids, tids, model_sim_lookup):
    """Pull the SliceCLIP cosine sub-matrix for a proxy's ids, if available."""
    if model_sim_lookup is None:
        return None
    S = np.zeros((len(qids), len(tids)), dtype=np.float32)
    ok = True
    for i, q in enumerate(qids):
        for j, t in enumerate(tids):
            key = (q, t)
            if key not in model_sim_lookup:
                ok = False
                break
            S[i, j] = model_sim_lookup[key]
        if not ok:
            break
    return S if ok else None


def run_offline(args, device):
    transform = _loader()
    data_root = args.data_root.resolve()
    pairs = load_training_pairs(data_root, args.train_pair_csv, [], [], [])
    rng = np.random.default_rng(args.seed)

    # hold out a subset of the labelled ds1 pairs as the proxy
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    holdout = [pairs[i] for i in idx[: args.holdout]]
    print(f"Offline proxy: {len(holdout)} held-out ds1 pairs (of {len(pairs)}).")

    q_paths = {str(p["query_id"]): Path(p["query_path"]) for p in holdout}
    t_paths = {str(p["target_id"]): Path(p["target_path"]) for p in holdout}
    gold = {str(p["query_id"]): str(p["target_id"]) for p in holdout}

    print("Computing MIND fields for held-out queries/targets...")
    q_fields, q_vols = mind_fields(q_paths, transform, args.grid, device, args.patch_radius)
    t_fields, t_vols = mind_fields(t_paths, transform, args.grid, device, args.patch_radius)

    # SliceCLIP cosine sub-matrix for the proxy. The rerank_baseline .npz only
    # caches the real prediction pools (not the train pairs we hold out here),
    # so for the proxy we embed the held-out images with a SliceCLIP checkpoint
    # if one is supplied via --ckpt. Without it we report MIND-only offline.
    S_cos = None
    if args.ckpt and Path(args.ckpt).exists():
        S_cos_full, cqids, ctids = slice_clip_cosine(args.ckpt, q_paths, t_paths, device)
        print(f"Computed SliceCLIP cosine for proxy from {args.ckpt}.")
    else:
        S_cos_full, cqids, ctids = None, None, None
        print("No --ckpt given; reporting MIND-only offline (no cosine/blend).")

    results = {}

    # --- Proxy A: aligned (ds1/ds3 stand-in) ---
    S_mind, qids, tids = mind_sim_matrix(q_fields, q_vols, t_fields, t_vols, align=False)
    rows = rank_argmax(S_mind, qids, tids)
    results["ds1holdout_MIND"] = mrr(rows, gold)
    results["ds1holdout_MIND_conflict"] = top1_conflict_rate(S_mind)

    if S_cos_full is not None and cqids == qids and ctids == tids:
        S_cos = S_cos_full
    if S_cos is not None:
        results["ds1holdout_cosine"] = mrr(rank_argmax(S_cos, qids, tids), gold)
        for w in args.blend_weights:
            Sb = blend(S_mind, S_cos, w)
            results[f"ds1holdout_blend_w{w}"] = mrr(rank_argmax(Sb, qids, tids), gold)

    # --- Proxy B: synthetic ds2 (independent rigid+elastic deform on the TARGET) ---
    print("Building synthetic-ds2 targets (independent rigid+elastic deform)...")
    t_vols_def = {}
    t_fields_def = {}
    for t, v in t_vols.items():
        vd = deform_volume(v, rng, args.rigid_deg, args.elastic_mag, args.elastic_sigma)
        t_vols_def[t] = vd
        t_fields_def[t] = mind_descriptor(vd, patch_radius=args.patch_radius)

    S_mind2, qids2, tids2 = mind_sim_matrix(q_fields, q_vols, t_fields_def, t_vols_def, align=False)
    results["synds2_MIND_raw"] = mrr(rank_argmax(S_mind2, qids2, tids2), gold)
    results["synds2_MIND_raw_conflict"] = top1_conflict_rate(S_mind2)

    if args.try_align:
        S_mind2a, _, _ = mind_sim_matrix(q_fields, q_vols, t_fields_def, t_vols_def, align=True, patch_radius=args.patch_radius)
        results["synds2_MIND_comalign"] = mrr(rank_argmax(S_mind2a, qids2, tids2), gold)

    if S_cos is not None:
        # cosine baseline is invariant to our synthetic deform only insofar as the
        # cached matrix was computed on undeformed images; we report MIND vs the
        # SAME cosine sub-matrix to gauge blend value on a deformed pool.
        for w in args.blend_weights:
            Sb2 = blend(S_mind2, S_cos, w)
            results[f"synds2_blend_w{w}"] = mrr(rank_argmax(Sb2, qids2, tids2), gold)

    print("\n=== OFFLINE MRR ===")
    print(json.dumps(results, indent=2))
    return results


def load_cosine_lookup(npz_path: Path) -> dict[tuple[str, str], float]:
    """Flatten a rerank_baseline similarities.npz into a {(qid,tid): cos} dict."""
    data = np.load(npz_path, allow_pickle=True)
    lookup: dict[tuple[str, str], float] = {}
    idx = 0
    while f"S_{idx}" in data:
        S = data[f"S_{idx}"]
        qids = [str(x) for x in data[f"qids_{idx}"]]
        tids = [str(x) for x in data[f"tids_{idx}"]]
        for i, q in enumerate(qids):
            for j, t in enumerate(tids):
                lookup[(q, t)] = float(S[i, j])
        idx += 1
    return lookup


# --------------------------------------------------------------------------- #
# Submission
# --------------------------------------------------------------------------- #


def run_submit(args, device):
    transform = _loader()
    data_root = args.data_root.resolve()
    prediction_sets = collect_prediction_sets(data_root, args.query_csv, args.gallery_csv)
    cos_lookup = None
    if args.cosine_npz and Path(args.cosine_npz).exists():
        cos_lookup = load_cosine_lookup(Path(args.cosine_npz))
        print(f"Loaded cosine lookup ({len(cos_lookup)} entries) for blending.")

    all_rows = []
    for idx, ps in enumerate(prediction_sets):
        # dataset is inferred from the pool's csv ordering; the caller passes
        # ds2 pools with --align-pool indices so we COM-align those.
        align = idx in set(args.align_pool)
        print(f"\nPool {idx}: Q={len(ps['queries'])} G={len(ps['targets'])} align={align}")
        q_fields, q_vols = mind_fields(ps["queries"], transform, args.grid, device, args.patch_radius)
        t_fields, t_vols = mind_fields(ps["targets"], transform, args.grid, device, args.patch_radius)
        S_mind, qids, tids = mind_sim_matrix(q_fields, q_vols, t_fields, t_vols, align=align, patch_radius=args.patch_radius)

        S = S_mind
        if cos_lookup is not None and args.blend_w > 0:
            S_cos = build_proxy_cosine(qids, tids, cos_lookup)
            if S_cos is not None:
                S = blend(S_mind, S_cos, args.blend_w)
                print("  blended with cosine.")
            else:
                print("  cosine sub-matrix incomplete for this pool; MIND-only.")
        print(f"  top-1 conflict rate: {top1_conflict_rate(S):.3f}")

        if args.rerank == "sinkhorn":
            all_rows.extend(rank_sinkhorn(S, qids, tids, args.sinkhorn_tau, args.sinkhorn_iter))
        else:
            all_rows.extend(rank_argmax(S, qids, tids))

    write_submission(args.out, all_rows)
    print(f"\nWrote {len(all_rows)} rows -> {args.out}")


# --------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["offline", "submit"], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--grid", type=int, default=64, help="common cube side for resizing")
    p.add_argument("--patch-radius", type=int, default=1, help="MIND patch radius (box)")
    p.add_argument("--seed", type=int, default=20260626)

    # offline
    p.add_argument("--train-pair-csv", type=Path, action="append", default=[])
    p.add_argument("--holdout", type=int, default=80)
    p.add_argument("--rigid-deg", type=float, default=10.0)
    p.add_argument("--elastic-mag", type=float, default=0.03)
    p.add_argument("--elastic-sigma", type=float, default=4.0)
    p.add_argument("--try-align", action="store_true")
    p.add_argument("--blend-weights", type=float, nargs="*", default=[0.3, 0.5, 0.7])
    p.add_argument("--ckpt", type=str, default="",
                   help="SliceCLIP checkpoint for the offline cosine/blend comparison")

    # submit
    p.add_argument("--query-csv", type=Path, action="append", default=[])
    p.add_argument("--gallery-csv", type=Path, action="append", default=[])
    p.add_argument("--align-pool", type=int, nargs="*", default=[])
    p.add_argument("--blend-w", type=float, default=0.0)
    p.add_argument("--rerank", choices=["argmax", "sinkhorn"], default="sinkhorn")
    p.add_argument("--sinkhorn-tau", type=float, default=10.0)
    p.add_argument("--sinkhorn-iter", type=int, default=50)
    p.add_argument("--out", type=Path, default=Path("mind_submission.csv"))

    # shared
    p.add_argument("--cosine-npz", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; grid={args.grid}; patch_radius={args.patch_radius}")
    if args.mode == "offline":
        run_offline(args, device)
    else:
        run_submit(args, device)


if __name__ == "__main__":
    main()
