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
Cross-attention reranker (FINDINGS axis D2) on top of the SliceCLIP bi-encoder.

WHY
---
The bi-encoder (slice_clip_baseline.py) compresses each volume to ONE 128-d
embedding and ranks by cosine similarity (then Sinkhorn bijection rerank ->
best score 0.555). That throws away spatial structure: a query and a candidate
are never looked at *together*. A cross-attention reranker keeps a small grid
of feature tokens per volume and lets query tokens attend to candidate tokens
(and vice-versa) to produce a sharper, pair-specific match logit. Galleries are
small (<=100), so we can cross-score every query x candidate pair in a pool.

HONEST RISK (from sibling experiments)
--------------------------------------
Fine-tuning encoders on dataset1's 350 pairs OVERFITS ds1 and does NOT transfer
to dataset2/3 (which dominate the leaderboard, no labels). A reranker trained on
the same 350 pairs risks the SAME failure. We therefore:
  (a) keep the cross-attention head TINY and freeze the SliceCLIP backbone,
  (b) train under the dataset2-style geometric augmentation (independent
      rigid+elastic deform of query & target) reused from slice_clip_ds2aug.py,
      so the head sees the kind of geometry mismatch it must generalise to,
  (c) validate on a held-out synthetic-ds2 set BEFORE submitting, comparing the
      reranker's MRR against the frozen bi-encoder cosine on the SAME deformed
      holdout. If it does not beat cosine there, it will not help ds2/3.

DESIGN
------
Token features: run the FROZEN SliceCLIP conv stack (.features, 64-ch) on the
96x96x3 slice stack -> 64 x 12 x 12 map -> adaptive-pool to 4x4 -> 16 tokens of
64-d per volume. The cross-attention head (projects to d_model=64, 2 layers of
bidirectional cross-attention, pool, MLP -> scalar logit) is the only trainable
part by default; optionally the backbone can be unfrozen.

Training: listwise softmax over in-batch candidates. For a batch of B true
pairs, build the BxB logit matrix L[i,j] = scorer(query_i, target_j); the
diagonal is positive, the rest are negatives. CE on rows + columns.

Inference: per pool build the full Q x G logit matrix with the scorer, feed it
(in place of cosine S) into argmax / Sinkhorn / Hungarian from rerank_baseline.
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

from slice_clip_baseline import CONFIG, SliceCLIP, collect_prediction_sets, load_training_pairs
from rerank_baseline import rank_argmax, rank_hungarian, rank_sinkhorn, top1_conflict_rate, write_submission


# ===========================================================================
# Volume bank + ds2-style geometric augmentation.
#
# Vendored from the sibling slice_clip_ds2aug.py experiment so this reranker is
# self-contained (it only needs slice_clip_baseline.py + rerank_baseline.py).
# Deterministic preprocessing (load / RAS / 1 mm / scale) is cached as the full
# 3D volume; volumes are center pad/cropped to a common canvas and loaded into a
# GPU bank, then the augmentation (independent 3D rigid + in-plane elastic) and
# the baseline 3-slice extraction run BATCHED on GPU. augment_batch(enabled=False)
# reproduces the baseline slice stack bit-for-bit, so inference stays comparable
# to the original bi-encoder.
# ===========================================================================
AUG = {
    "rotate_deg": 15.0,          # +-15 deg about each axis (3D, before slicing)
    "translate_frac": 0.08,      # +- fraction of axis length
    "elastic_prob": 0.8,         # in-plane non-linear elastic on the 96x96 stack
    "elastic_ctrl": 5,           # control-grid size for the displacement field
    "elastic_strength": 0.10,    # max displacement as fraction of image
}
SLICE_POSITIONS = tuple(CONFIG["slice_positions"])
IMAGE_SIZE = int(CONFIG["image_size"])


def volume_transform() -> Compose:
    return Compose([
        LoadImaged(keys="image", image_only=True),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RAS", labels=None),
        Spacingd(keys="image", pixdim=CONFIG["spacing_mm"], mode="bilinear"),
        ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
        EnsureTyped(keys="image"),
    ])


def make_volume_dataset(images: dict[str, Path], example_root: Path) -> PersistentDataset:
    cache_dir = example_root / (str(CONFIG["cache_dir"]) + "_vol")
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"image": str(path), "id": image_id} for image_id, path in sorted(images.items())]
    return PersistentDataset(data=rows, transform=volume_transform(), cache_dir=cache_dir)


def _pad_or_crop(v: torch.Tensor, target) -> torch.Tensor:
    """Center pad-with-zeros or crop a (1,H,W,D) volume to `target` (Ht,Wt,Dt)."""
    _, H, W, D = v.shape
    Ht, Wt, Dt = target
    out = torch.zeros((1, Ht, Wt, Dt), dtype=v.dtype)

    def span(s, t):
        if s >= t:
            o = (s - t) // 2
            return slice(o, o + t), slice(0, t)
        o = (t - s) // 2
        return slice(0, s), slice(o, o + s)

    sh, dh = span(H, Ht)
    sw, dw = span(W, Wt)
    sd, dd = span(D, Dt)
    out[:, dh, dw, dd] = v[:, sh, sw, sd]
    return out


def load_volume_bank(volume_dataset: PersistentDataset, ids: list[str], device):
    """Load preprocessed volumes for `ids` into one (N,1,H,W,D) GPU tensor.

    Volumes differ in shape across datasets (dataset2/3 were reshaped), so we
    center pad/crop each to a fixed canvas that holds the largest volume.
    """
    id_to_index = {row["id"]: i for i, row in enumerate(volume_dataset.data)}
    raw, maxshape = [], [0, 0, 0]
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
    bank = torch.stack([_pad_or_crop(v, target) for v in raw]).to(device)
    pos = {iid: i for i, iid in enumerate(ids)}
    return bank, pos


def _occupancy_zslices(vol_batch: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W,D) -> integer z-indices (B,3) at 35/50/65% occupancy."""
    B, _, H, W, D = vol_batch.shape
    nz = (vol_batch[:, 0].abs() > 1e-6).any(dim=1).any(dim=1)  # (B, D)
    z_idx = torch.zeros(B, len(SLICE_POSITIONS), dtype=torch.long, device=vol_batch.device)
    for b in range(B):
        occ = torch.nonzero(nz[b], as_tuple=False).flatten()
        zmin, zmax = (D // 2, D // 2) if len(occ) == 0 else (int(occ[0]), int(occ[-1]))
        for k, p in enumerate(SLICE_POSITIONS):
            z_idx[b, k] = int(round(zmin + p * (zmax - zmin)))
    return z_idx.clamp_(0, D - 1)


def _rand_rigid_grid(B: int, shape, device, gen):
    """Random 3D rigid (rotation+translation) sampling grid (B,H,W,D,3)."""
    H, W, D = shape
    rot = (torch.rand(B, 3, device=device, generator=gen) * 2 - 1) * (AUG["rotate_deg"] * np.pi / 180.0)
    trans = (torch.rand(B, 3, device=device, generator=gen) * 2 - 1) * (AUG["translate_frac"] * 2.0)
    cz, sz = torch.cos(rot[:, 0]), torch.sin(rot[:, 0])
    cy, sy = torch.cos(rot[:, 1]), torch.sin(rot[:, 1])
    cx, sx = torch.cos(rot[:, 2]), torch.sin(rot[:, 2])
    zero, one = torch.zeros(B, device=device), torch.ones(B, device=device)

    def stack3(r0, r1, r2):
        return torch.stack([torch.stack(r0, -1), torch.stack(r1, -1), torch.stack(r2, -1)], 1)

    Rz = stack3([cz, -sz, zero], [sz, cz, zero], [zero, zero, one])
    Ry = stack3([cy, zero, sy], [zero, one, zero], [-sy, zero, cy])
    Rx = stack3([one, zero, zero], [zero, cx, -sx], [zero, sx, cx])
    R = Rz @ Ry @ Rx
    theta = torch.cat([R, trans.unsqueeze(-1)], dim=-1)
    return F.affine_grid(theta, (B, 1, H, W, D), align_corners=False)


def _rand_elastic_grid(B: int, size: int, device, gen):
    """Random in-plane elastic deformation grid (B,size,size,2)."""
    c = AUG["elastic_ctrl"]
    disp = (torch.rand(B, 2, c, c, device=device, generator=gen) * 2 - 1) * AUG["elastic_strength"]
    disp = F.interpolate(disp, size=(size, size), mode="bilinear", align_corners=True)
    base = F.affine_grid(
        torch.eye(2, 3, device=device).unsqueeze(0).repeat(B, 1, 1), (B, 1, size, size), align_corners=False
    )
    return base + disp.permute(0, 2, 3, 1)


def augment_batch(vols: torch.Tensor, gen, enabled: bool, aug_prob: float = 1.0) -> torch.Tensor:
    """(B,1,H,W,D) -> (B,3,IMAGE_SIZE,IMAGE_SIZE) slice stacks.

    enabled=False reproduces the baseline deterministic 3-slice stack bit-for-bit
    (no rigid, no elastic). When enabled, each item draws its own rigid transform
    and its own clean/deform coin, so the query and target of a pair are deformed
    INDEPENDENTLY (the dataset2 recipe).
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

    z_idx = _occupancy_zslices(vols)
    v0 = vols[:, 0]
    out = []
    for k in range(len(SLICE_POSITIONS)):
        zk = z_idx[:, k].view(B, 1, 1, 1).expand(B, v0.shape[1], v0.shape[2], 1)
        out.append(torch.gather(v0, 3, zk).squeeze(-1))
    stack = torch.stack(out, dim=1)
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
# Cross-attention scorer.
# ---------------------------------------------------------------------------
class TokenExtractor(nn.Module):
    """Wrap a SliceCLIP encoder's conv stack to emit a small token sequence.

    Input  : (B, 3, 96, 96) slice stack.
    Output : (B, n_tokens, 64) feature tokens (adaptive-pooled conv grid).
    """

    def __init__(self, features: nn.Sequential, grid: int = 4) -> None:
        super().__init__()
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))
        self.grid = grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)                  # (B, 64, 12, 12)
        f = self.pool(f)                      # (B, 64, grid, grid)
        b, c, h, w = f.shape
        return f.flatten(2).transpose(1, 2)   # (B, h*w, 64)


class CrossAttnBlock(nn.Module):
    """One bidirectional cross-attention update (q<-c and c<-q) + FFN."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.q2c = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.c2q = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.nq1 = nn.LayerNorm(d_model)
        self.nc1 = nn.LayerNorm(d_model)
        self.nq2 = nn.LayerNorm(d_model)
        self.nc2 = nn.LayerNorm(d_model)
        self.ffq = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
        self.ffc = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))

    def forward(self, q: torch.Tensor, c: torch.Tensor):
        # Normalize the ORIGINAL q and c once, then update both symmetrically:
        # query attends to candidate and candidate attends to query off the same
        # pre-update tensors (true bidirectional cross-attention).
        qn, cn = self.nq1(q), self.nc1(c)
        qa, _ = self.q2c(qn, cn, cn, need_weights=False)
        ca, _ = self.c2q(cn, qn, qn, need_weights=False)
        q = q + qa
        c = c + ca
        q = q + self.ffq(self.nq2(q))
        c = c + self.ffc(self.nc2(c))
        return q, c


class CrossAttnScorer(nn.Module):
    """Score a (query, candidate) pair by cross-attending their feature grids,
    as a learned RESIDUAL on top of the frozen bi-encoder cosine.

    DIAGNOSIS THAT DROVE THIS DESIGN
    --------------------------------
    For this tiny SliceCLIP encoder the discriminative signal lives almost
    entirely in the projection MLP, NOT in the conv feature map: across volumes
    the 64-ch conv grid has std ~0.006 (nearly constant) while the 128-d
    embedding has std ~0.088 and separates clean ds1 perfectly (MRR 1.0). A
    cross-attention head fed only the conv tokens therefore has almost no signal
    and collapses to a constant score (top-1 conflict 1.0).

    So we keep the strong, discriminative bi-encoder cosine as the BASE score and
    let cross-attention over the conv tokens produce a small additive
    REFINEMENT, gated by a parameter initialised to ~0. At init the scorer
    exactly reproduces the bi-encoder cosine (the 0.555 baseline); training can
    only move it if the cross-attention finds a real correction. This both
    prevents collapse and makes the experiment a clean test of "does cross-
    attention refine beyond cosine".
    """

    def __init__(self, slice_clip: SliceCLIP, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, grid: int = 4, dropout: float = 0.2,
                 freeze_backbone: bool = True) -> None:
        super().__init__()
        # The frozen bi-encoder provides both the base cosine and the conv tokens.
        self.slice_clip = slice_clip
        self.q_tokens = TokenExtractor(slice_clip.query_encoder.features, grid)
        self.c_tokens = TokenExtractor(slice_clip.target_encoder.features, grid)
        if freeze_backbone:
            for p in self.slice_clip.parameters():
                p.requires_grad_(False)
        self.freeze_backbone = freeze_backbone

        self.q_proj = nn.Linear(64, d_model)
        self.c_proj = nn.Linear(64, d_model)
        n_tok = grid * grid
        self.q_pos = nn.Parameter(torch.zeros(1, n_tok, d_model))
        self.c_pos = nn.Parameter(torch.zeros(1, n_tok, d_model))
        nn.init.trunc_normal_(self.q_pos, std=0.02)
        nn.init.trunc_normal_(self.c_pos, std=0.02)
        self.blocks = nn.ModuleList([CrossAttnBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        # Refinement head: cross-attended pooled query/candidate -> scaled cosine.
        self.q_pool_norm = nn.LayerNorm(d_model)
        self.c_pool_norm = nn.LayerNorm(d_model)
        self.q_match = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.c_match = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        # Base cosine temperature (matches the bi-encoder similarity_scale) and a
        # refinement gate initialised SMALL but NONZERO. Zero would start exactly
        # at the cosine baseline but also zero out the gradient into the gated
        # cross-attention branch (d logits / d refine = gate = 0), so the head
        # could never start learning. A small value keeps the model close to the
        # baseline while letting gradients reach the refinement head.
        self.base_scale = nn.Parameter(torch.tensor(float(slice_clip.similarity_scale)))
        self.refine_gate = nn.Parameter(torch.full((1,), 0.1))

    def _backbone(self, q_stack: torch.Tensor, c_stack: torch.Tensor):
        """Run the (optionally frozen) SliceCLIP backbone: conv tokens + embeds.

        When the backbone is frozen we run it under no_grad and DETACH the
        outputs, so the trainable projections / cross-attention head applied
        afterwards still build a live autograd graph from these tensors.
        """
        if self.freeze_backbone:
            with torch.no_grad():
                qt = self.q_tokens(q_stack).detach()
                ct = self.c_tokens(c_stack).detach()
                qe = self.slice_clip.encode_query(q_stack).detach()
                ce = self.slice_clip.encode_target(c_stack).detach()
        else:
            qt = self.q_tokens(q_stack)
            ct = self.c_tokens(c_stack)
            qe = self.slice_clip.encode_query(q_stack)
            ce = self.slice_clip.encode_target(c_stack)
        return qt, ct, qe, ce

    def _embed(self, q_stack: torch.Tensor, c_stack: torch.Tensor):
        """Return conv tokens (q,c) AND the base bi-encoder cosine per pair."""
        qt, ct, qe, ce = self._backbone(q_stack, c_stack)
        q = self.q_proj(qt) + self.q_pos                 # trainable, builds graph
        c = self.c_proj(ct) + self.c_pos
        base_cos = (qe * ce).sum(-1)                     # (B,) cosine per pair
        return q, c, base_cos

    def _refine(self, q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            q, c = blk(q, c)
        qv = F.normalize(self.q_match(self.q_pool_norm(q.mean(1))), dim=-1)
        cv = F.normalize(self.c_match(self.c_pool_norm(c.mean(1))), dim=-1)
        return (qv * cv).sum(-1)                          # (B,) refinement cosine

    def score_from_parts(self, q, c, base_cos) -> torch.Tensor:
        return self.base_scale * base_cos + self.refine_gate * self._refine(q, c)

    def cross_matrix(self, q_stack: torch.Tensor, c_stack: torch.Tensor) -> torch.Tensor:
        """Full Q x G logit matrix. q_stack:(Q,3,96,96) c_stack:(G,3,96,96).

        Embed query/candidate tokens and embeddings once each, then cross every
        pair: base cosine (from the 128-d embeddings) + gated cross-attention
        refinement (from the conv tokens).
        """
        qt, ct, qe, ce = self._backbone(q_stack, c_stack)
        q_all = self.q_proj(qt) + self.q_pos                              # (Q, T, d)
        c_all = self.c_proj(ct) + self.c_pos                              # (G, T, d)
        base = self.base_scale * (qe @ ce.T)                              # (Q,G)
        Q, G = q_all.shape[0], c_all.shape[0]
        T, d = q_all.shape[1], q_all.shape[2]
        out = torch.empty(Q, G, device=q_all.device)
        for i in range(Q):
            qi = q_all[i:i + 1].expand(G, T, d).contiguous()
            out[i] = base[i] + self.refine_gate * self._refine(qi, c_all.clone())
        return out


# ---------------------------------------------------------------------------
# Training (listwise softmax over in-batch candidates).
# ---------------------------------------------------------------------------
def train_scorer(scorer, bank, pos, train_pairs, epochs, augment, device, gen,
                 lr, weight_decay, batch_size, log_every=25):
    params = [p for p in scorer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    q_idx = torch.tensor([pos[str(p["query_id"])] for p in train_pairs], device=device)
    t_idx = torch.tensor([pos[str(p["target_id"])] for p in train_pairs], device=device)
    n = len(train_pairs)
    for epoch in range(1, epochs + 1):
        scorer.train()
        perm = torch.randperm(n, device=device, generator=gen)
        tot, seen = 0.0, 0
        for s in range(0, n, batch_size):
            bidx = perm[s:s + batch_size]
            B = len(bidx)
            if B < 2:
                continue
            qv = bank[q_idx[bidx]]
            tv = bank[t_idx[bidx]]
            q_stack = augment_batch(qv, gen, augment)
            t_stack = augment_batch(tv, gen, augment)
            # build B x B logit matrix = base cosine + gated cross-attn refine.
            # Backbone (frozen) runs under no_grad and is detached inside
            # _backbone; the trainable projections + _refine below build a live
            # graph so the cross-attention head actually receives gradients.
            qtok_raw, ctok_raw, qe, cge = scorer._backbone(q_stack, t_stack)
            qtok = scorer.q_proj(qtok_raw) + scorer.q_pos                     # (B,T,d)
            ctok = scorer.c_proj(ctok_raw) + scorer.c_pos                     # (B,T,d)
            T, d = qtok.shape[1], qtok.shape[2]
            qi = qtok.unsqueeze(1).expand(B, B, T, d).reshape(B * B, T, d)
            cj = ctok.unsqueeze(0).expand(B, B, T, d).reshape(B * B, T, d)
            base = scorer.base_scale * (qe @ cge.T)                           # (B,B)
            refine = scorer._refine(qi, cj).reshape(B, B)
            logits = base + scorer.refine_gate * refine
            labels = torch.arange(B, device=device)
            loss = (ce(logits, labels) + ce(logits.T, labels)) / 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
            tot += float(loss.item()) * B
            seen += B
        if epoch % log_every == 0 or epoch in (1, epochs):
            print(f"epoch {epoch:04d} loss={tot / max(seen, 1):.5f}", flush=True)
    return scorer


# ---------------------------------------------------------------------------
# Synthetic-ds2 evaluation: deform holdout query/target independently, then
# compare cross-attention MRR vs frozen bi-encoder cosine MRR on the SAME data.
# ---------------------------------------------------------------------------
@torch.no_grad()
def synthetic_ds2_eval(scorer, slice_clip, bank, pos, holdout_pairs, device,
                       n_repeat=5, seed=12345):
    scorer.eval()
    slice_clip.eval()
    qids = [str(p["query_id"]) for p in holdout_pairs]
    tids = [str(p["target_id"]) for p in holdout_pairs]
    qv = bank[torch.tensor([pos[q] for q in qids], device=device)]
    tv = bank[torch.tensor([pos[t] for t in tids], device=device)]
    N = len(qids)
    rr_x, rr_cos = [], []
    for rep in range(n_repeat):
        g = torch.Generator(device=device).manual_seed(seed + rep)
        q_stack = augment_batch(qv, g, True)
        t_stack = augment_batch(tv, g, True)
        # cross-attention score matrix (gold = diagonal)
        Sx = scorer.cross_matrix(q_stack, t_stack).cpu().numpy()
        # frozen bi-encoder cosine on the SAME deformed stacks
        qe = slice_clip.encode_query(q_stack)
        te = slice_clip.encode_target(t_stack)
        Sc = (qe @ te.T).cpu().numpy()
        for S, rr in ((Sx, rr_x), (Sc, rr_cos)):
            order = np.argsort(-S, axis=1)
            for i in range(N):
                rank = int(np.where(order[i] == i)[0][0]) + 1
                rr.append(1.0 / rank)
    return float(np.mean(rr_x)), float(np.mean(rr_cos))


@torch.no_grad()
def clean_stacks_for_pool(volume_dataset, ids, device):
    """No-augmentation 96x96x3 slice stacks for a small id set (one pool).

    Loads only this pool's 3D volumes (galleries <=100, so <=200 ids), extracts
    the baseline-identical slice stacks on GPU, and returns just the tiny stacks
    so the large 3D volumes can be freed. Avoids holding the full inference set
    (mixed dataset2/3 shapes pad up to a ~280x280x315 canvas) on the GPU at once.
    """
    bank, pos = load_volume_bank(volume_dataset, ids, device)
    out = {}
    bs = 64
    for s in range(0, len(ids), bs):
        chunk = ids[s:s + bs]
        v = bank[torch.tensor([pos[i] for i in chunk], device=device)]
        stack = augment_batch(v, None, enabled=False)  # (b,3,96,96)
        for i, iid in enumerate(chunk):
            out[iid] = stack[i].detach().clone()
    del bank
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Cross-attention reranker on top of SliceCLIP.")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--train-pair-csv", type=Path, action="append", default=[])
    p.add_argument("--query-csv", type=Path, action="append", default=[])
    p.add_argument("--gallery-csv", type=Path, action="append", default=[])
    p.add_argument("--backbone-ckpt", type=Path, required=True,
                   help="SliceCLIP state_dict (rerank_out/model.pt) to load as frozen encoder")
    p.add_argument("--out-dir", type=Path, default=Path("xattn_out"))
    p.add_argument("--ckpt", type=Path, default=Path("xattn_out/scorer.pt"))
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--grid", type=int, default=4)
    p.add_argument("--holdout", type=int, default=50)
    p.add_argument("--augment", action="store_true", help="train under ds2-style geometric augmentation")
    p.add_argument("--unfreeze-backbone", action="store_true")
    p.add_argument("--sinkhorn-tau", type=float, default=10.0)
    p.add_argument("--sinkhorn-iter", type=int, default=50)
    p.add_argument("--no-train", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(int(CONFIG["seed"]))
    np.random.seed(int(CONFIG["seed"]))
    torch.manual_seed(int(CONFIG["seed"]))

    example_root = Path(__file__).resolve().parent
    data_root = args.data_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(CONFIG["device"]))

    # --- frozen bi-encoder backbone ---
    slice_clip = SliceCLIP(int(CONFIG["embedding_dim"])).to(device)
    slice_clip.load_state_dict(torch.load(args.backbone_ckpt, map_location=device))
    slice_clip.eval()
    scorer = CrossAttnScorer(
        slice_clip, d_model=args.d_model, n_layers=args.n_layers, grid=args.grid,
        dropout=args.dropout, freeze_backbone=not args.unfreeze_backbone,
    ).to(device)
    n_train_params = sum(p.numel() for p in scorer.parameters() if p.requires_grad)
    n_total_params = sum(p.numel() for p in scorer.parameters())

    # --- holdout split for synthetic-ds2 (same RNG recipe as ds2aug) ---
    all_pairs = load_training_pairs(data_root, args.train_pair_csv, [], [], [])
    rng = random.Random(int(CONFIG["seed"]))
    indices = list(range(len(all_pairs)))
    rng.shuffle(indices)
    holdout_idx = set(indices[: args.holdout])
    holdout_pairs = [all_pairs[i] for i in sorted(holdout_idx)]
    train_pairs = [all_pairs[i] for i in indices if i not in holdout_idx]

    prediction_sets = collect_prediction_sets(data_root, args.query_csv, args.gallery_csv)

    # Images needed for TRAINING + synthetic-ds2 holdout = only the ds1 pairs.
    # These are uniform (240,240,155), so a GPU bank fits comfortably and we
    # augment on it. The inference pools (mixed dataset2/3 shapes) are handled
    # one pool at a time later, so we never stack the full set on the GPU.
    train_images: dict[str, Path] = {}
    for pp in all_pairs:
        train_images[str(pp["query_id"])] = Path(pp["query_path"])
        train_images[str(pp["target_id"])] = Path(pp["target_path"])

    all_images: dict[str, Path] = dict(train_images)
    for ps in prediction_sets:
        all_images.update(ps["queries"])
        all_images.update(ps["targets"])

    print(json.dumps({
        "trainable_params": n_train_params,
        "total_params": n_total_params,
        "freeze_backbone": not args.unfreeze_backbone,
        "augment": args.augment,
        "epochs": args.epochs,
        "pairs_total": len(all_pairs), "pairs_train": len(train_pairs), "pairs_holdout": len(holdout_pairs),
        "num_train_images": len(train_images), "num_all_images": len(all_images),
        "num_prediction_sets": len(prediction_sets),
        "d_model": args.d_model, "n_layers": args.n_layers, "grid": args.grid,
    }, indent=2), flush=True)

    # One PersistentDataset over every image (cache is shared with ds2aug); the
    # bank below loads only the train ids into GPU memory.
    volume_dataset = make_volume_dataset(all_images, example_root)
    train_ids = sorted(train_images)
    bank, pos = load_volume_bank(volume_dataset, train_ids, device)
    print(f"train volume bank: {tuple(bank.shape)} on {bank.device}", flush=True)

    if args.no_train and args.ckpt.exists():
        print(f"Loading scorer from {args.ckpt}", flush=True)
        scorer.load_state_dict(torch.load(args.ckpt, map_location=device))
    else:
        gen = torch.Generator(device=device).manual_seed(int(CONFIG["seed"]))
        print(f"\nTraining cross-attention scorer ({n_train_params} trainable params) "
              f"augment={args.augment} for {args.epochs} epochs...", flush=True)
        train_scorer(scorer, bank, pos, train_pairs, args.epochs, args.augment, device, gen,
                     args.lr, args.weight_decay, args.batch_size)
        torch.save(scorer.state_dict(), args.ckpt)
        print(f"Saved scorer -> {args.ckpt}", flush=True)

    # --- synthetic-ds2 transfer check: cross-attn vs frozen cosine on deformed holdout ---
    mrr_x, mrr_cos = synthetic_ds2_eval(scorer, slice_clip, bank, pos, holdout_pairs, device)
    print(f"\n=== synthetic-ds2 (holdout={len(holdout_pairs)}, deformed independently) ===", flush=True)
    print(f"  cross-attention MRR = {mrr_x:.4f}", flush=True)
    print(f"  frozen-cosine   MRR = {mrr_cos:.4f}", flush=True)
    print(f"  delta (xattn-cos)   = {mrr_x - mrr_cos:+.4f}", flush=True)

    # --- inference: per-pool cross-attention score matrix -> argmax/sinkhorn/hungarian ---
    # Free the training bank before loading inference volumes (memory-frugal: the
    # GPU is shared and dataset2/3 volumes pad up to a large canvas).
    del bank
    if device.type == "cuda":
        torch.cuda.empty_cache()

    methods = {"argmax": [], "sinkhorn": [], "hungarian": []}
    scorer.eval()
    print("\n=== per-pool cross-attention scoring (one pool at a time) ===", flush=True)
    for idx, ps in enumerate(prediction_sets):
        qids = sorted(ps["queries"])
        tids = sorted(ps["targets"])
        pool_ids = sorted(set(qids) | set(tids))
        stacks = clean_stacks_for_pool(volume_dataset, pool_ids, device)
        q_stack = torch.stack([stacks[q] for q in qids]).to(device)
        c_stack = torch.stack([stacks[t] for t in tids]).to(device)
        with torch.no_grad():
            S = scorer.cross_matrix(q_stack, c_stack).cpu().numpy().astype(np.float64)
        print(f"  pool {idx}: Q={len(qids)} G={len(tids)} top-1 conflict={top1_conflict_rate(S):.3f}", flush=True)
        methods["argmax"].extend(rank_argmax(S, qids, tids))
        methods["sinkhorn"].extend(rank_sinkhorn(S, qids, tids, args.sinkhorn_tau, args.sinkhorn_iter))
        methods["hungarian"].extend(rank_hungarian(S, qids, tids))
        np.savez(args.out_dir / f"S_pool{idx}.npz", S=S, qids=np.array(qids), tids=np.array(tids))

    for name, rows in methods.items():
        out = args.out_dir / f"xattn_{name}_submission.csv"
        write_submission(out, rows)
        print(f"  {name}: {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
