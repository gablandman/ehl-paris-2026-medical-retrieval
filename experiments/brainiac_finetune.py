"""Fine-tune the BrainIAC encoder for cross-modal same-subject retrieval.

The frozen BrainIAC backbone (zero-shot) only scored ~0.307 on this T1c<->T2
same-subject matching task: its SimCLR-pretrained features are not tuned for
"is this the same patient across two modalities". Here we adapt it on the 350
labelled dataset1 pairs with a CLIP-style symmetric InfoNCE loss, in two modes:

  Mode A (head)  : freeze the whole backbone, train two small MLP heads
                   (one per modality) on top of the frozen 768-d feature.
  Mode B (lora)  : inject LoRA adapters into the backbone linear layers
                   (attn.qkv / attn.out_proj / mlp.linear1 / mlp.linear2),
                   keep base weights frozen, train LoRA params + the two heads.

Design choices (justified):
  * Per-modality heads. Query (T1 post-contrast) and target (T2) have genuinely
    different intensity/contrast distributions, so a single shared head would
    have to absorb that modality gap. Two small heads (768->256->128) let each
    modality learn its own projection into a shared cosine space; cheap enough
    not to overfit 350 pairs. The backbone is shared (same weights / same LoRA)
    so the heavy feature extractor stays modality-agnostic.
  * Symmetric InfoNCE with a learnable temperature, like CLIP. L2-normalize
    embeddings, cosine similarity scaled by 1/temperature.
  * Overfitting guards: few epochs, small LR, hold out a handful of pairs to
    watch a quick in-pool MRR; the real signal is the leaderboard.

After training we reuse brainiac_encode preprocessing and rerank_baseline
rerankers to embed all 6 pools, build per-pool sim matrices, and write
argmax / sinkhorn / hungarian submissions.

Empirical result (2026-06-27, dataset1's 350 pairs):
  * The feature-standardization fix is essential -- without it the head's
    InfoNCE is pinned at ln(batch) and learns nothing (the raw BrainIAC feature
    is near-constant across inputs: off-diagonal cosine ~0.9999). With it, both
    modes train cleanly and reach ~0.45-0.51 in-pool MRR on a 40-pair holdout.
  * BUT neither mode generalizes to the leaderboard. Public MRR:
      frozen BrainIAC  ~0.307
      adapter head     0.261 (argmax) / 0.268 (sinkhorn) / 0.288 (hungarian)
      LoRA r=8         0.299 (argmax) / 0.296 (sinkhorn) / 0.292 (hungarian)
      SliceCLIP (best) 0.555 (sinkhorn)
    Biased train MRR was high (head 0.54-0.68, LoRA 0.86-0.92), confirming the
    models *memorize* dataset1's 350 pairs but do not transfer -- the public LB
    is dominated by dataset2/3 pools that have no training labels and a likely
    domain shift. Fine-tuning BrainIAC on dataset1 alone is therefore a negative
    result here: it does not beat the frozen encoder, let alone SliceCLIP.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.data import DataLoader, PersistentDataset
from monai.networks.nets import ViT
from safetensors.torch import load_file
from torch.utils.data import Dataset
from tqdm import tqdm

from brainiac_encode import make_dataset, preprocess
from rerank_baseline import (
    cosine_sim,
    mrr,
    rank_argmax,
    rank_hungarian,
    rank_sinkhorn,
    top1_conflict_rate,
    write_submission,
)
from slice_clip_baseline import collect_prediction_sets, load_training_pairs


# ----------------------------- model --------------------------------------

def build_backbone(weights_path: Path) -> ViT:
    model = ViT(
        in_channels=1,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
    )
    state = load_file(str(weights_path))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded BrainIAC backbone ({len(state)} tensors); "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    return model


class ProjHead(nn.Module):
    """Small MLP: 768 -> hidden -> out, with GELU + dropout."""

    def __init__(self, in_dim: int = 768, hidden: int = 256, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RetrievalModel(nn.Module):
    """Shared BrainIAC backbone + two per-modality projection heads + temperature.

    IMPORTANT: the frozen BrainIAC features are near-collapsed -- every input
    maps to almost the same 768-d vector (off-diagonal cosine ~0.9999, a giant
    shared offset with a tiny per-sample residual). Feeding that raw into a head
    leaves InfoNCE stuck at ln(batch) because all rows look identical. We fix
    this with a per-dimension standardization (subtract dataset mean, divide by
    std) computed once from the frozen backbone, registered as buffers, so the
    head sees the residual signal at unit scale. For LoRA the backbone itself
    learns to produce discriminative features, but standardization is still a
    harmless, helpful normalization, so we apply it in both modes.
    """

    def __init__(self, backbone: ViT, hidden: int = 256, out_dim: int = 128,
                 dropout: float = 0.1, init_logit_scale: float = float(np.log(1 / 0.07))):
        super().__init__()
        self.backbone = backbone
        self.query_head = ProjHead(768, hidden, out_dim, dropout)
        self.target_head = ProjHead(768, hidden, out_dim, dropout)
        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))
        # standardization buffers (set via set_feature_stats); identity by default
        self.register_buffer("feat_mean", torch.zeros(768))
        self.register_buffer("feat_std", torch.ones(768))

    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std.clamp_min(1e-6))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)[0][:, 0]  # first patch token, (B, 768)
        return (f - self.feat_mean) / self.feat_std

    def encode_query(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_head(self.features(x)), dim=1)

    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.target_head(self.features(x)), dim=1)


def apply_lora(backbone: ViT, r: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["qkv", "out_proj", "linear1", "linear2"],
        bias="none",
    )
    return get_peft_model(backbone, cfg)


# ----------------------------- data ---------------------------------------

class PairVolumeDataset(Dataset):
    """Positive (query, target) volume pairs, backed by a cached PersistentDataset."""

    def __init__(self, pairs, image_dataset: PersistentDataset):
        self.image_dataset = image_dataset
        self.id_to_index = {row["id"]: i for i, row in enumerate(image_dataset.data)}
        self.examples = [(str(p["query_id"]), str(p["target_id"])) for p in pairs]

    def __len__(self):
        return len(self.examples)

    def _img(self, image_id: str) -> torch.Tensor:
        item = self.image_dataset[self.id_to_index[image_id]]
        return torch.as_tensor(item["image"]).float()

    def __getitem__(self, idx):
        q, t = self.examples[idx]
        return self._img(q), self._img(t)


# ----------------------------- train --------------------------------------

def train(model: RetrievalModel, train_ds: PairVolumeDataset, device, *,
          epochs: int, lr: float, batch_size: int, weight_decay: float,
          num_workers: int, hold_ds: PairVolumeDataset | None = None):
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_train:,}")
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    ce = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        tot, seen = 0.0, 0
        for q, t in loader:
            q, t = q.to(device), t.to(device)
            qe = model.encode_query(q)
            te = model.encode_target(t)
            scale = model.logit_scale.exp().clamp(max=100.0)
            logits = scale * qe @ te.T
            labels = torch.arange(len(q), device=device)
            loss = (ce(logits, labels) + ce(logits.T, labels)) / 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            tot += float(loss.item()) * len(q)
            seen += len(q)
        msg = f"epoch {epoch:02d} loss={tot/max(seen,1):.4f} scale={model.logit_scale.exp().item():.2f}"
        if hold_ds is not None and len(hold_ds) > 0:
            msg += f" holdout_mrr={quick_mrr(model, hold_ds, device):.4f}"
        print(msg, flush=True)
    return model


@torch.no_grad()
def quick_mrr(model: RetrievalModel, ds: PairVolumeDataset, device) -> float:
    """In-pool MRR over a small held-out set: rank each held-out query vs all held-out targets."""
    model.eval()
    qs, ts = [], []
    for q, t in ds:
        qs.append(model.encode_query(q.unsqueeze(0).to(device)).cpu().numpy()[0])
        ts.append(model.encode_target(t.unsqueeze(0).to(device)).cpu().numpy()[0])
    Q = np.stack(qs)
    T = np.stack(ts)
    S = Q @ T.T
    rr = []
    for i in range(len(Q)):
        order = np.argsort(-S[i])
        rr.append(1.0 / (int(np.where(order == i)[0][0]) + 1))
    return float(np.mean(rr))


@torch.no_grad()
def embed_all(model: RetrievalModel, dataset: PersistentDataset, device, *,
              batch_size: int, num_workers: int, encoder: str) -> dict[str, np.ndarray]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    out: dict[str, np.ndarray] = {}
    enc = model.encode_query if encoder == "query" else model.encode_target
    for batch in tqdm(loader, desc=f"embed[{encoder}]"):
        x = torch.as_tensor(batch["image"]).float().to(device)
        e = enc(x).cpu().numpy().astype(np.float32)
        for image_id, vec in zip(batch["id"], e):
            out[str(image_id)] = vec
    return out


@torch.no_grad()
def compute_feature_stats(model: RetrievalModel, dataset: PersistentDataset, device, *,
                          batch_size: int, num_workers: int):
    """Per-dimension mean/std of the RAW backbone feature over the training images.

    Computed with the head's standardization temporarily disabled (identity
    buffers) so we measure the backbone output directly. For LoRA this is the
    base (pre-adapter) feature stats, which is fine -- it just centers the input
    to the head; the adapters then learn on top.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feats = []
    for batch in tqdm(loader, desc="feat-stats"):
        x = torch.as_tensor(batch["image"]).float().to(device)
        f = model.backbone(x)[0][:, 0]  # raw, no standardization
        feats.append(f.cpu())
    F_all = torch.cat(feats, dim=0)
    return F_all.mean(0), F_all.std(0)


# ----------------------------- main ---------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["head", "lora"], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--train-pair-csv", type=Path, action="append", required=True)
    p.add_argument("--query-csv", type=Path, action="append", required=True)
    p.add_argument("--gallery-csv", type=Path, action="append", required=True)
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("ft_out"))
    p.add_argument("--cache-dir", type=Path, default=Path(".brainiac_cache"))
    # Defaults are the head-mode sweet spot (loss/holdout-MRR plateau ~ep12-14).
    # For LoRA we used --lr 1e-3 --batch-size 32 --epochs 18 on the CLI.
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--holdout", type=int, default=40, help="# pairs held out for quick MRR watch")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20260627)
    p.add_argument("--sinkhorn-tau", type=float, default=10.0)
    p.add_argument("--sinkhorn-iter", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_root = args.data_root.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = args.weights
    if weights is None:
        from huggingface_hub import hf_hub_download
        weights = Path(hf_hub_download(repo_id="eugenehp/brainiac", filename="backbone.safetensors"))
    print(f"mode={args.mode} weights={weights} device={device}")

    # ---- training pairs ----
    train_pairs = load_training_pairs(data_root, args.train_pair_csv, [], [], [])
    random.shuffle(train_pairs)
    hold_pairs = train_pairs[: args.holdout] if args.holdout > 0 else []
    fit_pairs = train_pairs[args.holdout:] if args.holdout > 0 else train_pairs
    print(f"pairs: total={len(train_pairs)} fit={len(fit_pairs)} holdout={len(hold_pairs)}")

    # ---- prediction pools ----
    prediction_sets = collect_prediction_sets(data_root, args.query_csv, args.gallery_csv)

    # ---- build image manifests ----
    train_images: dict[str, Path] = {}
    for p in train_pairs:
        train_images[str(p["query_id"])] = Path(p["query_path"])
        train_images[str(p["target_id"])] = Path(p["target_path"])

    query_inf: dict[str, Path] = {}
    target_inf: dict[str, Path] = {}
    for ps in prediction_sets:
        query_inf.update(ps["queries"])
        target_inf.update(ps["targets"])

    # caches: train uses both heads on its own volumes; build one shared cache dir
    train_ds_cache = make_dataset(train_images, args.cache_dir)
    train_dataset = PairVolumeDataset(fit_pairs, train_ds_cache)
    hold_dataset = PairVolumeDataset(hold_pairs, train_ds_cache) if hold_pairs else None

    print(json.dumps({
        "num_train_images": len(train_images),
        "num_query_inf": len(query_inf),
        "num_target_inf": len(target_inf),
        "num_prediction_sets": len(prediction_sets),
    }, indent=2))

    # ---- model ----
    backbone = build_backbone(weights)
    if args.mode == "head":
        for p in backbone.parameters():
            p.requires_grad = False
    elif args.mode == "lora":
        backbone = apply_lora(backbone, args.lora_r, args.lora_alpha, args.lora_dropout)
    model = RetrievalModel(backbone, dropout=args.dropout).to(device)

    # ---- standardize the (near-collapsed) backbone features ----
    # Without this the head's InfoNCE is stuck at ln(batch): the raw features
    # share a huge constant offset and look identical under cosine similarity.
    fmean, fstd = compute_feature_stats(model, train_ds_cache, device,
                                        batch_size=args.batch_size, num_workers=args.num_workers)
    model.set_feature_stats(fmean.to(device), fstd.to(device))
    print(f"feature stats: |mean|={fmean.norm():.2f} std(mean over dims)={fstd.mean():.4f}")

    # ---- train ----
    train(model, train_dataset, device,
          epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
          weight_decay=args.weight_decay, num_workers=args.num_workers,
          hold_ds=hold_dataset)

    # ---- embed all pools (per-modality) ----
    q_cache = make_dataset(query_inf, args.cache_dir)
    t_cache = make_dataset(target_inf, args.cache_dir)
    q_emb = embed_all(model, q_cache, device, batch_size=args.batch_size,
                      num_workers=args.num_workers, encoder="query")
    t_emb = embed_all(model, t_cache, device, batch_size=args.batch_size,
                      num_workers=args.num_workers, encoder="target")

    # ---- diagnostic: labelled-train MRR (biased) ----
    tr_q = {str(p["query_id"]): str(p["query_id"]) for p in train_pairs}
    # embed train images with both heads for diagnostic
    tr_qe = embed_all(model, train_ds_cache, device, batch_size=args.batch_size,
                      num_workers=args.num_workers, encoder="query")
    tr_te = embed_all(model, train_ds_cache, device, batch_size=args.batch_size,
                      num_workers=args.num_workers, encoder="target")
    qids_tr = sorted({str(p["query_id"]) for p in train_pairs})
    tids_tr = sorted({str(p["target_id"]) for p in train_pairs})
    gold = {str(p["query_id"]): str(p["target_id"]) for p in train_pairs}
    S_tr, q_ord, t_ord = cosine_sim({q: tr_qe[q] for q in qids_tr},
                                    {t: tr_te[t] for t in tids_tr})
    print(f"\n=== labelled-train diagnostic (biased) Q={len(q_ord)} G={len(t_ord)} ===")
    print(f"top-1 conflict rate: {top1_conflict_rate(S_tr):.3f}")
    print(f"  argmax   MRR={mrr(rank_argmax(S_tr, q_ord, t_ord), gold):.4f}")
    print(f"  sinkhorn MRR={mrr(rank_sinkhorn(S_tr, q_ord, t_ord, args.sinkhorn_tau, args.sinkhorn_iter), gold):.4f}")
    print(f"  hungari. MRR={mrr(rank_hungarian(S_tr, q_ord, t_ord), gold):.4f}")

    # ---- per-pool submissions ----
    methods = {"argmax": [], "sinkhorn": [], "hungarian": []}
    sim_cache = {}
    print("\n=== per-pool ranking ===")
    for idx, ps in enumerate(prediction_sets):
        qe = {qid: q_emb[qid] for qid in ps["queries"]}
        te = {tid: t_emb[tid] for tid in ps["targets"]}
        S, qids, tids = cosine_sim(qe, te)
        sim_cache[f"S_{idx}"] = S
        sim_cache[f"qids_{idx}"] = np.array(qids)
        sim_cache[f"tids_{idx}"] = np.array(tids)
        print(f"  pool {idx}: Q={len(qids)} G={len(tids)} conflict={top1_conflict_rate(S):.3f}")
        methods["argmax"].extend(rank_argmax(S, qids, tids))
        methods["sinkhorn"].extend(rank_sinkhorn(S, qids, tids, args.sinkhorn_tau, args.sinkhorn_iter))
        methods["hungarian"].extend(rank_hungarian(S, qids, tids))

    np.savez(args.out_dir / f"sim_{args.mode}.npz", **sim_cache)
    for name, rows in methods.items():
        out = args.out_dir / f"ft_{args.mode}_{name}_submission.csv"
        write_submission(out, rows)
        print(f"  {name}: {len(rows)} rows -> {out}")
    torch.save({"logit_scale": float(model.logit_scale.exp().item())}, args.out_dir / f"ft_{args.mode}_meta.pt")


if __name__ == "__main__":
    main()
