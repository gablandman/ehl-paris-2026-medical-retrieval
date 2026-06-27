"""Zero-shot retrieval with the BrainIAC 3D brain-MRI foundation model.

Replaces the tiny SliceCLIP encoder with BrainIAC (`eugenehp/brainiac`), a
MONAI ViT-B/16 pretrained with SimCLR on structural brain MRI. We embed every
query (T1 post-contrast) and gallery (T2) volume with the SAME frozen encoder,
then rank by cosine similarity and apply the same bijection-aware rerankers
(argmax / Sinkhorn / Hungarian) as rerank_baseline.py.

This is the frozen, no-fine-tune experiment: a direct apples-to-apples swap of
the encoder, to see whether a domain-matched 3D backbone beats the 0.555 result.

Preprocessing follows the model card: trilinear resize to 96^3 + per-volume
z-score over nonzero voxels. NOTE: BrainIAC was trained on skull-stripped,
MNI-registered, N4-corrected brains; we do NOT do those steps here (our volumes
still have skull/neck). Consistent preprocessing of query and gallery is what
matters for retrieval, but this is a known off-distribution caveat to revisit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, PersistentDataset
from monai.networks.nets import ViT
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Resized,
)
from safetensors.torch import load_file
from tqdm import tqdm

from rerank_baseline import (
    cosine_sim,
    rank_argmax,
    rank_hungarian,
    rank_sinkhorn,
    top1_conflict_rate,
    write_submission,
)
from slice_clip_baseline import CONFIG, collect_prediction_sets


def build_model(weights_path: Path, device: torch.device) -> ViT:
    """Construct the MONAI ViT-B/16^3 and load the BrainIAC backbone weights."""
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
    return model.to(device).eval()


def preprocess() -> Compose:
    """Model-card preprocessing: RAS, resize to 96^3, z-score over nonzero voxels."""
    return Compose([
        LoadImaged(keys="image", image_only=True),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        Resized(keys="image", spatial_size=(96, 96, 96), mode="trilinear", align_corners=False),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys="image"),
    ])


def make_dataset(images: dict[str, Path], cache_dir: Path) -> PersistentDataset:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"image": str(path), "id": image_id} for image_id, path in sorted(images.items())]
    return PersistentDataset(data=rows, transform=preprocess(), cache_dir=cache_dir)


@torch.no_grad()
def embed_all(model: ViT, dataset: PersistentDataset, device: torch.device,
              batch_size: int, num_workers: int) -> dict[str, np.ndarray]:
    """Embed every volume; BrainIAC's feature is the first patch token (768-d)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    embeddings: dict[str, np.ndarray] = {}
    for batch in tqdm(loader, desc="BrainIAC embedding"):
        x = torch.as_tensor(batch["image"]).float().to(device)
        feats = model(x)[0][:, 0]  # (B, 768) -- first patch token per model card
        feats = torch.nn.functional.normalize(feats, dim=1)
        feats = feats.detach().cpu().numpy().astype(np.float32)
        for image_id, vec in zip(batch["id"], feats):
            embeddings[str(image_id)] = vec
    return embeddings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BrainIAC frozen retrieval + bijection reranking.")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--query-csv", type=Path, action="append", required=True)
    p.add_argument("--gallery-csv", type=Path, action="append", required=True)
    p.add_argument("--weights", type=Path, default=None,
                   help="Path to backbone.safetensors; if omitted, download from HF.")
    p.add_argument("--out-dir", type=Path, default=Path("brainiac_out"))
    p.add_argument("--cache-dir", type=Path, default=Path(".brainiac_cache"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--sinkhorn-tau", type=float, default=10.0)
    p.add_argument("--sinkhorn-iter", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_root = args.data_root.resolve()

    weights = args.weights
    if weights is None:
        from huggingface_hub import hf_hub_download
        weights = Path(hf_hub_download(repo_id="eugenehp/brainiac", filename="backbone.safetensors"))
    print(f"weights: {weights}")

    prediction_sets = collect_prediction_sets(data_root, args.query_csv, args.gallery_csv)
    inference_images: dict[str, Path] = {}
    for ps in prediction_sets:
        inference_images.update(ps["queries"])
        inference_images.update(ps["targets"])
    print(json.dumps({
        "num_inference_images": len(inference_images),
        "num_prediction_sets": len(prediction_sets),
        "device": str(CONFIG["device"]),
    }, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else str(CONFIG["device"]))
    model = build_model(weights, device)
    dataset = make_dataset(inference_images, args.cache_dir)
    emb = embed_all(model, dataset, device, args.batch_size, args.num_workers)

    methods: dict[str, list] = {"argmax": [], "sinkhorn": [], "hungarian": []}
    sim_cache: dict[str, np.ndarray] = {}
    print("\n=== Per-pool conflict + ranking (BrainIAC embeddings) ===")
    for idx, ps in enumerate(prediction_sets):
        q_emb = {qid: emb[qid] for qid in ps["queries"]}
        t_emb = {tid: emb[tid] for tid in ps["targets"]}
        S, qids, tids = cosine_sim(q_emb, t_emb)
        sim_cache[f"S_{idx}"] = S
        sim_cache[f"qids_{idx}"] = np.array(qids)
        sim_cache[f"tids_{idx}"] = np.array(tids)
        print(f"  pool {idx}: Q={len(qids)} G={len(tids)} top-1 conflict rate={top1_conflict_rate(S):.3f}")
        methods["argmax"].extend(rank_argmax(S, qids, tids))
        methods["sinkhorn"].extend(rank_sinkhorn(S, qids, tids, args.sinkhorn_tau, args.sinkhorn_iter))
        methods["hungarian"].extend(rank_hungarian(S, qids, tids))

    np.savez(args.out_dir / "similarities.npz", **sim_cache)
    for name, rows in methods.items():
        out = args.out_dir / f"brainiac_{name}_submission.csv"
        write_submission(out, rows)
        print(f"  {name}: {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
