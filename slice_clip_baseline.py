from __future__ import annotations
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "monai>=1.5.0",
#   "torch>=2.7.0",
#   "numpy>=2.0",
#   "nibabel>=5.3",
#   "tqdm>=4.67",
# ]
# ///

"""
Tiny 2D slice CLIP baseline for the Brain MRI Cross-Modal Retrieval Challenge.

The script demonstrates:
- MONAI loading, channel handling, 1 mm spacing, intensity scaling, slice
  extraction, resizing, typing, and PersistentDataset caching.
- A tiny random image-noise augmentation during training.
- Two small 2D CNN encoders trained with a CLIP-style in-batch contrastive loss.
- Submission generation by ranking gallery targets by embedding similarity.

Run from this folder with uv. This trains only on the labelled dataset1
training pairs, then writes one combined submission containing validation and
test queries for dataset1, dataset2, and dataset3:

    DATA_ROOT=/path/to/kaggle_dataset

    uv run slice_clip_baseline.py \
      --data-root "$DATA_ROOT" \
      --train-pair-csv "$DATA_ROOT/dataset1/train_pairs.csv" \
      --query-csv "$DATA_ROOT/dataset1/val_queries.csv" \
      --gallery-csv "$DATA_ROOT/dataset1/val_gallery.csv" \
      --query-csv "$DATA_ROOT/dataset1/test_queries.csv" \
      --gallery-csv "$DATA_ROOT/dataset1/test_gallery.csv" \
      --query-csv "$DATA_ROOT/dataset2/val_queries.csv" \
      --gallery-csv "$DATA_ROOT/dataset2/val_gallery.csv" \
      --query-csv "$DATA_ROOT/dataset2/test_queries.csv" \
      --gallery-csv "$DATA_ROOT/dataset2/test_gallery.csv" \
      --query-csv "$DATA_ROOT/dataset3/val_queries.csv" \
      --gallery-csv "$DATA_ROOT/dataset3/val_gallery.csv" \
      --query-csv "$DATA_ROOT/dataset3/test_queries.csv" \
      --gallery-csv "$DATA_ROOT/dataset3/test_gallery.csv" \
      --out slice_clip_submission.csv
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import PersistentDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandGaussianNoised,
    ScaleIntensityd,
    Spacingd,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CONFIG = {
    "seed": 20260626,
    "cache_dir": ".monai_persistent",
    "spacing_mm": (1.0, 1.0, 1.0),
    "slice_positions": (0.35, 0.50, 0.65),
    "image_size": 96,
    "noise_probability": 0.3,
    "noise_std": 0.05,
    "epochs": 500,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "embedding_dim": 128,
    "encoder_hidden_dim": 512,
    "similarity_scale": 5.0,
    "max_grad_norm": 1.0,
    "num_workers": 0,
    "device": (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    ),
}


class SliceStackd(MapTransform):
    """MONAI map transform for three representative 2D slices."""

    def __init__(self, keys: str | list[str], positions: tuple[float, ...], image_size: int) -> None:
        """Remember where to sample slices and how large to resize them."""
        super().__init__(keys)
        self.positions = positions
        self.image_size = image_size

    def __call__(self, data: dict) -> dict:
        """Replace each 3D volume by a resized 3-channel slice stack."""
        d = dict(data)
        for key in self.key_iterator(d):
            volume = torch.as_tensor(d[key]).float()
            if volume.ndim != 4:
                raise ValueError(f"Expected channel-first 3D volume, got shape {tuple(volume.shape)}")
            volume = volume[0]
            finite = torch.where(torch.isfinite(volume), volume, torch.zeros_like(volume))
            nonzero_counts = torch.count_nonzero(finite, dim=(0, 1))
            occupied = torch.nonzero(nonzero_counts, as_tuple=False).flatten()
            if len(occupied) == 0:
                z_values = [volume.shape[-1] // 2] * len(self.positions)
            else:
                z_min = int(occupied[0])
                z_max = int(occupied[-1])
                z_values = [round(z_min + position * (z_max - z_min)) for position in self.positions]

            slices = torch.stack([volume[:, :, int(np.clip(z, 0, volume.shape[-1] - 1))] for z in z_values])
            slices = torch.nan_to_num(slices, nan=0.0, posinf=0.0, neginf=0.0)
            slices = F.interpolate(
                slices.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            d[key] = slices
        return d


class TinySliceEncoder(nn.Module):
    """A deliberately small 2D image encoder."""

    def __init__(self, embedding_dim: int) -> None:
        """Build a small CNN that keeps enough layout to memorize tiny sets."""
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        pooled_size = int(CONFIG["image_size"]) // 8
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * pooled_size * pooled_size, int(CONFIG["encoder_hidden_dim"])),
            nn.ReLU(),
            nn.Linear(int(CONFIG["encoder_hidden_dim"]), embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a 3-slice image stack to an embedding."""
        return self.projection(self.features(x))


class SliceCLIP(nn.Module):
    """Encode query and target images into one shared embedding space."""

    def __init__(self, embedding_dim: int) -> None:
        """Create two tiny encoders and a fixed similarity scale."""
        super().__init__()
        self.query_encoder = TinySliceEncoder(embedding_dim)
        self.target_encoder = TinySliceEncoder(embedding_dim)
        self.similarity_scale = float(CONFIG["similarity_scale"])

    def forward(self, query: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return all query-target similarities in the batch."""
        query_embedding = self.encode_query(query)
        target_embedding = self.encode_target(target)
        return self.similarity_scale * query_embedding @ target_embedding.T

    def encode_query(self, query: torch.Tensor) -> torch.Tensor:
        """Encode query images and normalize the embeddings."""
        return F.normalize(self.query_encoder(query), dim=1)

    def encode_target(self, target: torch.Tensor) -> torch.Tensor:
        """Encode target images and normalize the embeddings."""
        return F.normalize(self.target_encoder(target), dim=1)


class PairImageDataset(Dataset):
    """Training pairs backed by MONAI-cached slice stacks."""

    def __init__(self, pairs: list[dict[str, str | Path]], image_dataset: PersistentDataset) -> None:
        """Keep only positive pairs; each batch supplies in-batch negatives."""
        self.image_dataset = image_dataset
        self.id_to_index = {row["id"]: index for index, row in enumerate(image_dataset.data)}
        self.examples = [(str(pair["query_id"]), str(pair["target_id"])) for pair in pairs]

    def __len__(self) -> int:
        """Return the number of positive pair examples."""
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch a query-target image pair; training noise is sampled here."""
        query_id, target_id = self.examples[index]
        query_image = self._image(query_id)
        target_image = self._image(target_id)
        return query_image, target_image

    def _image(self, image_id: str) -> torch.Tensor:
        """Load one cached slice stack by image ID."""
        item = self.image_dataset[self.id_to_index[image_id]]
        return torch.as_tensor(item["image"]).float()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV into plain dictionaries."""
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def resolve_image_path(data_root: Path, image_path: str) -> Path:
    """Resolve manifest image paths relative to the dataset root."""
    path = Path(image_path)
    resolved = path if path.is_absolute() else data_root / path
    if resolved.exists():
        return resolved
    # The manifests list .nii.gz, but the released volumes are uncompressed .nii
    # (or vice versa); fall back to whichever extension exists on disk.
    text = str(resolved)
    alternate = Path(text[:-3]) if text.endswith(".nii.gz") else Path(text + ".gz")
    return alternate if alternate.exists() else resolved


def load_manifest_images(data_root: Path, csv_paths: list[Path], id_column: str, image_column: str) -> dict[str, Path]:
    """Load image IDs and paths from one or more manifest CSVs."""
    images: dict[str, Path] = {}
    for csv_path in csv_paths:
        for row in read_csv(csv_path):
            images[row[id_column]] = resolve_image_path(data_root, row[image_column])
    return images


def load_training_pairs(
    data_root: Path,
    train_pair_csvs: list[Path],
    train_query_csvs: list[Path],
    train_gallery_csvs: list[Path],
    train_label_csvs: list[Path],
) -> list[dict[str, str | Path]]:
    """Collect positive training pairs from pair CSVs or labelled pools."""
    pairs: list[dict[str, str | Path]] = []
    for csv_path in train_pair_csvs:
        for row in read_csv(csv_path):
            pairs.append(
                {
                    "query_id": row["query_id"],
                    "target_id": row["target_id"],
                    "query_path": resolve_image_path(data_root, row["query_image"]),
                    "target_path": resolve_image_path(data_root, row["target_image"]),
                }
            )

    query_images = load_manifest_images(data_root, train_query_csvs, "query_id", "query_image")
    target_images = load_manifest_images(data_root, train_gallery_csvs, "target_id", "target_image")
    for csv_path in train_label_csvs:
        for row in read_csv(csv_path):
            query_id = row["query_id"]
            target_id = row["target_id"]
            if query_id not in query_images or target_id not in target_images:
                continue
            pairs.append(
                {
                    "query_id": query_id,
                    "target_id": target_id,
                    "query_path": query_images[query_id],
                    "target_path": target_images[target_id],
                }
            )

    if not pairs:
        raise ValueError("No training pairs found. Pass --train-pair-csv or query/gallery/label CSVs.")
    return pairs


def collect_prediction_sets(
    data_root: Path,
    query_csvs: list[Path],
    gallery_csvs: list[Path],
) -> list[dict[str, dict[str, Path]]]:
    """Pair up query and gallery manifests for ranking."""
    if len(query_csvs) != len(gallery_csvs):
        raise ValueError("--query-csv and --gallery-csv must be passed the same number of times")

    sets = []
    for query_csv, gallery_csv in zip(query_csvs, gallery_csvs):
        queries = load_manifest_images(data_root, [query_csv], "query_id", "query_image")
        targets = load_manifest_images(data_root, [gallery_csv], "target_id", "target_image")
        sets.append({"queries": queries, "targets": targets})
    return sets


def monai_transform(augment: bool) -> Compose:
    """Build the MONAI pipeline; optional noise happens after cached slices."""
    deterministic = [
        LoadImaged(keys="image", image_only=True),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RAS", labels=None),
        Spacingd(keys="image", pixdim=CONFIG["spacing_mm"], mode="bilinear"),
        ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
        SliceStackd(
            keys="image",
            positions=tuple(CONFIG["slice_positions"]),
            image_size=int(CONFIG["image_size"]),
        ),
        EnsureTyped(keys="image"),
    ] # monai persistent dataset knows which transforms are deterministic and caches the results after the last deterministic transform automatically
    random_augmentation = []
    if augment:
        random_augmentation.append(
            RandGaussianNoised(
                keys="image",
                prob=float(CONFIG["noise_probability"]),
                mean=0.0,
                std=float(CONFIG["noise_std"]),
            )
        )
    return Compose(deterministic + random_augmentation)


def make_image_dataset(images: dict[str, Path], example_root: Path, augment: bool) -> PersistentDataset:
    """Create the MONAI dataset; it handles cache reads/writes itself."""
    cache_dir = example_root / str(CONFIG["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"image": str(path), "id": image_id} for image_id, path in sorted(images.items())]
    return PersistentDataset(data=rows, transform=monai_transform(augment=augment), cache_dir=cache_dir)


def train_model(train_dataset: PairImageDataset) -> SliceCLIP:
    """Train the tiny dual encoder with in-batch negatives."""
    torch.manual_seed(int(CONFIG["seed"]))
    device = torch.device(str(CONFIG["device"]))
    model = SliceCLIP(int(CONFIG["embedding_dim"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(CONFIG["learning_rate"]))
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        train_dataset,
        batch_size=int(CONFIG["batch_size"]),
        shuffle=True,
        num_workers=int(CONFIG["num_workers"]),
    )

    model.train()
    for epoch in range(1, int(CONFIG["epochs"]) + 1):
        total_loss = 0.0
        total_seen = 0
        for query_batch, target_batch in loader:
            query_batch = query_batch.to(device)
            target_batch = target_batch.to(device)
            labels = torch.arange(len(query_batch), device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(query_batch, target_batch)
            loss = (loss_fn(logits, labels) + loss_fn(logits.T, labels)) / 2
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(CONFIG["max_grad_norm"]))
            optimizer.step()
            total_loss += float(loss.item()) * len(query_batch)
            total_seen += len(query_batch)
        print(f"epoch {epoch:03d} loss={total_loss / max(total_seen, 1):.5f}")
    return model


@torch.no_grad()
def embed_images(model: SliceCLIP, image_dataset: PersistentDataset, encoder: str) -> dict[str, np.ndarray]:
    """Run one encoder once for each deterministic query/gallery image."""
    device = next(model.parameters()).device
    loader = DataLoader(image_dataset, batch_size=int(CONFIG["batch_size"]), shuffle=False)
    embeddings: dict[str, np.ndarray] = {}
    model.eval()
    for batch in tqdm(loader, desc="Embedding images"):
        image_batch = torch.as_tensor(batch["image"]).float().to(device)
        if encoder == "query":
            embedding_batch = model.encode_query(image_batch).detach().cpu().numpy().astype(np.float32)
        elif encoder == "target":
            embedding_batch = model.encode_target(image_batch).detach().cpu().numpy().astype(np.float32)
        else:
            raise ValueError(f"Unknown encoder: {encoder}")
        for image_id, embedding in zip(batch["id"], embedding_batch):
            embeddings[str(image_id)] = embedding
    return embeddings


@torch.no_grad()
def rank_targets(
    model: SliceCLIP,
    query_embeddings: dict[str, np.ndarray],
    target_embeddings: dict[str, np.ndarray],
) -> list[dict[str, str]]:
    """Score every query against every target and return rankings."""
    device = next(model.parameters()).device
    model.eval()
    rows: list[dict[str, str]] = []
    target_ids = sorted(target_embeddings)
    target_matrix = torch.from_numpy(np.stack([target_embeddings[target_id] for target_id in target_ids])).to(device)

    for query_id in tqdm(sorted(query_embeddings), desc="Ranking queries"):
        query_embedding = torch.from_numpy(query_embeddings[query_id]).to(device)
        scores = (model.similarity_scale * query_embedding.unsqueeze(0) @ target_matrix.T).squeeze(0)
        scores = scores.detach().cpu().numpy()
        ranking = [target_ids[index] for index in np.argsort(-scores)]
        rows.append({"query_id": query_id, "target_id_ranking": " ".join(ranking)})
    return rows


def write_submission(path: Path, rows: list[dict[str, str]]) -> None:
    """Write Kaggle-style query rankings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "target_id_ranking"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse the few CSV paths needed to train and rank."""
    parser = argparse.ArgumentParser(description="Simple MONAI + middle-slice CNN retrieval baseline.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-pair-csv", type=Path, action="append", default=[])
    parser.add_argument("--train-query-csv", type=Path, action="append", default=[])
    parser.add_argument("--train-gallery-csv", type=Path, action="append", default=[])
    parser.add_argument("--train-label-csv", type=Path, action="append", default=[])
    parser.add_argument("--query-csv", type=Path, action="append", required=True)
    parser.add_argument("--gallery-csv", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=Path("slice_cnn_submission.csv"))
    return parser.parse_args()


def main() -> None:
    """Wire the baseline together from CSVs to submission file."""
    args = parse_args()
    random.seed(int(CONFIG["seed"]))
    np.random.seed(int(CONFIG["seed"]))

    example_root = Path(__file__).resolve().parent
    data_root = args.data_root.resolve()
    train_pairs = load_training_pairs(
        data_root,
        args.train_pair_csv,
        args.train_query_csv,
        args.train_gallery_csv,
        args.train_label_csv,
    )
    prediction_sets = collect_prediction_sets(data_root, args.query_csv, args.gallery_csv)

    train_images: dict[str, Path] = {}
    for pair in train_pairs:
        train_images[str(pair["query_id"])] = Path(pair["query_path"])
        train_images[str(pair["target_id"])] = Path(pair["target_path"])

    inference_images: dict[str, Path] = {}
    for prediction_set in prediction_sets:
        inference_images.update(prediction_set["queries"])
        inference_images.update(prediction_set["targets"])

    print(
        json.dumps(
            {
                "config": CONFIG,
                "num_train_images": len(train_images),
                "num_inference_images": len(inference_images),
                "num_train_pairs": len(train_pairs),
            },
            indent=2,
        )
    )
    train_image_dataset = make_image_dataset(train_images, example_root, augment=True)
    inference_image_dataset = make_image_dataset(inference_images, example_root, augment=False)
    train_dataset = PairImageDataset(train_pairs, train_image_dataset)
    model = train_model(train_dataset)
    query_embeddings_all = embed_images(model, inference_image_dataset, encoder="query")
    target_embeddings_all = embed_images(model, inference_image_dataset, encoder="target")

    submission_rows: list[dict[str, str]] = []
    for prediction_set in prediction_sets:
        query_embeddings = {image_id: query_embeddings_all[image_id] for image_id in prediction_set["queries"]}
        target_embeddings = {image_id: target_embeddings_all[image_id] for image_id in prediction_set["targets"]}
        submission_rows.extend(rank_targets(model, query_embeddings, target_embeddings))

    write_submission(args.out, submission_rows)
    print(f"Wrote {len(submission_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
