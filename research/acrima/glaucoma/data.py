from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .common import read_csv, sha256_file


def dataset_rows(root: Path, expected_counts=None, verify_images=False):
    root = Path(root).resolve()
    rows = read_csv(root / "manifest.csv")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs.")
    if len({row["image_path"] for row in rows}) != len(rows):
        raise ValueError("Duplicate image paths.")
    groups = {}
    for row in rows:
        row["label"] = int(row["label"])
        if row["label"] not in (0, 1) or row["split"] not in {"train", "val", "test"}:
            raise ValueError("Invalid label or outer split.")
        if not row["group_id"]:
            raise ValueError("Every sample must have a similarity group ID.")
        image = (root / row["image_path"]).resolve()
        if root not in image.parents or not image.is_file():
            raise ValueError(f"Missing or out-of-root image: {image}")
        expected_class = "1_positive" if row["label"] else "0_negative"
        if Path(row["image_path"]).parts[:2] != (row["split"], expected_class):
            raise ValueError("Directory and explicit label/split disagree.")
        for field in ("group_id", "sha256", "pixel_sha256"):
            value = row.get(field)
            if not value:
                raise ValueError(f"Missing {field}.")
            key = (field, value)
            if key in groups and groups[key] != row["split"]:
                raise ValueError(f"Cross-split leakage in {field}: {value}")
            groups[key] = row["split"]
        if verify_images:
            if sha256_file(image) != row["sha256"]:
                raise ValueError(f"Image hash changed: {image}")
            with Image.open(image) as source:
                source.convert("RGB").load()
    for split in ("train", "val", "test"):
        subset = [row for row in rows if row["split"] == split]
        csv_rows = read_csv(root / f"{split}.csv")
        normalized = [{**row, "label": str(row["label"])} for row in subset]
        if normalized != csv_rows:
            raise ValueError(f"{split}.csv differs from manifest.csv.")
        if {row["label"] for row in subset} != {0, 1}:
            raise ValueError(f"{split} must contain both classes.")
        if expected_counts and len(subset) != expected_counts[split]:
            raise ValueError(f"Unexpected {split} sample count.")
    return rows


def make_folds(rows, n_splits=3, seed=42):
    if any(row["split"] != "train" for row in rows):
        raise ValueError("Only outer train samples may enter development folds.")
    y = np.array([row["label"] for row in rows])
    # Seeded group identifiers break assignment ties reproducibly. The splitter
    # retains its size/class-count ordering rather than shuffling group counts.
    groups = np.array([hashlib.sha256(f"{seed}|{row['group_id']}".encode()).hexdigest() for row in rows])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
    assignments = np.full(len(rows), -1, dtype=int)
    for fold, (training, validation) in enumerate(splitter.split(np.zeros(len(rows)), y, groups)):
        if set(groups[training]) & set(groups[validation]):
            raise ValueError("Similarity group crosses a fold boundary.")
        if len(np.unique(y[training])) != 2 or len(np.unique(y[validation])) != 2:
            raise ValueError("Each fold must contain both classes in both partitions.")
        assignments[validation] = fold
    if (assignments < 0).any():
        raise ValueError("Incomplete fold assignment.")
    return [{"sample_id": row["sample_id"], "label": row["label"], "group_id": row["group_id"],
             "fold": int(fold)} for row, fold in zip(rows, assignments)]


def image_transform(size, augmentation=False):
    operations = [transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC, antialias=True)]
    if augmentation:
        # No random crops; reflection padding limits artificial black rotation corners.
        padding = max(1, int(np.ceil(size * 0.1)))
        operations += [transforms.Pad(padding, padding_mode="reflect"),
                       transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR),
                       transforms.CenterCrop(size), transforms.RandomHorizontalFlip(0.5),
                       transforms.ColorJitter(brightness=0.1, contrast=0.1)]
    return transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class FundusDataset(Dataset):
    def __init__(self, root, rows, size=224, augmentation=False):
        self.root, self.rows = Path(root), list(rows)
        self.transform = image_transform(size, augmentation)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["image_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, float(row["label"]), row["sample_id"]


class MergeSingletonBatchSampler(Sampler):
    """Never silently discard a sample, and avoid a final singleton BN batch."""
    def __init__(self, count, batch_size, seed, shuffle):
        self.count, self.batch_size, self.seed, self.shuffle = count, batch_size, seed, shuffle

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(self.count, generator=torch.Generator().manual_seed(self.seed)).tolist()
        else:
            order = list(range(self.count))
        batches = [order[i:i + self.batch_size] for i in range(0, self.count, self.batch_size)]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        yield from batches

    def __len__(self):
        full, remainder = divmod(self.count, self.batch_size)
        return full if remainder == 1 and full else full + bool(remainder)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(root, rows, experiment, config, device, training=False, epoch_seed=42, features=False):
    options = config["training"]
    dataset = FundusDataset(root, rows, experiment["image_size"], training and experiment["augmentation"])
    batch_size = options["feature_batch_size"] if features else options["batch_size"]
    workers = options["workers"]
    kwargs = {"num_workers": workers, "pin_memory": device.type == "cuda",
              "worker_init_fn": seed_worker, "generator": torch.Generator().manual_seed(epoch_seed)}
    if workers:
        kwargs["prefetch_factor"] = 2
    if training:
        sampler = MergeSingletonBatchSampler(len(rows), batch_size, epoch_seed, True)
        return DataLoader(dataset, batch_sampler=sampler, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
