"""Isolated copy of the verified E00 model and deterministic data primitives."""
from pathlib import Path
import csv
import random
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import models


def read_rows(root, split):
    with (Path(root) / f'{split}.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'Empty split: {split}')
    names = [row['Image_name'] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f'Duplicate names in {split}')
    return rows


def age_value(row):
    try:
        age = float(row['Age'])
        return age if np.isfinite(age) and 0 < age <= 120 else None
    except (ValueError, TypeError, KeyError):
        return None


def fit_age(rows):
    values = [age_value(row) for row in rows if age_value(row) is not None]
    median = float(np.median(values))
    filled = np.asarray([age_value(row) if age_value(row) is not None else median for row in rows])
    return {'median': median, 'mean': float(filled.mean()), 'std': max(float(filled.std()), 1.)}


def encode_age(row, stats):
    age = age_value(row)
    return torch.tensor([((stats['median'] if age is None else age) - stats['mean']) / stats['std'],
                         float(age is None)], dtype=torch.float32)


def valid_bbox(image):
    thumb = image.copy()
    thumb.thumbnail((160, 160))
    array = np.asarray(thumb)
    mask = array.max(axis=2) > 8
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return (0, 0, image.width, image.height)
    sx, sy = image.width / thumb.width, image.height / thumb.height
    pad = max(2, int(min(image.size) * .025))
    return (max(0, int(xs.min() * sx) - pad), max(0, int(ys.min() * sy) - pad),
            min(image.width, int((xs.max() + 1) * sx) + pad),
            min(image.height, int((ys.max() + 1) * sy) + pad))


def open_rgb(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
        image.load()
    return image


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class Encoder(nn.Module):
    def __init__(self, name, pretrained=False):
        super().__init__()
        if name == 'efficientnet_v2_s':
            net = models.efficientnet_v2_s(
                weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
            self.features = net.features
            self.dim = net.classifier[1].in_features
            self.norm = nn.Identity()
        elif name == 'convnext_tiny':
            net = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
            self.features = net.features
            self.dim = net.classifier[2].in_features
            self.norm = net.classifier[0]
        else:
            raise ValueError(name)

    def forward(self, image):
        feature_map = self.features(image)
        return self.norm(feature_map.mean(dim=(-2, -1), keepdim=True)).flatten(1)


class FundusClassifier(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.get('backbone', 'efficientnet_v2_s'), pretrained)
        self.use_age = config.get('use_age', True)
        self.age_mlp = (nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
                        if self.use_age else None)
        dimension = self.encoder.dim + (16 if self.use_age else 0)
        self.head = nn.Sequential(nn.Linear(dimension, 128), nn.ReLU(), nn.Dropout(.2), nn.Linear(128, 1))

    def forward(self, image, age):
        features = [self.encoder(image)]
        if self.use_age:
            features.append(self.age_mlp(age))
        return self.head(torch.cat(features, dim=1)).flatten()
