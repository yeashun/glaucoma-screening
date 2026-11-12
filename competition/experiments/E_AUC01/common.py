"""Shared project imports, atomic artifacts and reproducibility utilities."""
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for candidate in (HERE.parents[1], HERE.parents[1] / 'glaucoma_project'):
    if (candidate / 'glaucoma').is_dir():
        sys.path.insert(1, str(candidate))
        break
else:
    raise RuntimeError('Place experiments/E_AUC01 beside the existing glaucoma project package')

import numpy as np
import torch
from glaucoma.data import FundusDataset, read_rows, seed_worker
from glaucoma.models import FundusClassifier


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def save_torch(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    torch.save(state, temp)
    temp.replace(path)


def save_predictions(path, names, labels, probabilities, logits):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image_name', 'True_Label', 'Probability', 'Logit'])
        writer.writerows(zip(names, labels, probabilities, logits))
    temp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(path, device='cuda'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['config']
    if config['backbone'] != 'efficientnet_v2_s' or config.get('use_age', True) or config['kind'] != 'global':
        raise ValueError('This E-AUC01 run requires the agreed E00 image-only architecture')
    model = FundusClassifier(config, pretrained=False)
    model.load_state_dict(state['model'], strict=True)
    return model.to(device), state


def dataset(root, split, state, training=False):
    ready = HERE / 'performance/cache_ready.json'
    if split in ('train', 'valid') and ready.exists():
        cache = json.loads(ready.read_text())
        if not cache.get('ready'):
            raise RuntimeError('Cache ready marker is invalid')
        cache_root = Path(cache['cache_root'])
        if (cache_root / f'{split}.json').is_file() and (cache_root / f'{split}.npy').is_file():
            from cached_data import CachedFundusDataset
            value = CachedFundusDataset(root, split, state['age_stats'], cache_root,
                                        state['config']['image_size'], training)
            import os
            marker = HERE / 'performance/cache_users' / f'{os.getpid()}.json'
            if not marker.exists():
                write_json(marker, {'pid': os.getpid(), 'cache_root': str(cache_root),
                           'image_size': state['config']['image_size'], 'random_augmentation_unchanged': True})
            return value
        # A copied result archive need not contain this server's private 25GB cache.
        # The original raw-image path produces the same inputs and is always usable.
    return FundusDataset(root, split, state['age_stats'], state['config']['image_size'], training)


def assert_data(root):
    for split, count, positives in [('train', 51396, 1657), ('valid', 5849, 191)]:
        rows = read_rows(root, split)
        labels = [int(r['True_Label']) for r in rows]
        if len(rows) != count or sum(labels) != positives or set(labels) != {0, 1}:
            raise ValueError(f'Unexpected official {split} split')
    return {f'{split}_csv_sha256': sha256(Path(root) / f'{split}.csv') for split in ('train', 'valid')}


def better(metrics, current):
    return current is None or (metrics['auroc'], metrics['ap']) > (current['auroc'], current['ap'])
