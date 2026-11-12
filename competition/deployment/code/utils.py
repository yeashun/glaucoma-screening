"""固定预处理、融合与文件校验；predict.py 和 cost.py 共用。"""
import csv
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from preprocessing import preprocess


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def configure(seed=42, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fuse(probabilities, stack):
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
    z = (np.log(p / (1 - p)) - np.asarray(stack['logit_mean'])) / np.asarray(stack['logit_std'])
    score = stack['coefficients'][0] + z @ np.asarray(stack['coefficients'][1:])
    return 1 / (1 + np.exp(-np.clip(score, -40, 40)))


def compare_probabilities(names, actual, reference):
    """同时检查逐图误差、二值标签、样本间排序（含原有并列）。"""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.array([float(reference[n.casefold()]['Probability']) for n in names])
    labels = np.array([int(reference[n.casefold()]['Pred_Label']) for n in names])
    error = np.abs(actual - expected)
    rank_changed = np.zeros(len(names), dtype=bool)
    for i in range(len(names)):
        changed = np.sign(actual[i] - actual) != np.sign(expected[i] - expected)
        if changed.any():
            rank_changed[i] = True
            rank_changed[changed] = True
    label_changed = (actual >= .5).astype(int) != labels
    inconsistent = (error > 1e-3) | label_changed | rank_changed
    return {'samples': len(names), 'inconsistent_images': int(inconsistent.sum()),
            'label_mismatches': int(label_changed.sum()), 'ranking_mismatched_images': int(rank_changed.sum()),
            'max_absolute_probability_error': float(error.max()),
            'inconsistent_names': [name for name, bad in zip(names, inconsistent) if bad],
            'passed': not bool(inconsistent.any())}
