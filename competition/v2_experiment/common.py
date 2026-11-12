"""Shared utilities for the frozen V2.0 experiment."""
from pathlib import Path
import csv
import hashlib
import json
import os
import random
import sys

HERE = Path(__file__).resolve().parent
for candidate in (HERE.parents[1], HERE.parent):
    if (candidate / 'glaucoma_project/glaucoma').is_dir():
        PROJECT, BASE_PROJECT = candidate, candidate / 'glaucoma_project'
        break
    if (candidate / 'glaucoma').is_dir():
        PROJECT = BASE_PROJECT = candidate
        break
else:
    raise RuntimeError('Cannot locate the existing glaucoma project')
sys.path.insert(0, str(BASE_PROJECT))

import numpy as np
import torch
from core import fit_age, read_rows, seed_worker


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def save_predictions(path, names, labels, probabilities, logits):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Image_name', 'True_Label', 'Probability', 'Logit'])
        writer.writerows(zip(names, labels, probabilities, logits))
    temporary.replace(path)


def read_predictions(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    names = [row['Image_name'] for row in rows]
    labels = np.asarray([int(row['True_Label']) for row in rows], dtype=np.int8)
    probabilities = np.asarray([float(row['Probability']) for row in rows], dtype=np.float64)
    if len(names) != len(set(names)) or not np.isfinite(probabilities).all():
        raise ValueError(f'Invalid prediction file: {path}')
    return names, labels, probabilities


def compute_metrics(labels, probabilities):
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(probabilities, dtype=np.float64)
    if len(y) != len(p) or set(np.unique(y)) != {0, 1} or not np.isfinite(p).all():
        raise ValueError('Invalid labels or probabilities')
    order = np.argsort(-p, kind='stable')
    sorted_p, sorted_y = p[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_p)), len(sorted_p) - 1]
    tp_curve = sorted_y.cumsum()[ends].astype(np.float64)
    fp_curve = (1 - sorted_y).cumsum()[ends].astype(np.float64)
    positives, negatives = float(y.sum()), float((y == 0).sum())
    recall = tp_curve / positives
    precision = tp_curve / (tp_curve + fp_curve)
    tpr = np.r_[0., recall]
    fpr = np.r_[0., fp_curve / negatives]
    thresholds = np.r_[np.inf, sorted_p[ends]]
    ap = float((np.diff(np.r_[0., recall]) * precision).sum())
    pr_auc = float(np.trapezoid(np.r_[1., precision], np.r_[0., recall]))
    predicted = p >= .5
    tn = int(((y == 0) & ~predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    tp = int(((y == 1) & predicted).sum())
    output = {'auroc': float(np.trapezoid(tpr, fpr)), 'ap': ap,
              'pr_auc_trapezoid': pr_auc, 'n': int(len(y)),
              'positive': int(y.sum()), 'confusion_05': [[tn, fp], [fn, tp]],
              'recall_05': tp / max(1, tp + fn), 'precision_05': tp / max(1, tp + fp),
              'f1_05': 2 * tp / max(1, 2 * tp + fp + fn)}
    for specificity in (.95, .98):
        eligible = np.flatnonzero(fpr <= 1 - specificity + 1e-12)
        index = eligible[np.argmax(tpr[eligible])]
        output[f'recall_at_spec_{specificity:.2f}'] = float(tpr[index])
        output[f'threshold_at_spec_{specificity:.2f}'] = (float(thresholds[index])
                                                               if np.isfinite(thresholds[index]) else None)
    return output


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def assert_official_data(root):
    result = {}
    for split, expected, positives in [('train', 51396, 1657), ('valid', 5849, 191), ('test', 5855, None)]:
        rows = read_rows(root, split)
        if len(rows) != expected:
            raise ValueError(f'Unexpected {split} count: {len(rows)}')
        names = [row['Image_name'] for row in rows]
        if len(names) != len(set(names)):
            raise ValueError(f'Duplicate names in {split}')
        record = {'rows': len(rows), 'csv_sha256': sha256(Path(root) / f'{split}.csv')}
        if positives is not None:
            actual = sum(int(row['True_Label']) for row in rows)
            if actual != positives:
                raise ValueError(f'Unexpected {split} positive count: {actual}')
            record['positive'] = actual
        result[split] = record
    return result
