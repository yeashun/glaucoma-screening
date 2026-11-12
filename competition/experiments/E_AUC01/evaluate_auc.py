"""Full fixed official validation with the original E00 AMP evaluation path."""
import argparse
import time
from common import *
from torch.utils.data import DataLoader
from sklearn.metrics import (average_precision_score, auc, confusion_matrix,
                            precision_recall_curve, roc_auc_score, roc_curve)


def compute_metrics(labels, probabilities, logits=None):
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if set(np.unique(y)) != {0, 1} or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid validation labels/probabilities')
    precision, recall, _ = precision_recall_curve(y, p)
    fpr, tpr, thresholds = roc_curve(y, p)
    tn, fp, fn, tp = confusion_matrix(y, p >= .5, labels=[0, 1]).ravel().tolist()
    result = {'auroc': float(roc_auc_score(y, p)), 'ap': float(average_precision_score(y, p)),
              'pr_auc_trapezoid': float(auc(recall, precision)), 'n': int(len(y)), 'positive': int(y.sum()),
              'confusion_05': [[tn, fp], [fn, tp]], 'recall_05': tp / max(1, tp+fn),
              'precision_05': tp / max(1, tp+fp), 'f1_05': 2*tp / max(1, 2*tp+fp+fn)}
    for specificity in (.95, .98):
        allowed = np.flatnonzero(fpr <= 1 - specificity + 1e-12)
        index = allowed[np.argmax(tpr[allowed])]
        threshold = thresholds[index]
        result[f'recall_at_spec_{specificity:.2f}'] = float(tpr[index])
        result[f'threshold_at_spec_{specificity:.2f}'] = float(threshold) if np.isfinite(threshold) else None
    if logits is not None:
        z = np.asarray(logits, dtype=float)
        result['valid_bce'] = float((np.logaddexp(0, z) - y*z).mean())
        result['valid_pairwise_rank'] = float(np.logaddexp(0, z[y == 0, None] - z[y == 1][None, :]).mean())
    return result


@torch.inference_mode()
def evaluate(model, loader, device='cuda'):
    model.eval()
    names, labels, probabilities, logits = [], [], [], []
    for step, batch in enumerate(loader):
        image = batch['image'].to(device, non_blocking=True)
        age = batch['age'].to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = model(image, age)
        output = output.float()
        names.extend(batch['name'])
        labels.extend(batch['label'].int().tolist())
        probabilities.extend(output.sigmoid().cpu().tolist())
        logits.extend(output.cpu().tolist())
        if step % 50 == 0:
            print(json.dumps({'event': 'validation_progress', 'images': len(names), 'total': len(loader.dataset)}), flush=True)
    return compute_metrics(labels, probabilities, logits), names, labels, probabilities, logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(4)
    model, state = load_model(args.checkpoint)
    valid = dataset(args.data_root, 'valid', state)
    loader = DataLoader(valid, batch_size=32, shuffle=False, num_workers=args.workers, pin_memory=True)
    started = time.time()
    metrics, names, labels, probabilities, logits = evaluate(model, loader)
    metrics['elapsed_s'] = time.time() - started
    save_predictions(args.output, names, labels, probabilities, logits)
    write_json(Path(args.output).with_suffix('.metrics.json'), metrics)
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
