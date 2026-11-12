"""Paired stratified bootstrap and positive/negative ordering error audit."""
import argparse
from common import *
from evaluate_auc import compute_metrics


def read_prediction(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    return rows, np.array([int(r['True_Label']) for r in rows]), np.array([float(r['Probability']) for r in rows])


def bootstrap_weights(y, repeats=2000, seed=20261029):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    weights = np.zeros((repeats, len(y)), dtype=np.uint16)
    for index in range(repeats):
        weights[index, pos] = np.bincount(rng.integers(0, len(pos), len(pos)), minlength=len(pos))
        weights[index, neg] = np.bincount(rng.integers(0, len(neg), len(neg)), minlength=len(neg))
    return weights


def weighted_bootstrap_auc(y, p, weights):
    # Weighted Mann-Whitney AUC with exact half credit for ties; no repeated sorting.
    order = np.argsort(p, kind='stable')
    sorted_p = p[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_p)) + 1]
    ys = y[order]
    values = []
    for start in range(0, len(weights), 100):
        w = weights[start:start+100, order].astype(np.float64)
        positive = np.add.reduceat(w * (ys == 1)[None, :], starts, axis=1)
        negative = np.add.reduceat(w * (ys == 0)[None, :], starts, axis=1)
        wins = (positive * (negative.cumsum(axis=1) - .5 * negative)).sum(axis=1)
        values.extend((wins / (positive.sum(axis=1) * negative.sum(axis=1))).tolist())
    return np.asarray(values)


def ordering_audit(root, name, rows, y, base_p, candidate_p):
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    old = np.sign(base_p[positive, None] - base_p[negative][None, :])
    new = np.sign(candidate_p[positive, None] - candidate_p[negative][None, :])
    gains = (new.astype(float) - old) * .5
    net_by_positive = gains.sum(axis=1)
    net_by_negative = gains.sum(axis=0)
    output = Path(root) / 'comparisons'
    output.mkdir(exist_ok=True)
    with (output / f'{name}_positive_changes.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image_name', 'Baseline_Probability', 'Candidate_Probability',
                         'Baseline_Inverted_Negatives', 'Candidate_Inverted_Negatives', 'Net_Pair_Credit_Gain'])
        for index, i in enumerate(positive):
            writer.writerow([rows[i]['Image_name'], base_p[i], candidate_p[i], int((old[index] < 0).sum()),
                             int((new[index] < 0).sum()), net_by_positive[index]])
    with (output / f'{name}_negative_changes.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image_name', 'Baseline_Probability', 'Candidate_Probability', 'Net_Pair_Credit_Gain'])
        for index in np.argsort(-base_p[negative]):
            i = negative[index]
            writer.writerow([rows[i]['Image_name'], base_p[i], candidate_p[i], net_by_negative[index]])
    gross_positive_gain = np.maximum(net_by_positive, 0)
    return {'baseline_inverted_candidate_corrected': int(((old < 0) & (new > 0)).sum()),
            'baseline_correct_candidate_inverted': int(((old > 0) & (new < 0)).sum()),
            'baseline_tied_pairs': int((old == 0).sum()), 'candidate_tied_pairs': int((new == 0).sum()),
            'net_pair_credit_gain': float(gains.sum()),
            'positive_images_with_pair_improvement': int((net_by_positive > 0).sum()),
            'positive_images_with_pair_worsening': int((net_by_positive < 0).sum()),
            'top5_share_of_gross_positive_pair_gain': float(np.sort(gross_positive_gain)[-5:].sum()/max(1., gross_positive_gain.sum())),
            'baseline_positive_p10': float(np.quantile(base_p[positive], .1)),
            'candidate_positive_p10': float(np.quantile(candidate_p[positive], .1)),
            'baseline_negative_p99': float(np.quantile(base_p[negative], .99)),
            'candidate_negative_p99': float(np.quantile(candidate_p[negative], .99))}


def compare(root):
    root = Path(root)
    rows, y, base_p = read_prediction(root / 'predictions/baseline.csv')
    if len(y) != 5849 or y.sum() != 191:
        raise ValueError('Bootstrap requires the original complete validation split')
    weights = bootstrap_weights(y)
    baseline_auc = weighted_bootstrap_auc(y, base_p, weights)
    names = [r['Image_name'] for r in rows]
    baseline_metrics = json.loads((root / 'predictions/baseline.metrics.json').read_text())
    summary = [{'experiment': 'baseline', 'epoch': 5, 'delta_auroc': 0., **baseline_metrics}]
    results = {}
    for path in sorted((root / 'predictions').glob('AUC-C*_s*_best.csv')):
        name = path.stem.removesuffix('_best')
        other_rows, other_y, p = read_prediction(path)
        if [r['Image_name'] for r in other_rows] != names or not np.array_equal(y, other_y):
            raise ValueError('Cannot pair different validation datasets/orders')
        difference = weighted_bootstrap_auc(y, p, weights) - baseline_auc
        metrics = json.loads((root / 'training_logs' / name / 'best_metrics.json').read_text())
        result = {'experiment': name, 'repeats': 2000, 'unit': 'image', 'stratified': True,
                  'paired': True, 'seed': 20261029,
                  'observed_delta_auroc': metrics['auroc'] - baseline_metrics['auroc'],
                  'bootstrap_mean_delta': float(difference.mean()),
                  'ci95_low': float(np.quantile(difference, .025)),
                  'ci95_high': float(np.quantile(difference, .975)),
                  'fraction_delta_gt_zero': float((difference > 0).mean()),
                  'caution': 'Image-level interval conditional on validation-based selection; not patient-level or external validation',
                  'ordering_audit': ordering_audit(root, name, rows, y, base_p, p)}
        results[name] = result
        write_json(root / 'comparisons' / f'{name}_paired_bootstrap.json', result)
        summary.append({**metrics, **{k: result[k] for k in ('ci95_low', 'ci95_high', 'fraction_delta_gt_zero')}})
    fields = ['experiment', 'rank_lambda', 'seed', 'epoch', 'auroc', 'delta_auroc', 'ap', 'pr_auc_trapezoid',
              'recall_at_spec_0.95', 'recall_at_spec_0.98', 'recall_05', 'precision_05', 'f1_05',
              'ci95_low', 'ci95_high', 'fraction_delta_gt_zero', 'train_bce', 'train_rank', 'train_total_loss',
              'amp_overflow_retries', 'amp_skipped_steps', 'max_grad_norm_before_clip']
    with (root / 'results_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary)
    write_json(root / 'comparisons/summary.json', results)
    return results


def check_bootstrap():
    from sklearn.metrics import roc_auc_score
    y = np.array([1, 0, 1, 0, 0, 1, 0])
    p = np.array([.5, .5, .9, .1, .2, .2, .9])
    weights = bootstrap_weights(y, 20, 42)
    values = weighted_bootstrap_auc(y, p, weights)
    for w, value in zip(weights, values):
        assert abs(value - roc_auc_score(y, p, sample_weight=w)) < 1e-12
    return {'weighted_auc_matches_sklearn_including_ties': True, 'replicates_checked': 20}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--experiment-root', required=True)
    args = p.parse_args()
    print(json.dumps(compare(args.experiment_root)), flush=True)
