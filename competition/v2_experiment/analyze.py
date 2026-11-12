"""Stage selection, fusion grid search, correlations and paired bootstrap."""
import argparse
import itertools
from common import *


def average_ranks(values):
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
    return ranks


def bootstrap_weights(y, repeats, seed):
    rng = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    weights = np.zeros((repeats, len(y)), dtype=np.uint16)
    for index in range(repeats):
        weights[index, positive] = np.bincount(rng.integers(0, len(positive), len(positive)),
                                                minlength=len(positive))
        weights[index, negative] = np.bincount(rng.integers(0, len(negative), len(negative)),
                                                minlength=len(negative))
    return weights


def weighted_auc(y, p, weights):
    order = np.argsort(p, kind='stable')
    sorted_p, sorted_y = p[order], y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_p)) + 1]
    output = []
    for offset in range(0, len(weights), 100):
        w = weights[offset:offset + 100, order].astype(np.float64)
        positive = np.add.reduceat(w * (sorted_y == 1)[None, :], starts, axis=1)
        negative = np.add.reduceat(w * (sorted_y == 0)[None, :], starts, axis=1)
        wins = (positive * (negative.cumsum(axis=1) - .5 * negative)).sum(axis=1)
        output.extend((wins / (positive.sum(axis=1) * negative.sum(axis=1))).tolist())
    return np.asarray(output)


def paired_bootstrap(y, reference, candidate, protocol):
    weights = bootstrap_weights(y, protocol['fusion']['bootstrap_repeats'],
                                protocol['fusion']['bootstrap_seed'])
    difference = weighted_auc(y, candidate, weights) - weighted_auc(y, reference, weights)
    return {'repeats': len(weights), 'seed': protocol['fusion']['bootstrap_seed'],
            'observed_delta_auroc': compute_metrics(y, candidate)['auroc'] - compute_metrics(y, reference)['auroc'],
            'mean_delta_auroc': float(difference.mean()),
            'ci95_low': float(np.quantile(difference, .025)),
            'ci95_high': float(np.quantile(difference, .975)),
            'fraction_delta_gt_zero': float((difference > 0).mean()),
            'unit': 'image', 'paired': True, 'stratified': True,
            'caution': 'Conditional on validation-set model and fusion selection; not patient-level external validation.'}


def prediction_path(root, name):
    if name == 'E00':
        return root.parent / 'E_AUC01/predictions/baseline.csv'
    return root / 'runs' / name / 'best_valid.csv'


def load_aligned(root, names):
    data, reference_names, reference_labels = {}, None, None
    for name in names:
        current_names, labels, probabilities = read_predictions(prediction_path(root, name))
        if reference_names is None:
            reference_names, reference_labels = current_names, labels
        if current_names != reference_names or not np.array_equal(labels, reference_labels):
            raise ValueError(f'Validation rows are not aligned: {name}')
        if len(labels) != 5849 or labels.sum() != 191:
            raise ValueError('Expected the complete official validation set')
        data[name] = probabilities
    return reference_names, reference_labels, data


def first_stage(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    primary = protocol['first_stage']
    names, y, data = load_aligned(root, ['E00', *primary])
    rows = []
    for name, p in data.items():
        rows.append({'experiment': name, **compute_metrics(y, p)})
    baseline = next(row for row in rows if row['experiment'] == 'E00')
    candidate = max((row for row in rows if row['experiment'] != 'E00'),
                    key=lambda row: (row['auroc'], row['ap']))
    confirmation_name = candidate['experiment'].removesuffix('_s42') + '_s43'
    confirmation_exists = prediction_path(root, confirmation_name).exists()
    comparison = paired_bootstrap(y, data['E00'], data[candidate['experiment']], protocol)
    stable = False
    confirmation = None
    if confirmation_exists:
        _, _, confirmation_data = load_aligned(root, ['E00', confirmation_name])
        confirmation = {'experiment': confirmation_name,
                        **compute_metrics(y, confirmation_data[confirmation_name])}
        confirmation['bootstrap'] = paired_bootstrap(y, data['E00'],
                                                     confirmation_data[confirmation_name], protocol)
        stable = (candidate['auroc'] > baseline['auroc'] and confirmation['auroc'] > baseline['auroc']
                  and comparison['ci95_low'] > 0 and confirmation['bootstrap']['ci95_low'] > 0
                  and min(candidate['ap'], confirmation['ap']) >= baseline['ap'] - .01)
    strategy = {'gamma': False, 'weight_decay': .0001, 'exclude_bias_norm_from_wd': False,
                'source': 'E00 original settings'}
    if stable:
        base = candidate['experiment'].removesuffix('_s42')
        if base == 'E00_GAMMA':
            strategy.update(gamma=True, source='stable E00_GAMMA')
        elif base.startswith('WD'):
            wd = {'WD0': .0001, 'WD1': .0005, 'WD2': .001}[base]
            strategy.update(weight_decay=wd, exclude_bias_norm_from_wd=True, source=f'stable {base}')
    result = {'baseline': baseline, 'trials': rows[1:], 'candidate': candidate,
              'candidate_bootstrap': comparison, 'confirmation_required': candidate['auroc'] > baseline['auroc'],
              'confirmation_name': confirmation_name, 'confirmation': confirmation,
              'stable_improvement': stable, 'second_stage_strategy': strategy,
              'test_consulted': False}
    write_json(root / 'first_stage_selection.json', result)
    return result


def add_candidate(records, name, components, weights, y, predictions, family):
    probability = sum(weight * predictions[component] for component, weight in zip(components, weights))
    metrics = compute_metrics(y, probability)
    records.append({'name': name, 'family': family, 'components': components,
                    'weights': [float(value) for value in weights], **metrics})
    return probability


def fusion(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    run_names = ['E448_s42', 'C00_s42']
    if (root / 'runs/E512_s42/complete.json').exists():
        run_names.append('E512_s42')
    first = json.loads((root / 'first_stage_selection.json').read_text())
    if first['stable_improvement']:
        run_names.extend([first['candidate']['experiment'], first['confirmation_name']])
    # Confirmation seeds are stability checks, not extra candidates in the
    # validation-set fusion search. This avoids choosing the luckier seed.
    names, y, predictions = load_aligned(root, ['E00', *run_names])
    records, generated = [], {}
    for name in ['E00', *run_names]:
        generated[name] = predictions[name]
        add_candidate(records, name, [name], [1.], y, predictions, 'single')

    dual_weights = protocol['fusion']['dual_e00_weights']
    for other, family in [('E448_s42', 'F01'), ('C00_s42', 'F03')]:
        for w0 in dual_weights:
            name = f'{family}_E00_{w0:.2f}_{other}_{1-w0:.2f}'
            generated[name] = add_candidate(records, name, ['E00', other], [w0, 1-w0],
                                             y, predictions, family)
    if 'E512_s42' in predictions:
        for w0 in dual_weights:
            name = f'F02_E00_{w0:.2f}_E512_{1-w0:.2f}'
            generated[name] = add_candidate(records, name, ['E00', 'E512_s42'], [w0, 1-w0],
                                             y, predictions, 'F02')
    if first['stable_improvement']:
        enhanced = first['candidate']['experiment']
        for w0 in dual_weights:
            name = f'F05_E00_{w0:.2f}_{enhanced}_{1-w0:.2f}'
            generated[name] = add_candidate(records, name, ['E00', enhanced], [w0, 1-w0],
                                             y, predictions, 'F05')

    components = ['E00', 'E448_s42', 'C00_s42']
    coarse = []
    for a in np.arange(0, 1.0001, .2):
        for b in np.arange(0, 1-a+.0001, .2):
            c = round(1 - a - b, 10)
            if c < -.0001:
                continue
            name = f'F04c_{a:.2f}_{b:.2f}_{c:.2f}'
            probability = add_candidate(records, name, components, [a, b, c], y, predictions, 'F04-coarse')
            generated[name] = probability
            coarse.append(records[-1])
    coarse_best = max(coarse, key=lambda row: (row['auroc'], row['ap']))
    center = np.asarray(coarse_best['weights'])
    for a_i in range(21):
        for b_i in range(21 - a_i):
            a, b = a_i * .05, b_i * .05
            c = 1 - a - b
            weight = np.asarray([a, b, c])
            if np.max(np.abs(weight - center)) > .200001:
                continue
            name = f'F04r_{a:.2f}_{b:.2f}_{c:.2f}'
            generated[name] = add_candidate(records, name, components, weight, y, predictions, 'F04-refine')

    best_single = max((row for row in records if row['family'] == 'single'),
                      key=lambda row: (row['auroc'], row['ap']))
    selected = max(records, key=lambda row: (row['auroc'], row['ap']))
    final_p = generated[selected['name']]
    correlation = {}
    for left, right in itertools.combinations(['E00', *run_names], 2):
        correlation[f'{left}__{right}'] = {
            'pearson': float(np.corrcoef(predictions[left], predictions[right])[0, 1]),
            'spearman': float(np.corrcoef(average_ranks(predictions[left]),
                                         average_ranks(predictions[right]))[0, 1])}
    reference_p = generated[best_single['name']]
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    old_order = np.sign(reference_p[positive, None] - reference_p[negative][None, :])
    new_order = np.sign(final_p[positive, None] - final_p[negative][None, :])
    error_change = {
        'positive_misses_corrected_at_05': int(((reference_p < .5) & (final_p >= .5) & (y == 1)).sum()),
        'new_positive_misses_at_05': int(((reference_p >= .5) & (final_p < .5) & (y == 1)).sum()),
        'negative_false_positives_corrected_at_05': int(((reference_p >= .5) & (final_p < .5) & (y == 0)).sum()),
        'new_negative_false_positives_at_05': int(((reference_p < .5) & (final_p >= .5) & (y == 0)).sum()),
        'inverted_pairs_corrected': int(((old_order < 0) & (new_order > 0)).sum()),
        'new_inverted_pairs': int(((old_order > 0) & (new_order < 0)).sum())}
    result = {'selected': selected, 'best_single': best_single,
              'bootstrap_vs_best_single': paired_bootstrap(y, reference_p, final_p, protocol),
              'bootstrap_vs_e00': paired_bootstrap(y, predictions['E00'], final_p, protocol),
              'correlations': correlation, 'error_change_vs_best_single': error_change,
              'coarse_three_model_best': coarse_best, 'test_consulted': False}
    write_json(root / 'fusion_selection.json', result)
    save_predictions(root / 'predictions/selected_valid.csv', names, y.tolist(), final_p.tolist(),
                     np.log(np.clip(final_p, 1e-12, 1-1e-12) / np.clip(1-final_p, 1e-12, 1)).tolist())
    fields = ['name', 'family', 'auroc', 'ap', 'pr_auc_trapezoid', 'recall_at_spec_0.95',
              'recall_at_spec_0.98', 'components', 'weights']
    with (root / 'fusion_grid.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in sorted(records, key=lambda item: (item['auroc'], item['ap']), reverse=True):
            writer.writerow({**row, 'components': json.dumps(row['components']), 'weights': json.dumps(row['weights'])})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['first', 'fusion'])
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    value = first_stage(Path(args.root)) if args.phase == 'first' else fusion(Path(args.root))
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)
