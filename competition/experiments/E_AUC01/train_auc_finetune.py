"""Independent natural-BCE and 4+/12- ranking streams, frozen BN statistics."""
import argparse
import math
import os
import time
from common import *
from torch import nn
from torch.utils.data import DataLoader, Sampler
from auc_pairwise_loss import pairwise_auc_loss
from evaluate_auc import evaluate


class RankingBatchSampler(Sampler):
    def __init__(self, labels, steps, seed):
        labels = np.asarray(labels)
        self.pos, self.neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
        self.steps, self.seed = steps, seed
        if len(self.pos) < 4 or len(self.neg) < 12:
            raise ValueError('Need >=4 positive and >=12 negative training images')

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.steps):
            indices = np.concatenate((rng.choice(self.pos, 4, replace=False),
                                      rng.choice(self.neg, 12, replace=False)))
            rng.shuffle(indices)
            yield indices.tolist()


def freeze_bn_stats(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()  # Weight/bias remain trainable.


def bn_buffers(model):
    return {name: value.detach().clone() for name, value in model.named_buffers()
            if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}


def grad_norm(parameters):
    values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def train_batch_step(model, optimizer, scaler, batch_a, batch_b, config, probe, warning):
    """Replay the same batch/RNG after AMP overflow, with exactly one real update."""
    parameters = list(model.parameters())
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    first_losses = None
    for attempt in range(13):
        if attempt:
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
        optimizer.zero_grad(set_to_none=True)
        target_a = batch_a['label'].cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logits_a = model(batch_a['image'].cuda(non_blocking=True), batch_a['age'].cuda(non_blocking=True))
            bce = nn.functional.binary_cross_entropy_with_logits(logits_a.float(), target_a)
        if not torch.isfinite(bce):
            raise FloatingPointError('Nonfinite BCE; this is not a recoverable gradient-scale overflow')
        scaler.scale(bce).backward()
        snapshots = [p.grad.detach().clone() if p.grad is not None else None for p in parameters] if probe else None
        bce_norm = grad_norm(parameters) / scaler.get_scale() if probe else None
        target_b = batch_b['label'].cuda(non_blocking=True)
        if int((target_b == 1).sum()) != 4 or int((target_b == 0).sum()) != 12:
            raise ValueError('Ranking stream lost the agreed 4+/12- composition')
        # C0 also consumes the ranking forward's Dropout/StochasticDepth draws.
        with torch.autocast('cuda', dtype=torch.float16):
            logits_b = model(batch_b['image'].cuda(non_blocking=True), batch_b['age'].cuda(non_blocking=True))
            rank = pairwise_auc_loss(logits_b, target_b)
        if not torch.isfinite(rank):
            raise FloatingPointError('Nonfinite ranking loss')
        losses = (float(bce.detach()), float(rank.detach()))
        if first_losses is None:
            first_losses = losses
        elif not np.allclose(losses, first_losses, atol=1e-6, rtol=1e-5):
            raise RuntimeError('AMP replay changed forward losses despite restoring the same RNG')
        if config['rank_lambda']:
            scaler.scale(config['rank_lambda'] * rank).backward()
        weighted_rank_norm = None
        if probe:
            squared = []
            for param, snapshot in zip(parameters, snapshots):
                if param.grad is not None:
                    change = param.grad.detach().float() - (snapshot.float() if snapshot is not None else 0)
                    squared.append(change.square().sum())
            weighted_rank_norm = float(torch.stack(squared).sum().sqrt()) / scaler.get_scale()
            del snapshots
        scaler.unscale_(optimizer)
        norm = float(nn.utils.get_total_norm([p.grad for p in parameters if p.grad is not None],
                                             error_if_nonfinite=False))
        if not math.isfinite(norm):
            previous_scale = scaler.get_scale()
            if previous_scale <= 1 or attempt == 12:
                raise FloatingPointError('Nonfinite gradients persist at low scale or exhaust 12 AMP retries')
            # unscale_ recorded found_inf; update lowers the scale WITHOUT an optimizer step.
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                raise FloatingPointError('Nonfinite gradient norm is not explained by AMP overflow')
            warning({'event': 'amp_overflow_retry', 'attempt': attempt+1,
                     'scale_before': previous_scale, 'scale_after': scaler.get_scale(),
                     'bce': losses[0], 'rank': losses[1], 'same_batch_and_rng_replayed': True,
                     'optimizer_updated': False})
            del logits_a, logits_b, bce, rank
            continue
        nn.utils.clip_grad_norm_(parameters, config['grad_clip'], error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        return {'bce': losses[0], 'rank': losses[1], 'norm': norm, 'amp_retries': attempt,
                'bce_grad_norm': bce_norm, 'weighted_rank_grad_norm': weighted_rank_norm,
                'amp_scale': scaler.get_scale()}
    raise RuntimeError('Unreachable AMP replay state')


def train(args):
    config = json.loads(Path(args.config).read_text())
    config['seed'] = args.seed
    run = Path(args.experiment_root)
    name = f"{config['name']}_s{args.seed}"
    logs = run / 'training_logs' / name
    logs.mkdir(parents=True, exist_ok=True)
    complete = logs / 'complete.json'
    if complete.exists():
        return json.loads(complete.read_text())
    write_json(logs / 'config.json', config)
    seed_all(args.seed)
    torch.set_num_threads(config['cpu_threads'])
    torch.backends.cudnn.benchmark = True
    model, initial = load_model(args.initial)
    optimizer = torch.optim.AdamW([
        {'params': list(model.encoder.parameters()), 'lr': config['backbone_lr']},
        {'params': list(model.head.parameters()), 'lr': config['head_lr']}], weight_decay=config['weight_decay'])
    scaler = torch.amp.GradScaler('cuda')
    start_epoch, best = 0, None
    last_path = run / 'checkpoints' / f'{name}_last.pt'
    best_path = run / 'checkpoints' / f'{name}_best.pt'
    if last_path.exists():
        previous = torch.load(last_path, map_location='cpu', weights_only=False)
        if previous['experiment_config'] != config or previous['initial_sha256'] != sha256(args.initial):
            raise ValueError('Resume configuration/initial checkpoint differs')
        model.load_state_dict(previous['model'])
        optimizer.load_state_dict(previous['optimizer'])
        scaler.load_state_dict(previous['scaler'])
        start_epoch, best = previous['epoch'], previous['best_metrics']
        random.setstate(previous['rng']['python'])
        np.random.set_state(previous['rng']['numpy'])
        torch.set_rng_state(previous['rng']['torch'])
        torch.cuda.set_rng_state_all(previous['rng']['cuda'])
    frozen = bn_buffers(model)
    parameters = list(model.parameters())
    initial_hash = sha256(args.initial)
    initial_metrics = json.loads((run / 'predictions/baseline.metrics.json').read_text())
    started = time.monotonic()
    for epoch in range(start_epoch, config['epochs']):
        natural = dataset(args.data_root, 'train', initial, training=True)
        ranking = dataset(args.data_root, 'train', initial, training=True)
        valid = dataset(args.data_root, 'valid', initial)
        labels = [int(r['True_Label']) for r in ranking.rows]
        generator_a = torch.Generator().manual_seed(args.seed + epoch)
        generator_b = torch.Generator().manual_seed(args.seed + 100000 + epoch)
        # Independent workers and generators keep stream A augmentations identical across lambdas.
        options = dict(num_workers=config['workers_per_stream'], pin_memory=True,
                       worker_init_fn=seed_worker, persistent_workers=False)
        loader_a = DataLoader(natural, batch_size=16, shuffle=True, generator=generator_a, **options)
        sampler_b = RankingBatchSampler(labels, len(loader_a), args.seed + 200000 + epoch)
        loader_b = DataLoader(ranking, batch_sampler=sampler_b, generator=generator_b, **options)
        validloader = DataLoader(valid, batch_size=32, shuffle=False, **options)
        warmup_steps = max(1, math.ceil(len(loader_a) * .05))
        total_steps = len(loader_a) * config['epochs']
        model.train()
        freeze_bn_stats(model)
        sums = dict(bce=0., rank=0., total=0.)
        seen = 0
        max_grad, overflow_retries = 0., 0
        epoch_started = time.monotonic()
        order_hash = hashlib.sha256()
        tensor_hash = hashlib.sha256()
        data_wait_seconds, compute_seconds = 0., 0.
        paired_iterator = iter(zip(loader_a, loader_b))
        for step in range(len(loader_a)):
            wait_started = time.monotonic()
            batch_a, batch_b = next(paired_iterator)
            data_wait_seconds += time.monotonic()-wait_started
            global_step = epoch * len(loader_a) + step
            if global_step < warmup_steps:
                factor = (global_step + 1) / warmup_steps
            else:
                progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
                factor = (1 + math.cos(math.pi * progress)) / 2
            optimizer.param_groups[0]['lr'] = config['backbone_lr'] * factor
            optimizer.param_groups[1]['lr'] = config['head_lr'] * factor
            for image_name in batch_a['name']:
                order_hash.update((image_name + '\n').encode())
            if step < 4:
                tensor_hash.update(batch_a['image'].numpy().tobytes())
            probe = step % 500 == 0
            def warning(value):
                value.update(experiment=name, epoch=epoch+1, step=step+1)
                print(json.dumps(value), flush=True)
                with (logs / 'amp_retries.jsonl').open('a') as f:
                    f.write(json.dumps(value) + '\n')
            compute_started = time.monotonic()
            outcome = train_batch_step(model, optimizer, scaler, batch_a, batch_b, config, probe, warning)
            compute_seconds += time.monotonic()-compute_started
            norm = outcome['norm']
            max_grad = max(max_grad, norm)
            overflow_retries += outcome['amp_retries']
            n = len(batch_a['label'])
            sums['bce'] += outcome['bce'] * n
            sums['rank'] += outcome['rank'] * n
            sums['total'] += (outcome['bce'] + config['rank_lambda'] * outcome['rank']) * n
            seen += n
            if step % 100 == 0 or step + 1 == len(loader_a):
                progress = {'event': 'training', 'experiment': name, 'pid': os.getpid(), 'epoch': epoch+1, 'step': step+1,
                            'steps': len(loader_a), 'bce': sums['bce']/seen, 'rank': sums['rank']/seen,
                            'total_loss': sums['total']/seen, 'grad_norm_before_clip': norm,
                            'gpu_memory_gb': torch.cuda.max_memory_allocated()/1e9,
                            'elapsed_s': time.monotonic()-epoch_started, 'amp_scale': outcome['amp_scale'],
                            'amp_overflow_retries_epoch': overflow_retries,
                            'mean_data_wait_ms': data_wait_seconds/(step+1)*1000,
                            'mean_training_step_ms': compute_seconds/(step+1)*1000,
                            'lossless_data_cache': natural.__class__.__name__ == 'CachedFundusDataset'}
                if probe:
                    bce_norm, weighted_rank_norm = outcome['bce_grad_norm'], outcome['weighted_rank_grad_norm']
                    progress.update(bce_grad_norm=bce_norm, weighted_rank_grad_norm=weighted_rank_norm,
                                    rank_to_bce_grad_ratio=weighted_rank_norm/max(bce_norm, 1e-12))
                print(json.dumps(progress), flush=True)
                write_json(logs / 'progress.json', progress)
                with (logs / 'steps.jsonl').open('a') as f:
                    f.write(json.dumps(progress) + '\n')
        if any(not torch.equal(value, dict(model.named_buffers())[key]) for key, value in frozen.items()):
            raise RuntimeError('BN running statistics changed despite freezing')
        metrics, names, ys, ps, zs = evaluate(model, validloader)
        metrics.update(experiment=name, rank_lambda=config['rank_lambda'], seed=args.seed, epoch=epoch+1,
                       delta_auroc=metrics['auroc']-initial_metrics['auroc'],
                       train_bce=sums['bce']/seen, train_rank=sums['rank']/seen,
                       train_total_loss=sums['total']/seen, max_grad_norm_before_clip=max_grad,
                       amp_skipped_steps=0, amp_overflow_retries=overflow_retries, bn_statistics_unchanged=True,
                       data_wait_seconds=data_wait_seconds, compute_seconds=compute_seconds,
                       natural_sample_order_sha256=order_hash.hexdigest(),
                       natural_first_four_batches_tensor_sha256=tensor_hash.hexdigest(),
                       epoch_elapsed_s=time.monotonic()-epoch_started)
        save_predictions(run / 'predictions' / f'{name}_epoch{epoch+1:02d}.csv', names, ys, ps, zs)
        if better(metrics, best):
            best = metrics
            save_torch(best_path, {'model': model.state_dict(), 'config': initial['config'],
                                  'age_stats': initial['age_stats'], 'epoch': epoch+1, 'metrics': metrics,
                                  'experiment_config': config, 'initial_sha256': initial_hash})
            save_predictions(run / 'predictions' / f'{name}_best.csv', names, ys, ps, zs)
            write_json(logs / 'best_metrics.json', metrics)
        save_torch(last_path, {'model': model.state_dict(), 'config': initial['config'],
                              'age_stats': initial['age_stats'], 'epoch': epoch+1, 'metrics': metrics,
                              'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                              'experiment_config': config, 'initial_sha256': initial_hash,
                              'best_metrics': best,
                              'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                                      'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}})
        write_json(logs / f'epoch{epoch+1:02d}.json', metrics)
        print(json.dumps({'event': 'validation_complete', **metrics}), flush=True)
        del loader_a, loader_b, validloader
    result = {'experiment': name, 'best_metrics': best, 'training_elapsed_s': time.monotonic()-started,
              'checkpoint': str(best_path), 'initial_sha256': initial_hash}
    write_json(complete, result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--initial', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--experiment-root', required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    try:
        train(args)
    except BaseException as exc:
        write_json(Path(args.experiment_root) / 'training_logs' / 'last_failure.json',
                   {'type': type(exc).__name__, 'message': str(exc), 'config': args.config, 'seed': args.seed})
        raise


if __name__ == '__main__':
    main()
