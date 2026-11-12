"""Resumable, configuration-frozen single-model training for V2.0."""
import argparse
import fcntl
import math
import platform
import shutil
import time
from common import *
from data_v2 import CachedDataset
from torch import nn
from torch.utils.data import DataLoader
from core import FundusClassifier


def parameter_groups(model, config):
    groups = {}
    exclude = config.get('exclude_bias_norm_from_wd', False)
    decay = float(config['weight_decay'])
    for name, parameter in model.named_parameters():
        family = 'backbone' if name.startswith('encoder.') else 'head'
        no_decay = exclude and (parameter.ndim <= 1 or name.endswith('.bias'))
        key = (family, no_decay)
        groups.setdefault(key, []).append(parameter)
    output = []
    for (family, no_decay), parameters in groups.items():
        output.append({'params': parameters,
                       'lr': float(config[f'{family}_lr']),
                       'weight_decay': 0. if no_decay else decay,
                       'family': family, 'no_decay': no_decay})
    return output


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    names, labels, probabilities, logits = [], [], [], []
    for batch in loader:
        image = batch['image'].to(device, non_blocking=True)
        age = batch['age'].to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):
            output = model(image, age)
        output = output.float()
        names.extend(batch['name'])
        labels.extend(batch['label'].int().tolist())
        probabilities.extend(output.sigmoid().cpu().tolist())
        logits.extend(output.cpu().tolist())
    return compute_metrics(labels, probabilities), names, labels, probabilities, logits


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'train.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / 'complete.json').exists():
        print(json.dumps({'already_complete': str(run)}), flush=True)
        return

    seed = int(config['seed'])
    seed_all(seed)
    torch.set_num_threads(int(config.get('cpu_threads', 4)))
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available() and not args.smoke:
        raise RuntimeError('Full training requires CUDA')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = read_rows(args.data_root, 'train')
    stats = fit_age(rows)
    last_path = run / 'last.pt'
    checkpoint = torch.load(last_path, map_location='cpu', weights_only=False) if last_path.exists() else None
    if checkpoint and checkpoint['config'] != config:
        raise ValueError('Resume configuration differs from the frozen run configuration')
    if checkpoint:
        stats = checkpoint['age_stats']
    model = FundusClassifier(config, pretrained=checkpoint is None and config.get('pretrained', True)).to(device)
    optimizer = torch.optim.AdamW(parameter_groups(model, config))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    start_epoch, best_metrics = 0, None
    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = int(checkpoint['epoch'])
        best_metrics = checkpoint['best_metrics']
        random.setstate(checkpoint['rng']['python'])
        np.random.set_state(checkpoint['rng']['numpy'])
        torch.set_rng_state(checkpoint['rng']['torch'])
        torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])

    epochs = 1 if args.smoke else int(config['epochs'])
    batch_size = 4 if args.smoke else int(config['batch_size'])
    effective_batch = batch_size if args.smoke else int(config['effective_batch'])
    workers = 2 if args.smoke else int(config.get('workers', 4))
    size = int(config['image_size'])
    limit = 64 if args.smoke else None
    write_json(run / 'config.json', config)
    write_json(run / 'environment.json', {'python': platform.python_version(), 'torch': torch.__version__,
               'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu',
               'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'cache_root': str(args.cache_root)})
    write_json(run / 'age_stats.json', stats)
    started = time.monotonic()
    amp_skipped_total = int(checkpoint.get('amp_skipped_total', 0)) if checkpoint else 0

    for epoch in range(start_epoch, epochs):
        generator = torch.Generator().manual_seed(seed + epoch)
        train = CachedDataset(args.data_root, 'train', stats, args.cache_root, size,
                              training=True, gamma=bool(config.get('gamma', False)), limit=limit)
        valid = CachedDataset(args.data_root, 'valid', stats, args.cache_root, size,
                              training=False, gamma=False, limit=limit)
        options = dict(num_workers=workers, pin_memory=device.type == 'cuda', worker_init_fn=seed_worker,
                       persistent_workers=False)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator, **options)
        valid_loader = DataLoader(valid, batch_size=max(8, batch_size * 2), shuffle=False, **options)
        warmup = epoch < int(config.get('warmup_epochs', 1))
        for name, parameter in model.named_parameters():
            if name.startswith('encoder.'):
                parameter.requires_grad_(not warmup)
        model.train()
        if warmup:
            model.encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        accumulation = max(1, effective_batch // batch_size)
        running_loss, seen, amp_skipped_epoch = 0., 0, 0
        epoch_started = time.monotonic()
        for step, batch in enumerate(train_loader):
            progress = (epoch + step / max(1, len(train_loader))) / epochs
            factor = .05 + .95 * (1 + math.cos(math.pi * progress)) / 2
            for group in optimizer.param_groups:
                group['lr'] = float(config[f"{group['family']}_lr"]) * factor
            image = batch['image'].to(device, non_blocking=True)
            age = batch['age'].to(device, non_blocking=True)
            target = batch['label'].to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                logits = model(image, age)
                loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite forward loss')
            group_start = (step // accumulation) * accumulation
            divisor = min(accumulation, len(train_loader) - group_start)
            scaler.scale(loss / divisor).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                previous_scale = scaler.get_scale()
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.)
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < previous_scale:
                    amp_skipped_epoch += 1
                    amp_skipped_total += 1
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach()) * len(target)
            seen += len(target)
            if step % 100 == 0 or step + 1 == len(train_loader):
                progress_record = {'event': 'train', 'run': run.name, 'epoch': epoch + 1,
                                   'step': step + 1, 'steps': len(train_loader),
                                   'loss': running_loss / seen,
                                   'images_s': seen / max(.001, time.monotonic() - epoch_started),
                                   'amp_skipped_epoch': amp_skipped_epoch,
                                   'gpu_memory_gb': torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0}
                write_json(run / 'progress.json', progress_record)
                print(json.dumps(progress_record), flush=True)

        metrics, names, labels, probabilities, logits = evaluate(model, valid_loader, device)
        metrics.update({'experiment': run.name, 'epoch': epoch + 1, 'image_size': size,
                        'train_loss': running_loss / seen, 'amp_skipped_epoch': amp_skipped_epoch,
                        'amp_skipped_total': amp_skipped_total,
                        'epoch_elapsed_s': time.monotonic() - epoch_started,
                        'total_elapsed_s': time.monotonic() - started})
        improved = best_metrics is None or (metrics['auroc'], metrics['ap']) > (best_metrics['auroc'], best_metrics['ap'])
        if improved:
            best_metrics = metrics
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                 'config': config, 'age_stats': stats, 'epoch': epoch + 1, 'metrics': metrics,
                 'best_metrics': best_metrics, 'amp_skipped_total': amp_skipped_total,
                 'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                         'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}}
        save_checkpoint(last_path, state)
        if improved:
            shutil.copy2(last_path, run / 'best.pt')
            save_predictions(run / 'best_valid.csv', names, labels, probabilities, logits)
            write_json(run / 'best_metrics.json', metrics)
        with (run / 'metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(metrics, allow_nan=False) + '\n')
        write_json(run / 'progress.json', metrics)
        print(json.dumps({'event': 'validation', **metrics}, allow_nan=False), flush=True)
        del train_loader, valid_loader, train, valid
    write_json(run / 'complete.json', {'experiment': run.name, 'best_metrics': best_metrics,
               'epochs': epochs, 'elapsed_s_this_process': time.monotonic() - started,
               'amp_skipped_total': amp_skipped_total, 'completed_unix': time.time()})


if __name__ == '__main__':
    main()
