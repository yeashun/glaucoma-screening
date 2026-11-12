from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .common import digest_object, write_csv, write_json
from .data import make_loader
from .metrics import binary_metrics, selection_key
from .models import (autocast_context, build_model, configure_stage, get_device, make_scaler,
                     move_images, save_torch, seed_everything, train_mode)


@torch.inference_mode()
def predict_logits(model, root, rows, experiment, config, device):
    model.eval()
    logits, labels, ids = [], [], []
    loader = make_loader(root, rows, experiment, config, device)
    for images, target, sample_ids in loader:
        with autocast_context(config, device):
            output = model(move_images(images, device, config)).flatten()
        logits.extend(output.float().cpu().numpy())
        labels.extend(target.numpy())
        ids.extend(sample_ids)
    expected_ids = [row["sample_id"] for row in rows]
    if ids != expected_ids:
        raise RuntimeError("Prediction ordering does not match the manifest.")
    result = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite predictions.")
    return result


def extract_features(root, rows, experiment, config, output):
    """Frozen backbone cache is keyed by actual weights, transforms and image hashes."""
    seed_everything(config["seed"], config)
    device = get_device(config)
    model, provenance = build_model(experiment, output, dropout=config["training"]["dropout"])
    model.classifier = nn.Identity()
    model.requires_grad_(False)
    model.eval().to(device)
    if config["training"]["channels_last"]:
        model.to(memory_format=torch.channels_last)
    identity = {"weights": provenance, "image_size": experiment["image_size"], "transform": "resize_bicubic_full_rgb_imagenet_v1",
                "samples": [(r["sample_id"], r["sha256"]) for r in rows],
                "amp": config["training"]["amp"], "device_type": device.type}
    key = digest_object(identity)
    path = Path(output) / "cache" / f"features_{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            if cached["ids"].tolist() != [r["sample_id"] for r in rows]:
                raise ValueError("Cached feature IDs differ.")
            result = cached["features"].copy()
        if not np.isfinite(result).all() or len(result) != len(rows):
            raise ValueError("Invalid feature cache.")
        del model
        return result, provenance
    features, ids = [], []
    loader = make_loader(root, rows, experiment, config, device, features=True)
    with torch.inference_mode():
        for images, _, sample_ids in tqdm(loader, desc="Frozen features", leave=False):
            with autocast_context(config, device):
                feature = model(move_images(images, device, config))
            features.append(feature.float().cpu().numpy())
            ids.extend(sample_ids)
    result = np.concatenate(features)
    if ids != [r["sample_id"] for r in rows] or not np.isfinite(result).all():
        raise ValueError("Invalid extracted features.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, features=result, ids=np.array(ids))
    temporary.replace(path)
    write_json(path.with_suffix(".json"), identity)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, provenance


def fit_network(root, training, validation, experiment, config, output, work, seed,
                final_choice=None):
    """CV saves its best epoch; final fitting uses the locked CV stage/epoch only.

    Completed fold artifacts can be reused by the caller. An interrupted fold is
    restarted from its deterministic seed, never silently resumed with a new LR.
    """
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    seed_everything(seed, config)
    device = get_device(config)
    model, provenance = build_model(experiment, output, dropout=config["training"]["dropout"])
    model.to(device)
    if config["training"]["channels_last"]:
        model.to(memory_format=torch.channels_last)
    options = config["training"]
    scaler = make_scaler(config, device)
    criterion = nn.BCEWithLogitsLoss()
    best_key, best_choice, best_metrics = None, None, None
    logs, stage_details = [], []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    completed_epochs = 0
    final_stage = final_choice["stage"] if final_choice else "finetune"
    for stage in ("warmup", "finetune"):
        if final_choice and final_stage == "warmup" and stage == "finetune":
            break
        stage_details.append(configure_stage(model, stage))
        epochs = options["warmup_epochs"] if stage == "warmup" else options["finetune_epochs"]
        if final_choice and stage == final_stage:
            epochs = final_choice["epoch_in_stage"]
        head = [p for p in model.classifier.parameters() if p.requires_grad]
        if stage == "warmup":
            parameter_groups = [{"params": head, "lr": options["head_warmup_lr"]}]
        else:
            parameter_groups = [
                {"params": [p for p in model.features.parameters() if p.requires_grad], "lr": options["backbone_lr"]},
                {"params": head, "lr": options["head_finetune_lr"]},
            ]
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=options["weight_decay"])
        # Keep the CV cosine horizon when final fitting stops at the median epoch.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=options["finetune_epochs"]) if stage == "finetune" else None
        stale = 0
        for epoch in range(1, epochs + 1):
            epoch_seed = seed + completed_epochs
            torch.manual_seed(epoch_seed)
            np.random.seed(epoch_seed)
            import random
            random.seed(epoch_seed)
            train_mode(model, stage)
            loader = make_loader(root, training, experiment, config, device, training=True, epoch_seed=epoch_seed)
            total_loss = 0.0
            progress = tqdm(loader, desc=f"{experiment['name']} {stage} {epoch}/{epochs}", leave=False)
            for images, targets, _ in progress:
                images = move_images(images, device, config)
                targets = targets.to(device, dtype=torch.float32, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(config, device):
                    logits = model(images).flatten()
                    loss = criterion(logits, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss. Set amp=off in a new run to diagnose.")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach()) * len(targets)
                progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
            completed_epochs += 1
            record = {"stage": stage, "epoch_in_stage": epoch, "train_nll": total_loss / len(training),
                      "learning_rates": [g["lr"] for g in optimizer.param_groups]}
            if validation:
                z = predict_logits(model, root, validation, experiment, config, device)
                metrics = binary_metrics([r["label"] for r in validation], z, bins=config["metrics"]["ece_bins"])
                record.update({key: metrics[key] for key in ("auroc", "auprc", "average_precision", "nll", "brier")})
                key = selection_key(metrics)
                if best_key is None or key > best_key:
                    best_key, best_metrics = key, metrics
                    best_choice = {"stage": stage, "epoch_in_stage": epoch}
                    save_torch(work / "best.pt", {"state_dict": model.state_dict(), "experiment": experiment,
                               "pretrained": provenance, "choice": best_choice})
                    write_csv(work / "best_predictions.csv", [
                        {"sample_id": r["sample_id"], "label": r["label"], "group_id": r["group_id"], "logit": float(v)}
                        for r, v in zip(validation, z)])
                    stale = 0
                else:
                    stale += 1
                print(f"{experiment['name']} {stage} {epoch}: AUROC={metrics['auroc']:.5f} AUPRC={metrics['auprc']:.5f} NLL={metrics['nll']:.5f}", flush=True)
            logs.append(record)
            write_json(work / "history.json", logs)
            if scheduler:
                scheduler.step()
            if validation and stage == "finetune" and stale >= options["patience"]:
                break
    if not validation:
        best_choice = final_choice
        save_torch(work / "model.pt", {"state_dict": model.state_dict(), "experiment": experiment,
                                     "pretrained": provenance, "choice": best_choice})
    result = {"best_choice": best_choice, "metrics": best_metrics, "pretrained": provenance,
              "stages": stage_details, "training_seconds": time.perf_counter() - started,
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
              "seed": seed, "epochs_completed": completed_epochs}
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
