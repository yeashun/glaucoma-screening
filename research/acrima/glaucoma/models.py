from __future__ import annotations

import hashlib
import random
from contextlib import nullcontext
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn
from torchvision import models

from .common import sha256_file
from .protocol import PROJECT


def seed_everything(seed, config):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def get_device(config):
    requested = config["training"]["device"]
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Check the CUDA PyTorch installation, or set device=cpu for checks.")
    return device


def autocast_context(config, device):
    if device.type != "cuda" or config["training"]["amp"] == "off":
        return nullcontext()
    mode = config["training"]["amp"]
    if mode == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 is unsupported on this GPU; use amp=fp16 or off.")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16 if mode == "bf16" else torch.float16)


def make_scaler(config, device):
    enabled = device.type == "cuda" and config["training"]["amp"] == "fp16"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def build_model(experiment, output, pretrained=True, dropout=0.2):
    name = experiment["backbone"]
    weights_enum = models.EfficientNet_B0_Weights if name == "efficientnet_b0" else models.EfficientNet_B2_Weights
    builder = models.efficientnet_b0 if name == "efficientnet_b0" else models.efficientnet_b2
    model = builder(weights=None)
    provenance = {"backbone": name, "weights": "none", "sha256": None}
    if pretrained:
        torch.hub.set_dir(str(Path(output) / "cache" / "torch"))
        local = experiment.get("weights_path")
        if local:
            path = (PROJECT / local).resolve()
            state = load_torch(path)
        else:
            weights = weights_enum.IMAGENET1K_V1
            state = weights.get_state_dict(progress=True, check_hash=True)
            path = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
        model.load_state_dict(state, strict=True)
        provenance = {"backbone": name, "weights": "IMAGENET1K_V1", "sha256": sha256_file(path)}
    width = model.classifier[-1].in_features
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 1))
    return model, provenance


def configure_stage(model, stage):
    if stage not in {"warmup", "finetune"}:
        raise ValueError("Unknown training stage.")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    trainable_blocks = []
    if stage == "finetune":
        for index in range(len(model.features) - 3, len(model.features)):
            trainable_blocks.append(f"features.{index}")
            for parameter in model.features[index].parameters():
                parameter.requires_grad = True
    return {"stage": stage, "trainable_blocks": trainable_blocks + ["classifier"],
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def train_mode(model, stage):
    model.train()
    for index, block in enumerate(model.features):
        if stage == "warmup" or index < len(model.features) - 3:
            block.eval()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def move_images(images, device, config):
    images = images.to(device, non_blocking=True)
    if config["training"]["channels_last"]:
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def load_torch(path):
    # All runner checkpoints are local artifacts from this project.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch releases before weights_only was exposed.
        return torch.load(path, map_location="cpu")


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
