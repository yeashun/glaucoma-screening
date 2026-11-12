from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_object(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_config(path: Path) -> dict:
    config = read_json(path)
    if config["metrics"]["primary"] != "auroc" or config["metrics"]["secondary"] != "auprc":
        raise ValueError("This protocol requires primary=auroc, secondary=auprc.")
    if config["metrics"]["auprc_definition"] != "trapezoidal" or config["metrics"]["ece_bins"] != 5:
        raise ValueError("This protocol fixes trapezoidal AUPRC and five reliability bins.")
    if config["cv"]["folds"] < 2 or config["training"]["batch_size"] < 2:
        raise ValueError("At least two folds and batch_size >= 2 are required.")
    if config["training"]["warmup_epochs"] < 1 or config["training"]["finetune_epochs"] < 1:
        raise ValueError("Both training stages must have at least one epoch.")
    if not 0 < config["calibration"]["target_sensitivity"] <= 1:
        raise ValueError("target_sensitivity must be in (0, 1].")
    options = config["training"]
    if options["amp"] not in {"off", "fp16", "bf16"} or options["workers"] < 0:
        raise ValueError("Invalid AMP mode or worker count.")
    if options["feature_batch_size"] < 1 or options["cpu_threads"] < 1 or options["patience"] < 1:
        raise ValueError("Invalid loader, CPU thread or patience setting.")
    if not config["logistic"]["c_values"] or any(c <= 0 for c in config["logistic"]["c_values"]):
        raise ValueError("Logistic regularization C must be positive.")
    if config["evaluation"]["bootstrap_samples"] < 1:
        raise ValueError("At least one bootstrap sample is required.")
    if config["evaluation"]["benchmark_repeats"] < 1 or config["evaluation"]["benchmark_warmup"] < 0:
        raise ValueError("Invalid inference benchmark repetitions.")
    names = [entry["name"] for entry in config["experiments"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("Experiment names must be unique and nonempty.")
    for entry in config["experiments"]:
        if entry["method"] not in {"logistic", "finetune"}:
            raise ValueError("Unsupported experiment method.")
        if entry["backbone"] not in {"efficientnet_b0", "efficientnet_b2"}:
            raise ValueError("First-stage runner supports EfficientNet-B0/B2.")
        if entry["weights"] != "IMAGENET1K_V1":
            raise ValueError("Production experiments require ImageNet1K V1 pretrained weights.")
        if entry["method"] == "logistic" and entry["augmentation"]:
            raise ValueError("Cached frozen features must use deterministic preprocessing.")
    return config
