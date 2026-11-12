import hashlib
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.models import efficientnet_b0

from glaucoma.common import read_json, sha256_file, write_csv


@pytest.fixture
def synthetic_run(tmp_path):
    """Random weights and synthetic images, never a scientific performance run."""
    torch.set_num_threads(1)
    torch.manual_seed(7)
    root = tmp_path / "dataset"
    rows = []
    rng = np.random.default_rng(12)
    for split, count in [("train", 24), ("val", 8), ("test", 8)]:
        for i in range(count):
            label = i % 2
            name = f"{split}_{i:03d}"
            class_name = "1_positive" if label else "0_negative"
            relative = f"{split}/{class_name}/{name}.jpg"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = rng.integers(0, 180, (32, 32, 3), dtype=np.uint8)
            pixels[:, :, 0] += 30 * label
            Image.fromarray(pixels).save(path)
            rows.append({"sample_id": name, "image_path": relative, "label": label, "split": split,
                         "group_id": f"{split}_group_{i // 2}", "sha256": sha256_file(path),
                         "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()})
    write_csv(root / "manifest.csv", rows)
    for split in ("train", "val", "test"):
        write_csv(root / f"{split}.csv", [r for r in rows if r["split"] == split])
    weights = tmp_path / "SYNTHETIC_RANDOM_WEIGHTS_FOR_TESTS_ONLY.pth"
    torch.save(efficientnet_b0(weights=None).state_dict(), weights)
    config = read_json(Path(__file__).resolve().parents[1] / "configs/acrima_4090.json")
    config["data"] = {"root": str(root), "expected_counts": {"train": 24, "val": 8, "test": 8}}
    config["output_dir"] = str(tmp_path / "synthetic_output")
    config["training"].update({"device": "cpu", "amp": "off", "channels_last": False, "workers": 0,
                               "cpu_threads": 1, "batch_size": 8, "feature_batch_size": 8,
                               "warmup_epochs": 1, "finetune_epochs": 1, "patience": 1})
    config["cv"]["folds"] = 2
    config["logistic"]["c_values"] = [0.1]
    config["evaluation"]["bootstrap_samples"] = 20
    config["evaluation"].update({"benchmark_warmup": 1, "benchmark_repeats": 2})
    for experiment in config["experiments"]:
        experiment.update({"image_size": 32, "weights_path": str(weights)})
    return config, rows
