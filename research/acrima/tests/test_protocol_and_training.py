import copy

import numpy as np
import pytest
import torch
from torchvision.models import efficientnet_b0

from glaucoma.data import MergeSingletonBatchSampler, dataset_rows, make_folds
from glaucoma.models import configure_stage, train_mode
from glaucoma.pipeline import selected_epoch
from glaucoma.protocol import prepare


def test_fixed_folds_are_deterministic_and_group_disjoint(synthetic_run):
    config, _ = synthetic_run
    root, output, rows, folds, protocol = prepare(config)
    train = [r for r in rows if r["split"] == "train"]
    assert make_folds(train, 2, 42) == folds
    group_folds = {}
    for row in folds:
        group_folds.setdefault(row["group_id"], set()).add(row["fold"])
    assert all(len(v) == 1 for v in group_folds.values())
    assert protocol["roles"]["val"] == "calibration_only"
    assert len(folds) == 24
    prepare(config)
    with pytest.raises(ValueError, match="Only outer train"):
        make_folds(rows, 2, 42)


def test_changed_config_and_fold_file_cannot_reuse_run(synthetic_run):
    config, _ = synthetic_run
    _, output, _, _, _ = prepare(config)
    changed = copy.deepcopy(config)
    changed["training"]["batch_size"] = 4
    with pytest.raises(ValueError, match="changed"):
        prepare(changed)
    with (output / "folds.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="fold file changed"):
        prepare(config)


def test_image_hash_changes_are_detected(synthetic_run):
    config, rows = synthetic_run
    from pathlib import Path
    path = Path(config["data"]["root"]) / rows[0]["image_path"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        dataset_rows(path.parents[2], config["data"]["expected_counts"], verify_images=True)


@pytest.mark.parametrize("count", [8, 9, 16, 17, 23])
def test_training_sampler_never_drops_samples_or_emits_singleton(count):
    sampler = MergeSingletonBatchSampler(count, 8, 42, True)
    batches = list(sampler)
    assert all(len(batch) >= 2 for batch in batches)
    assert sorted(i for batch in batches for i in batch) == list(range(count))
    assert len(batches) == len(sampler)


def test_frozen_blocks_and_batchnorm_stats_stay_frozen():
    torch.set_num_threads(1)
    model = efficientnet_b0(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 1)
    detail = configure_stage(model, "finetune")
    train_mode(model, "finetune")
    assert detail["trainable_blocks"] == ["features.6", "features.7", "features.8", "classifier"]
    assert not model.features[0].training
    norms = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    previous = [m.running_mean.clone() for m in norms]
    model(torch.randn(2, 3, 32, 32)).sum().backward()
    assert all(torch.equal(before, module.running_mean) for before, module in zip(previous, norms))
    assert next(model.features[0].parameters()).grad is None
    assert model.classifier[-1].weight.grad is not None
    assert any(p.grad is not None for p in model.features[6].parameters())


def test_final_epoch_is_derived_without_touching_calibration():
    results = {"folds": [
        {"best_choice": {"stage": "warmup", "epoch_in_stage": 2}},
        {"best_choice": {"stage": "finetune", "epoch_in_stage": 4}},
        {"best_choice": {"stage": "finetune", "epoch_in_stage": 6}},
    ]}
    assert selected_epoch(results, 3) == {"stage": "finetune", "epoch_in_stage": 4}
