from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path

from .common import digest_object, read_csv, read_json, sha256_file, write_csv, write_json
from .data import dataset_rows, make_folds


PROJECT = Path(__file__).resolve().parents[1]


def resolve_paths(config):
    return (PROJECT / config["data"]["root"]).resolve(), (PROJECT / config["output_dir"]).resolve()


def software_versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "Pillow", "matplotlib", "joblib"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def fingerprint(config, root):
    source = {path.name: sha256_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    manifests = {name: sha256_file(root / name) for name in ("manifest.csv", "train.csv", "val.csv", "test.csv")}
    local_weights = {entry["name"]: sha256_file((PROJECT / entry["weights_path"]).resolve())
                     for entry in config["experiments"] if entry.get("weights_path")}
    return digest_object({"config": config, "source": source, "manifests": manifests, "local_weights": local_weights})


def prepare(config, verify_images=True):
    root, output = resolve_paths(config)
    rows = dataset_rows(root, config["data"]["expected_counts"], verify_images=verify_images)
    current = fingerprint(config, root)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        if protocol["fingerprint"] != current:
            raise ValueError("Run config, source, manifests or weights changed; use a new output_dir.")
        folds = read_csv(output / "folds.csv")
        if sha256_file(output / "folds.csv") != protocol["folds_sha256"]:
            raise ValueError("Fixed fold file changed.")
    else:
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError("Output directory is nonempty but has no valid protocol; choose a fresh directory.")
        training = [row for row in rows if row["split"] == "train"]
        folds = make_folds(training, config["cv"]["folds"], config["seed"])
        write_csv(output / "folds.csv", folds)
        protocol = {
            "fingerprint": current, "config": config, "software_at_preparation": software_versions(),
            "roles": {"train": "development_only; fixed grouped cross-validation", "val": "calibration_only", "test": "final_evaluation_only"},
            "primary_metric": "auroc", "secondary_metric": "auprc",
            "auprc_definition": "sklearn.metrics.auc(recall, precision), trapezoidal PR area; AP reported separately",
            "folds_sha256": sha256_file(output / "folds.csv"),
            "fold_method": "StratifiedGroupKFold(shuffle=False); group identifiers hashed with seed before splitting",
            "grouping": "Unchanged conservative dHash similarity groups; candidates include false matches; not patient IDs",
            "patient_level_independence_verified": False,
            "outer_counts": {split: len([r for r in rows if r["split"] == split]) for split in ("train", "val", "test")},
            "fold_counts": [{"fold": i, "validation": sum(int(r["fold"]) == i for r in folds),
                             "positive": sum(int(r["label"]) for r in folds if int(r["fold"]) == i)}
                            for i in range(config["cv"]["folds"])],
        }
        write_json(protocol_path, protocol)
    return root, output, rows, folds, protocol


def get_partitions(rows, folds, fold):
    lookup = {row["sample_id"]: int(row["fold"]) for row in folds}
    train = [r for r in rows if r["split"] == "train" and lookup[r["sample_id"]] != fold]
    validation = [r for r in rows if r["split"] == "train" and lookup[r["sample_id"]] == fold]
    if set(r["group_id"] for r in train) & set(r["group_id"] for r in validation):
        raise ValueError("Group leakage.")
    return train, validation
