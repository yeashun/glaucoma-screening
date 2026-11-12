from __future__ import annotations

import argparse
import json
import os
import socket
from contextlib import contextmanager
from pathlib import Path

# Set before importing torch/CUDA or matplotlib.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))

from .common import digest_object, load_config, write_json
from .protocol import PROJECT, prepare, resolve_paths, software_versions


@contextmanager
def command_lock(output):
    lock_dir = output.parent / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_dir / (digest_object(str(output))[:24] + ".lock")
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError(f"Another command holds {lock}. If a process was killed, verify it stopped before removing this lock.") from error
    try:
        with handle:
            json.dump({"pid": os.getpid(), "hostname": socket.gethostname(), "output": str(output)}, handle)
        yield
    finally:
        lock.unlink(missing_ok=True)


def doctor(config):
    import torch
    result = {"software": software_versions(), "cuda_available": torch.cuda.is_available(),
              "torch_cuda_runtime": torch.version.cuda, "requested_device": config["training"]["device"],
              "configured_batch_size": config["training"]["batch_size"], "configured_amp": config["training"]["amp"]}
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        result.update({"gpu": properties.name, "total_memory_gib": properties.total_memory / 1024 ** 3,
                       "bf16_supported": torch.cuda.is_bf16_supported()})
    print(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="ACRIMA experiments: primary AUROC, secondary trapezoidal AUPRC.")
    parser.add_argument("command", choices=["doctor", "prepare", "cv", "fit", "calibrate", "evaluate", "all", "predict"])
    parser.add_argument("--config", default=str(PROJECT / "configs" / "acrima_4090.json"))
    parser.add_argument("--images", nargs="+", help="Unlabeled image files or directories, for predict.")
    parser.add_argument("--predictions", default="predictions.csv", help="CSV destination for predict.")
    args = parser.parse_args(argv)
    config = load_config(Path(args.config))
    if args.command == "doctor":
        doctor(config)
        return
    from .pipeline import (calibrate_final, cv_experiments, evaluate_test, fit_final, predict_images, run_all)
    _, output = resolve_paths(config)
    with command_lock(output):
        if args.command == "prepare":
            _, output, _, _, protocol = prepare(config)
            print(json.dumps({"output": str(output), "fold_counts": protocol["fold_counts"], "roles": protocol["roles"]}, indent=2))
        elif args.command == "predict":
            if not args.images:
                parser.error("predict requires --images")
            paths = []
            for value in args.images:
                path = Path(value)
                if path.is_dir():
                    paths.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}))
                else:
                    paths.append(path)
            result = predict_images(config, paths, args.predictions)
            print(f"Exported {len(result)} positive probabilities to {args.predictions}.")
        else:
            functions = {"cv": cv_experiments, "fit": fit_final, "calibrate": calibrate_final,
                         "evaluate": evaluate_test, "all": run_all}
            result = functions[args.command](config)
            if args.command in {"evaluate", "all"}:
                metrics = result["calibrated_at_0.5"]
                print(f"Final AUROC={metrics['auroc']:.6f}; AUPRC={metrics['auprc']:.6f}; AP={metrics['average_precision']:.6f}")


if __name__ == "__main__":
    main()
