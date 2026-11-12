from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .calibration import choose_threshold, fit_temperature
from .common import read_csv, read_json, sha256_file, write_csv, write_json
from .data import FundusDataset
from .engine import extract_features, fit_network, predict_logits
from .metrics import binary_metrics, cluster_bootstrap, plot_evaluation, selection_key
from .models import autocast_context, build_model, get_device, load_torch, move_images
from .protocol import get_partitions, prepare, resolve_paths, software_versions


SUMMARY_KEYS = ("auroc", "auprc", "average_precision", "nll", "brier")


def aggregate(results):
    return {"mean": {k: float(np.mean([r[k] for r in results])) for k in SUMMARY_KEYS},
            "std": {k: float(np.std([r[k] for r in results], ddof=1)) for k in SUMMARY_KEYS}}


def logistic_model(c, seed):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, solver="lbfgs", max_iter=3000, random_state=seed))


def save_joblib(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(value, temporary)
    temporary.replace(path)


def cv_experiments(config):
    root, output, rows, folds, protocol = prepare(config)
    if (output / "final" / "selection.json").exists():
        raise ValueError("Final selection already locked; do not rerun model search in this run.")
    training = [r for r in rows if r["split"] == "train"]
    index = {r["sample_id"]: i for i, r in enumerate(training)}
    y = np.array([r["label"] for r in training])
    cv_root = output / "cv"
    cv_root.mkdir(exist_ok=True)
    baseline_scores = []
    for fold in range(config["cv"]["folds"]):
        tr, va = get_partitions(rows, folds, fold)
        constant = np.full(len(va), logit(np.mean([r["label"] for r in tr])))
        baseline_scores.append(binary_metrics([r["label"] for r in va], constant))
    write_json(cv_root / "E0_constant.json", {"folds": baseline_scores, **aggregate(baseline_scores)})
    summaries = []
    for experiment in config["experiments"]:
        name = experiment["name"]
        directory = cv_root / name
        directory.mkdir(exist_ok=True)
        if experiment["method"] == "logistic":
            features, provenance = extract_features(root, training, experiment, config, output)
            c_results = []
            for c in config["logistic"]["c_values"]:
                scores, predictions = [], []
                for fold in range(config["cv"]["folds"]):
                    tr, va = get_partitions(rows, folds, fold)
                    ti = [index[r["sample_id"]] for r in tr]
                    vi = [index[r["sample_id"]] for r in va]
                    classifier = logistic_model(c, config["seed"] + fold)
                    with threadpool_limits(limits=config["training"]["cpu_threads"]):
                        classifier.fit(features[ti], y[ti])
                        z = classifier.decision_function(features[vi])
                    scores.append(binary_metrics(y[vi], z))
                    predictions += [{"sample_id": r["sample_id"], "label": r["label"], "group_id": r["group_id"],
                                     "fold": fold, "logit": float(v)} for r, v in zip(va, z)]
                c_results.append({"c": c, "folds": scores, **aggregate(scores), "predictions": predictions})
            chosen = max(c_results, key=lambda r: selection_key(r["mean"]))
            write_csv(directory / "oof_predictions.csv", chosen["predictions"])
            result = {"name": name, "method": "logistic", "best_c": chosen["c"], "pretrained": provenance,
                      "folds": chosen["folds"], "mean": chosen["mean"], "std": chosen["std"],
                      "c_search": [{k: v for k, v in item.items() if k != "predictions"} for item in c_results]}
            write_json(directory / "result.json", result)
        else:
            fold_results, predictions = [], []
            for fold in range(config["cv"]["folds"]):
                work = directory / f"fold_{fold}"
                completed = work / "result.json"
                if completed.exists():
                    result = read_json(completed)
                    if sha256_file(work / "best.pt") != result["checkpoint_sha256"] or sha256_file(work / "best_predictions.csv") != result["predictions_sha256"]:
                        raise ValueError("Completed fold checkpoint or predictions changed.")
                    print(f"Reusing completed {name} fold {fold}.", flush=True)
                else:
                    tr, va = get_partitions(rows, folds, fold)
                    result = fit_network(root, tr, va, experiment, config, output, work, config["seed"] + fold)
                    result.update({"fold": fold, "checkpoint_sha256": sha256_file(work / "best.pt"),
                                   "predictions_sha256": sha256_file(work / "best_predictions.csv")})
                    write_json(completed, result)
                fold_results.append(result)
                predictions += [{**r, "fold": fold} for r in read_csv(work / "best_predictions.csv")]
            write_csv(directory / "oof_predictions.csv", predictions)
            scores = [r["metrics"] for r in fold_results]
            result = {"name": name, "method": "finetune", "folds": fold_results, **aggregate(scores)}
            write_json(directory / "result.json", result)
        summaries.append(result)
        print(f"{name}: mean AUROC={result['mean']['auroc']:.5f}, AUPRC={result['mean']['auprc']:.5f}", flush=True)
    winner = max(summaries, key=lambda r: selection_key(r["mean"]))
    summary = {"fingerprint": protocol["fingerprint"], "primary": "auroc", "secondary": "auprc",
               "selection_order": ["mean_auroc descending", "mean_auprc descending", "mean_nll ascending"],
               "selected_experiment": winner["name"], "experiments": summaries,
               "warning": "Cross-validation scores used for selection are development results, not unbiased final test performance."}
    write_json(output / "cv_summary.json", summary)
    write_csv(output / "leaderboard.csv", [
        {"experiment": r["name"], **{f"{k}_mean": r["mean"][k] for k in SUMMARY_KEYS},
         **{f"{k}_std": r["std"][k] for k in SUMMARY_KEYS}}
        for r in sorted(summaries, key=lambda r: selection_key(r["mean"]), reverse=True)])
    return summary


def selected_epoch(result, warmup_epochs):
    totals = [r["best_choice"]["epoch_in_stage"] + (warmup_epochs if r["best_choice"]["stage"] == "finetune" else 0)
              for r in result["folds"]]
    total = max(1, int(np.median(totals)))
    return {"stage": "warmup", "epoch_in_stage": total} if total <= warmup_epochs else {
        "stage": "finetune", "epoch_in_stage": total - warmup_epochs}


def fit_final(config):
    root, output, rows, _, protocol = prepare(config)
    summary = read_json(output / "cv_summary.json")
    if summary["fingerprint"] != protocol["fingerprint"]:
        raise ValueError("CV summary protocol mismatch.")
    final = output / "final"
    final.mkdir(exist_ok=True)
    if (final / "fit.json").exists():
        locked = read_json(final / "fit.json")
        verify_artifact(final, locked)
        print("Final model already fitted; reusing locked artifact.", flush=True)
        return locked
    if (output / "evaluation_started.json").exists() or (final / "calibration.json").exists():
        raise ValueError("Cannot refit after calibration or final evaluation.")
    name = summary["selected_experiment"]
    experiment = next(e for e in config["experiments"] if e["name"] == name)
    cv_result = next(r for r in summary["experiments"] if r["name"] == name)
    selection = {"experiment": experiment, "cv_summary_sha256": sha256_file(output / "cv_summary.json"),
                 "fingerprint": protocol["fingerprint"], "software": software_versions()}
    if experiment["method"] == "logistic":
        selection["c"] = cv_result["best_c"]
    else:
        selection["choice"] = selected_epoch(cv_result, config["training"]["warmup_epochs"])
    if (final / "selection.json").exists() and read_json(final / "selection.json") != selection:
        raise ValueError("Locked selection changed.")
    write_json(final / "selection.json", selection)
    training = [r for r in rows if r["split"] == "train"]
    started = time.perf_counter()
    if experiment["method"] == "logistic":
        features, provenance = extract_features(root, training, experiment, config, output)
        classifier = logistic_model(selection["c"], config["seed"])
        with threadpool_limits(limits=config["training"]["cpu_threads"]):
            classifier.fit(features, [r["label"] for r in training])
        save_joblib(final / "classifier.joblib", classifier)
        result = {"artifact": "classifier.joblib", "pretrained": provenance,
                  "training_seconds": time.perf_counter() - started}
    else:
        result = fit_network(root, training, [], experiment, config, output, final, config["seed"], selection["choice"])
        result["artifact"] = "model.pt"
    result.update({"experiment": experiment, "fingerprint": protocol["fingerprint"],
                   "artifact_sha256": sha256_file(final / result["artifact"]),
                   "selection_sha256": sha256_file(final / "selection.json"), "training_samples": len(training)})
    write_json(final / "fit.json", result)
    return result


def verify_artifact(final, fitted):
    if sha256_file(final / fitted["artifact"]) != fitted["artifact_sha256"]:
        raise ValueError("Final model artifact changed.")
    if sha256_file(final / "selection.json") != fitted["selection_sha256"]:
        raise ValueError("Final selection changed.")


def final_logits(root, rows, config, output, fitted):
    final = output / "final"
    verify_artifact(final, fitted)
    experiment = fitted["experiment"]
    if experiment["method"] == "logistic":
        features, provenance = extract_features(root, rows, experiment, config, output)
        if provenance != fitted["pretrained"]:
            raise ValueError("Feature backbone weights differ from final fitting.")
        classifier = joblib.load(final / fitted["artifact"])
        return classifier.decision_function(features)
    device = get_device(config)
    model, _ = build_model(experiment, output, pretrained=False, dropout=config["training"]["dropout"])
    model.load_state_dict(load_torch(final / fitted["artifact"])["state_dict"], strict=True)
    model.to(device)
    if config["training"]["channels_last"]:
        model.to(memory_format=torch.channels_last)
    result = predict_logits(model, root, rows, experiment, config, device)
    del model
    return result


def calibrate_final(config):
    root, output, rows, _, protocol = prepare(config)
    final = output / "final"
    fitted = read_json(final / "fit.json")
    verify_artifact(final, fitted)
    path = final / "calibration.json"
    if path.exists():
        result = read_json(path)
        verify_calibration(final, fitted, result)
        return result
    if (output / "evaluation_started.json").exists():
        raise ValueError("Cannot recalibrate after opening the test set.")
    calibration_rows = [r for r in rows if r["split"] == "val"]
    y = [r["label"] for r in calibration_rows]
    z = final_logits(root, calibration_rows, config, output, fitted)
    result = fit_temperature(y, z)
    scaled = z / result["temperature"]
    result.update({"artifact_sha256": fitted["artifact_sha256"], "fingerprint": protocol["fingerprint"],
                   "fit_sha256": sha256_file(final / "fit.json"), "calibration_split": "val.csv",
                   "calibration_samples": len(y), "operating_point": choose_threshold(y, scaled, config["calibration"]["target_sensitivity"]),
                   "raw_descriptive_metrics": binary_metrics(y, z),
                   "calibrated_fitting_metrics": binary_metrics(y, scaled),
                   "warning": "Temperature and operating threshold were fitted here. These are not independent evaluation metrics."})
    write_csv(final / "calibration_predictions.csv", prediction_rows(calibration_rows, z, result["temperature"]))
    write_json(path, result)
    return result


def verify_calibration(final, fitted, calibration):
    verify_artifact(final, fitted)
    if calibration["artifact_sha256"] != fitted["artifact_sha256"] or calibration["fit_sha256"] != sha256_file(final / "fit.json"):
        raise ValueError("Calibration belongs to a different final model.")


def prediction_rows(rows, z, temperature):
    return [{"sample_id": r["sample_id"], "label": r["label"], "group_id": r["group_id"],
             "raw_logit": float(value), "p_raw": float(expit(value)), "p_positive": float(expit(value / temperature))}
            for r, value in zip(rows, z)]


@torch.inference_mode()
def benchmark_final(root, row, config, output, fitted, temperature):
    """Single-image warm latency, including classifier/calibration, excluding I/O."""
    experiment = fitted["experiment"]
    device = get_device(config)
    model, provenance = build_model(experiment, output, pretrained=experiment["method"] == "logistic",
                                    dropout=config["training"]["dropout"])
    classifier = None
    if experiment["method"] == "logistic":
        if provenance != fitted["pretrained"]:
            raise ValueError("Benchmark feature backbone weights differ.")
        model.classifier = torch.nn.Identity()
        classifier = joblib.load(output / "final" / fitted["artifact"])
    else:
        model.load_state_dict(load_torch(output / "final" / fitted["artifact"])["state_dict"])
    model.eval().to(device)
    if config["training"]["channels_last"]:
        model.to(memory_format=torch.channels_last)
    image, _, _ = FundusDataset(root, [row], experiment["image_size"])[0]
    image = move_images(image.unsqueeze(0), device, config)

    def forward():
        with autocast_context(config, device):
            value = model(image)
        if classifier is not None:
            value = classifier.decision_function(value.float().cpu().numpy())
        else:
            value = value.float().cpu().numpy().reshape(-1)
        return expit(value / temperature)

    with threadpool_limits(limits=config["training"]["cpu_threads"]):
        for _ in range(config["evaluation"]["benchmark_warmup"]):
            forward()
        latencies = []
        for _ in range(config["evaluation"]["benchmark_repeats"]):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            forward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - start) * 1000)
    result = {"batch_size": 1, "repetitions": len(latencies), "mean_ms": float(np.mean(latencies)),
              "median_ms": float(np.median(latencies)), "device": str(device),
              "scope": "Warm model forward + CPU classifier if used + temperature sigmoid; excludes disk I/O, resize and model loading",
              "total_backbone_and_head_parameters": sum(p.numel() for p in model.parameters())}
    if classifier is not None:
        result["logistic_parameters"] = int(classifier[-1].coef_.size + classifier[-1].intercept_.size)
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update({"gpu": properties.name, "total_memory_gib": properties.total_memory / 1024 ** 3})
    del model
    return result


def evaluate_test(config):
    root, output, rows, _, protocol = prepare(config)
    final = output / "final"
    fitted, calibration = read_json(final / "fit.json"), read_json(final / "calibration.json")
    verify_calibration(final, fitted, calibration)
    identities = {"fingerprint": protocol["fingerprint"], "artifact_sha256": fitted["artifact_sha256"],
                  "calibration_sha256": sha256_file(final / "calibration.json")}
    marker = output / "evaluation_started.json"
    if marker.exists():
        if read_json(marker) != identities:
            raise ValueError("Final evaluation configuration changed.")
        if (output / "test_report.json").exists():
            return read_json(output / "test_report.json")
        if not (output / "test_predictions.csv").exists():
            raise RuntimeError("Evaluation was interrupted before prediction export. Inspect this run; automatic reopening is disabled.")
        # Recovery only reuses exported logits; no new model or test-image inference.
        exported = read_csv(output / "test_predictions.csv")
        testing = [r for r in rows if r["split"] == "test"]
        if [r["sample_id"] for r in exported] != [r["sample_id"] for r in testing]:
            raise ValueError("Interrupted evaluation prediction IDs differ.")
        z = np.array([float(r["raw_logit"]) for r in exported])
    else:
        # Exclusive marker prevents accidental concurrent or repeated test searches.
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(identities, handle, indent=2)
        testing = [r for r in rows if r["split"] == "test"]
        z = final_logits(root, testing, config, output, fitted)
        write_csv(output / "test_predictions.csv", prediction_rows(testing, z, calibration["temperature"]))
    y = np.array([r["label"] for r in testing])
    scaled = z / calibration["temperature"]
    threshold = calibration["operating_point"]["threshold"]
    result = {**identities, "primary_metric": "auroc", "secondary_metric": "auprc",
              "auprc_definition": "trapezoidal PR area; average_precision is reported separately",
              "selected_experiment": fitted["experiment"]["name"], "temperature": calibration["temperature"],
              "raw_at_0.5": binary_metrics(y, z), "calibrated_at_0.5": binary_metrics(y, scaled),
              "calibrated_at_operating_threshold": binary_metrics(y, scaled, threshold),
              "bootstrap": cluster_bootstrap(y, scaled, [r["group_id"] for r in testing], threshold,
                                             config["evaluation"]["bootstrap_samples"], config["seed"]),
              "inference_benchmark": benchmark_final(root, testing[0], config, output, fitted, calibration["temperature"]),
              "software": software_versions(), "patient_level_independence_verified": False,
              "warning": "Small held-out research set; intervals use heuristic similarity groups, not verified patients."}
    plot_evaluation(y, z, scaled, output / "test_curves.png")
    point = result["calibrated_at_0.5"]
    intervals = result["bootstrap"]["intervals"]
    report = ("# Final held-out test evaluation\n\n"
              f"Selected experiment: **{fitted['experiment']['name']}**. Samples: {len(testing)}.\n\n"
              f"Primary AUROC: **{point['auroc']:.4f}**, 95% group-bootstrap CI {intervals['auroc']}.\n\n"
              f"Secondary AUPRC (trapezoidal): **{point['auprc']:.4f}**, 95% CI {intervals['auprc']}.\n\n"
              f"AP (different integration rule): {point['average_precision']:.4f}.\n\n"
              f"Calibrated NLL: {point['nll']:.4f}; Brier: {point['brier']:.4f}.\n\n"
              "Calibration and threshold were fixed using val.csv, not this test set.\n\n"
              "Similarity groups are heuristic; patient-level independence is unverified.\n")
    (output / "test_report.md").write_text(report, encoding="utf-8")
    write_json(output / "test_report.json", result)
    return result


def run_all(config):
    _, output, _, _, _ = prepare(config)
    if not (output / "final" / "selection.json").exists():
        cv_experiments(config)
    fit_final(config)
    calibrate_final(config)
    return evaluate_test(config)


def predict_images(config, image_paths, destination):
    """Inference entry point accepts unlabeled images and exports positive probabilities."""
    from PIL import Image
    root, output = resolve_paths(config)
    protocol = read_json(output / "protocol.json")
    if protocol["config"] != config:
        raise ValueError("Inference configuration differs from the locked training protocol.")
    final = output / "final"
    fitted, calibration = read_json(final / "fit.json"), read_json(final / "calibration.json")
    verify_calibration(final, fitted, calibration)
    rows = []
    for index, path in enumerate(image_paths):
        path = Path(path).resolve()
        with Image.open(path) as image:
            image.convert("RGB").load()
        rows.append({"sample_id": f"input_{index:06d}", "image_path": str(path), "label": 0,
                     "group_id": f"input_{index}", "sha256": sha256_file(path)})
    if not rows:
        raise ValueError("No input images.")
    z = final_logits(root, rows, config, output, fitted)
    predictions = [{"sample_id": r["sample_id"], "image_path": r["image_path"],
                    "p_positive": float(expit(v / calibration["temperature"]))} for r, v in zip(rows, z)]
    write_csv(Path(destination), predictions)
    return predictions
