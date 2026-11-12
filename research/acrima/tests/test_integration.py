import pytest

from glaucoma.common import read_csv, read_json
from glaucoma import pipeline


def test_complete_cpu_pipeline_preserves_holdouts_and_exports_probabilities(synthetic_run, monkeypatch, tmp_path):
    config, rows = synthetic_run
    observed = []
    original_fit = pipeline.fit_network
    original_features = pipeline.extract_features
    original_final_logits = pipeline.final_logits

    def checked_fit(root, training, validation, *args, **kwargs):
        assert {r["split"] for r in training} == {"train"}
        assert all(r["split"] == "train" for r in validation)
        observed.append(("fit", len(training), len(validation)))
        return original_fit(root, training, validation, *args, **kwargs)

    def checked_features(root, subset, *args, **kwargs):
        # CV calls only train; calibration/test feature extraction, if selected,
        # passes through final_logits after locking the model.
        assert subset
        return original_features(root, subset, *args, **kwargs)

    def checked_final_logits(root, subset, config, output, fitted):
        assert (output / "final" / "selection.json").exists()
        if subset[0].get("split") == "test":
            assert (output / "final" / "calibration.json").exists()
            assert (output / "evaluation_started.json").exists()
        observed.append(("final_inference", subset[0].get("split")))
        return original_final_logits(root, subset, config, output, fitted)

    monkeypatch.setattr(pipeline, "fit_network", checked_fit)
    monkeypatch.setattr(pipeline, "extract_features", checked_features)
    monkeypatch.setattr(pipeline, "final_logits", checked_final_logits)
    report = pipeline.run_all(config)
    from pathlib import Path
    output = Path(config["output_dir"])
    assert report["primary_metric"] == "auroc"
    assert report["secondary_metric"] == "auprc"
    assert len([item for item in observed if item[0] == "fit" and item[2] > 0]) == 4
    assert observed.count(("final_inference", "val")) == 1
    assert observed.count(("final_inference", "test")) == 1
    predictions = read_csv(output / "test_predictions.csv")
    assert len(predictions) == 8
    assert all(0 <= float(r["p_positive"]) <= 1 for r in predictions)
    assert len(read_csv(output / "leaderboard.csv")) == 3
    assert (output / "test_curves.png").is_file()
    assert report["inference_benchmark"]["mean_ms"] > 0
    # Repeating all must reuse the locked model, calibration and report, never
    # redo a test search or retrain against the holdout.
    before = len(observed)
    assert pipeline.run_all(config) == report
    assert len(observed) == before
    with pytest.raises(ValueError, match="selection already locked"):
        pipeline.cv_experiments(config)
    image = Path(config["data"]["root"]) / rows[0]["image_path"]
    exported = pipeline.predict_images(config, [image], tmp_path / "unlabeled.csv")
    assert 0 <= exported[0]["p_positive"] <= 1
    assert "label" not in exported[0]


def test_final_neural_path_and_calibration_artifact_lock(synthetic_run):
    config, _ = synthetic_run
    config["experiments"] = [config["experiments"][1]]
    result = pipeline.run_all(config)
    assert result["selected_experiment"] == "E2"
    from pathlib import Path
    final = Path(config["output_dir"]) / "final"
    assert (final / "model.pt").exists()
    artifact = final / "model.pt"
    artifact.write_bytes(artifact.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        pipeline.evaluate_test(config)


def test_logistic_final_and_prediction_without_original_dataset(synthetic_run, tmp_path):
    config, rows = synthetic_run
    config["experiments"] = [config["experiments"][0]]
    report = pipeline.run_all(config)
    assert report["selected_experiment"] == "E1"
    from pathlib import Path
    root = Path(config["data"]["root"])
    image = tmp_path / "unlabeled.jpg"
    image.write_bytes((root / rows[0]["image_path"]).read_bytes())
    # A deployed predictor needs model artifacts and input images, not the
    # development dataset or its ground truth.
    root.rename(root.with_name("dataset_not_available_to_inference"))
    result = pipeline.predict_images(config, [image], tmp_path / "inference.csv")
    assert 0 <= result[0]["p_positive"] <= 1
