import numpy as np
import pytest

from glaucoma.calibration import choose_threshold, fit_temperature
from glaucoma.metrics import binary_metrics, cluster_bootstrap, selection_key


def test_auprc_is_not_silently_replaced_by_average_precision():
    result = binary_metrics([0, 1, 0, 1], [0.8, 0.7, 0.6, 0.5])
    assert result["auprc"] == pytest.approx(1 / 3)
    assert result["average_precision"] == pytest.approx(0.5)
    assert result["auroc"] == pytest.approx(0.25)


def test_perfect_ranking_and_extreme_logits_remain_finite():
    result = binary_metrics([0, 0, 1, 1], [-1000, -500, 500, 1000])
    assert result["auroc"] == result["auprc"] == 1
    assert result["nll"] >= 0
    assert np.isfinite(result["nll"])
    assert sum(b["count"] for b in result["reliability_bins"]) == 4


@pytest.mark.parametrize("labels,scores", [([0, 0], [0, 1]), ([0, 1], [np.nan, 1]), ([0, 1], [1])])
def test_invalid_or_single_class_metrics_fail(labels, scores):
    with pytest.raises(ValueError):
        binary_metrics(labels, scores)


def test_selection_prioritizes_auroc_then_auprc():
    a = {"auroc": 0.8, "auprc": 0.7, "nll": 0.1}
    b = {"auroc": 0.9, "auprc": 0.6, "nll": 0.9}
    c = {"auroc": 0.9, "auprc": 0.8, "nll": 1.0}
    assert selection_key(c) > selection_key(b) > selection_key(a)


def test_temperature_changes_confidence_but_not_ranking_or_default_decisions():
    y = np.array([0, 1, 0, 1, 0, 1])
    z = np.array([-4, 3, 2, -1, -2, 1], dtype=float)
    result = fit_temperature(y, z)
    t = result["temperature"]
    raw, calibrated = binary_metrics(y, z), binary_metrics(y, z / t)
    assert t > 0
    assert result["calibrated_fitting_nll"] <= result["raw_fitting_nll"] + 1e-10
    assert raw["auroc"] == calibrated["auroc"]
    assert raw["auprc"] == calibrated["auprc"]
    assert raw["confusion_matrix"] == calibrated["confusion_matrix"]


def test_threshold_handles_ties_and_meets_empirical_target():
    y = [0, 0, 1, 1, 1]
    z = [-2, 1, 1, 1, 2]
    result = choose_threshold(y, z, target_sensitivity=1)
    assert result["empirical_sensitivity"] == 1
    assert result["empirical_specificity"] == 0.5


def test_group_bootstrap_is_reproducible():
    arguments = ([0, 1, 0, 1, 0, 1], [-2, 2, 1, 3, -1, 0.5], ["a", "a", "b", "b", "c", "c"], 0.5)
    first = cluster_bootstrap(*arguments, samples=30, seed=17)
    assert first == cluster_bootstrap(*arguments, samples=30, seed=17)
    assert first["valid_replicates"] == 30
