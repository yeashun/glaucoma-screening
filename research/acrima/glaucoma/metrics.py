from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    auc, average_precision_score, confusion_matrix, precision_recall_curve,
    roc_auc_score, roc_curve,
)


def validate_binary(y, logits):
    y = np.asarray(y, dtype=np.int64)
    z = np.asarray(logits, dtype=np.float64)
    if y.ndim != 1 or z.shape != y.shape or not len(y):
        raise ValueError("Expected equally sized, nonempty one-dimensional labels and logits.")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(z).all():
        raise ValueError("Invalid binary labels or nonfinite logits.")
    return y, z


def binary_metrics(y, logits, threshold=0.5, bins=5):
    y, z = validate_binary(y, logits)
    if len(np.unique(y)) != 2:
        raise ValueError("AUROC/AUPRC evaluation requires both classes.")
    if not 0 <= threshold <= 1 or bins < 1:
        raise ValueError("Invalid threshold or bin count.")
    p = expit(z)
    precision, recall, _ = precision_recall_curve(y, z)
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    bin_ids = np.minimum((p * bins).astype(int), bins - 1)
    reliability = []
    ece = 0.0
    for index in range(bins):
        mask = bin_ids == index
        count = int(mask.sum())
        mean_p = float(p[mask].mean()) if count else None
        observed = float(y[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(mean_p - observed)
        reliability.append({"lower": index / bins, "upper": (index + 1) / bins,
                            "count": count, "mean_probability": mean_p, "positive_fraction": observed})
    return {
        "auroc": float(roc_auc_score(y, z)),
        # Deliberately distinct from AP: linear/trapezoidal area under the PR curve.
        "auprc": float(auc(recall, precision)),
        "average_precision": float(average_precision_score(y, z)),
        "nll": float(np.mean(np.logaddexp(0, z) - y * z)),
        "brier": float(np.mean((p - y) ** 2)), "ece": float(ece),
        "accuracy": float((tp + tn) / len(y)), "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        "threshold": float(threshold), "samples": len(y), "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()), "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "reliability_bins": reliability,
    }


def selection_key(metrics):
    """AUROC first; AUPRC breaks ties; then smaller NLL."""
    return metrics["auroc"], metrics["auprc"], -metrics["nll"]


def cluster_bootstrap(y, logits, groups, threshold, samples=2000, seed=42):
    y, z = validate_binary(y, logits)
    groups = np.asarray(groups)
    if groups.shape != y.shape or samples < 1:
        raise ValueError("Invalid bootstrap groups or repetitions.")
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    keys = ["auroc", "auprc", "average_precision", "nll", "brier", "sensitivity", "specificity", "accuracy"]
    values = {key: [] for key in keys}
    for _ in range(samples):
        indices = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        if len(np.unique(y[indices])) < 2:
            continue
        result = binary_metrics(y[indices], z[indices], threshold)
        for key in keys:
            values[key].append(result[key])
    if len(values["auroc"]) < max(1, samples // 2):
        raise ValueError("Too few two-class bootstrap replicates.")
    return {
        "method": "percentile bootstrap resampling similarity groups; not patient groups",
        "requested_replicates": samples, "valid_replicates": len(values["auroc"]), "seed": seed,
        "confidence_level": 0.95,
        "intervals": {key: [float(v) for v in np.percentile(value, [2.5, 97.5])] for key, value in values.items()},
    }


def plot_evaluation(y, raw_logits, calibrated_logits, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, z in [("Raw", raw_logits), ("Calibrated", calibrated_logits)]:
        result = binary_metrics(y, z)
        fpr, tpr, _ = roc_curve(y, z)
        precision, recall, _ = precision_recall_curve(y, z)
        axes[0].plot(fpr, tpr, label=f"{name}: AUROC={result['auroc']:.3f}")
        axes[1].plot(recall, precision, label=f"{name}: AUPRC={result['auprc']:.3f}")
        occupied = [item for item in result["reliability_bins"] if item["count"]]
        axes[2].plot([b["mean_probability"] for b in occupied], [b["positive_fraction"] for b in occupied], "o-", label=name)
        if name == "Calibrated":
            for b in occupied:
                axes[2].annotate(f"n={b['count']}", (b["mean_probability"], b["positive_fraction"]))
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.7)
    axes[1].axhline(np.mean(y), color="black", linestyle="--", linewidth=0.7, label="Positive fraction")
    axes[2].plot([0, 1], [0, 1], "k--", linewidth=0.7)
    for ax, title, xlabel, ylabel in zip(axes, ["ROC", "Precision-recall", "Reliability (5 bins)"],
                                        ["False positive rate", "Recall", "Predicted probability"],
                                        ["True positive rate", "Precision", "Observed positive fraction"]):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)
