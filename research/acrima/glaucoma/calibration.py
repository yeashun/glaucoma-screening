from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit

from .metrics import validate_binary


def fit_temperature(y, logits):
    y, z = validate_binary(y, logits)
    if len(np.unique(y)) != 2:
        raise ValueError("Calibration needs both classes.")

    def objective(log_temperature):
        scaled = z / np.exp(log_temperature)
        return float(np.mean(np.logaddexp(0, scaled) - y * scaled))

    bound = float(np.log(100.0))
    fitted = minimize_scalar(objective, bounds=(-bound, bound), method="bounded", options={"xatol": 1e-8})
    if not fitted.success:
        raise RuntimeError("Temperature fitting failed.")
    # Include the identity and endpoints, so constrained optimization cannot worsen fitting NLL.
    candidates = [0.0, -bound, bound, float(fitted.x)]
    log_t = min(candidates, key=objective)
    return {"temperature": float(np.exp(log_t)), "temperature_bounds": [0.01, 100.0],
            "at_bound": bool(abs(abs(log_t) - bound) < 1e-5),
            "raw_fitting_nll": objective(0.0), "calibrated_fitting_nll": objective(log_t),
            "warning": "Fitting-set improvement is not independent validation of calibration."}


def choose_threshold(y, logits, target_sensitivity=0.9):
    y, z = validate_binary(y, logits)
    if len(np.unique(y)) != 2 or not 0 < target_sensitivity <= 1:
        raise ValueError("Invalid threshold selection data or sensitivity target.")
    probabilities = expit(z)
    best = None
    for threshold in np.unique(np.r_[0.0, probabilities]):
        predicted = probabilities >= threshold
        sensitivity = float(predicted[y == 1].mean())
        specificity = float((~predicted[y == 0]).mean())
        if sensitivity + 1e-12 >= target_sensitivity:
            key = (specificity, float(threshold))
            if best is None or key > best[0]:
                best = (key, {"threshold": float(threshold), "empirical_sensitivity": sensitivity,
                              "empirical_specificity": specificity, "target_sensitivity": target_sensitivity})
    return best[1]
