"""Temporal consistency analysis for facial emotion time series."""

from typing import Dict, List
import numpy as np


def analyze_temporal_consistency(
    stress_series: List[float],
    fear_series: List[float],
    window_size: int = 5,
    drift_threshold: float = 0.2,
    variance_threshold: float = 0.12,
) -> Dict[str, float]:
    """Compute temporal consistency metrics for stress and fear series."""
    if not stress_series or not fear_series:
        return {
            "stress_variance": 0.0,
            "fear_variance": 0.0,
            "stress_drift": 0.0,
            "fear_drift": 0.0,
            "inconsistency_score": 0.0,
            "status": "No Data",
        }

    stress_arr = np.asarray(stress_series, dtype="float32")
    fear_arr = np.asarray(fear_series, dtype="float32")

    stress_var = float(np.var(stress_arr))
    fear_var = float(np.var(fear_arr))

    # Drift: moving average difference between first and last window
    def window_mean(a):
        n = len(a)
        if n < window_size:
            return float(np.mean(a))
        return float(np.mean(a[-window_size:]) - np.mean(a[:window_size]))

    stress_drift = abs(window_mean(stress_arr))
    fear_drift = abs(window_mean(fear_arr))

    inconsistency = (stress_var + fear_var) / 2.0 + (stress_drift + fear_drift) / 2.0
    inconsistency = float(np.clip(inconsistency, 0.0, 1.0))

    if inconsistency < 0.09 and stress_drift < drift_threshold and fear_drift < drift_threshold:
        status = "Consistent"
    elif inconsistency < 0.18:
        status = "Moderate"
    else:
        status = "Inconsistent"

    return {
        "stress_variance": stress_var,
        "fear_variance": fear_var,
        "stress_drift": stress_drift,
        "fear_drift": fear_drift,
        "inconsistency_score": inconsistency,
        "status": status,
    }
