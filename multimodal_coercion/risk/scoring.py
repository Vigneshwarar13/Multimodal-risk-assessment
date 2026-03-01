from typing import Tuple


def label_from_prob(prob: float, good_max: float, poor_min: float) -> Tuple[str, float]:
    if prob < good_max:
        return "Good", float(prob)
    if prob >= poor_min:
        return "Poor", float(prob)
    return "Average", float(prob)

