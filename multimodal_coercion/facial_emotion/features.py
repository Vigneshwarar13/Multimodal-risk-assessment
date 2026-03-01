from typing import Dict


def stress_from_emotions(probs: Dict[str, float]) -> float:
    """
    Derive a simple stress index from emotion probabilities.
    Combines fear, anger, and sadness as proxies for stress.
    """
    return float(
        probs.get("fear", 0.0) + probs.get("anger", 0.0) + probs.get("sadness", 0.0)
    )

