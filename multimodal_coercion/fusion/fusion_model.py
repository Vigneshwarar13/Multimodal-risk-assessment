from typing import Dict, Any
import numpy as np


def fuse_features(video_features: Dict[str, Any], speech_features: Dict[str, Any]) -> float:
    v = 0.0
    s = 0.0
    vp = video_features.get("emotion_probs")
    if isinstance(vp, list) and vp:
        arr = np.array(vp, dtype=float)
        v = float(np.clip(arr.mean(), 0.0, 1.0))
    sp = speech_features.get("nlp_prob")
    if isinstance(sp, (float, int)):
        s = float(np.clip(sp, 0.0, 1.0))
    return float(0.5 * v + 0.5 * s)

def fuse_scores(facial_prob: float, speech_prob: float) -> float:
    facial_prob = float(np.clip(facial_prob, 0.0, 1.0))
    speech_prob = float(np.clip(speech_prob, 0.0, 1.0))
    return float(0.5 * facial_prob + 0.5 * speech_prob)

def classify_risk(score: float) -> str:
    score = float(np.clip(score, 0.0, 1.0))
    if score < 0.4:
        return "Good"
    if score < 0.7:
        return "Average"
    return "Poor"

def fuse_and_classify(facial_prob: float, speech_prob: float):
    score = fuse_scores(facial_prob, speech_prob)
    label = classify_risk(score)
    return score, label
