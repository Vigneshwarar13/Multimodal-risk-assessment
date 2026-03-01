from typing import Dict, Any, List
from multimodal_coercion.core.config import Config, project_root
from .video_preprocessing import iterate_video_frames
from .frame_emotion_inference import FrameEmotionInferer


def classify(score: float, good_max: float, poor_min: float) -> str:
    if score < good_max:
        return "Good"
    if score >= poor_min:
        return "Poor"
    return "Average"


def run_video_emotion(video_path: str) -> Dict[str, Any]:
    base = project_root()
    cfg = Config(base)
    t_good = float(cfg.thresholds["good_max"])
    t_poor = float(cfg.thresholds["poor_min"])
    inferer = FrameEmotionInferer()
    frames: List[Dict[str, Any]] = []
    stress_vals: List[float] = []
    fear_vals: List[float] = []
    ts_list: List[float] = []
    for idx, ts, frame in iterate_video_frames(video_path):
        probs, stress, fear = inferer.process(frame)
        frames.append(
            {"index": idx, "timestamp": ts, "emotions": probs, "stress_prob": float(stress), "fear_prob": float(fear)}
        )
        stress_vals.append(float(stress))
        fear_vals.append(float(fear))
        ts_list.append(float(ts))
    avg_stress = float(sum(stress_vals) / len(stress_vals)) if stress_vals else 0.0
    avg_fear = float(sum(fear_vals) / len(fear_vals)) if fear_vals else 0.0
    label = classify(avg_stress, t_good, t_poor)
    return {
        "frames": frames,
        "avg_stress_prob": avg_stress,
        "avg_fear_prob": avg_fear,
        "classification": label,
        "timestamps": ts_list,
    }

