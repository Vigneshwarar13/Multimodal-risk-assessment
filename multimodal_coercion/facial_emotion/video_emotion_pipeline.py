import os
from typing import Dict, Any, List
import os
from multimodal_coercion.core.config import Config, project_root, get_config
from .video_preprocessing import iterate_video_frames
from .frame_emotion_inference import FrameEmotionInferer
from .temporal_consistency import analyze_temporal_consistency


def classify(score: float, good_max: float, poor_min: float) -> str:
    if score < good_max:
        return "Good"
    if score >= poor_min:
        return "Poor"
    return "Average"


def run_video_emotion(video_path: str) -> Dict[str, Any]:
    base = project_root()
    cfg = get_config(base)
    t_good = float(cfg.thresholds["good_max"])
    t_poor = float(cfg.thresholds["poor_min"])
    inferer = FrameEmotionInferer()
    # aggregate values as we go to avoid large lists
    stress_sum = 0.0
    fear_sum = 0.0
    count = 0
    stress_series: List[float] = []
    fear_series: List[float] = []
    # optionally keep frames/timestamps only if debugging
    frames = []
    ts_list = []
    max_frames_env = os.getenv("MAX_FRAMES")
    max_frames = int(max_frames_env) if (max_frames_env and max_frames_env.isdigit()) else None
    for idx, ts, frame in iterate_video_frames(video_path):
        probs, stress, fear = inferer.process(frame)
        # only store per-frame detail when DEBUG environment variable is set
        if os.getenv("DEBUG_FRAMES"):
            frames.append({"index": idx, "timestamp": ts, "emotions": probs, "stress_prob": float(stress), "fear_prob": float(fear)})
            ts_list.append(float(ts))
        stress_sum += float(stress)
        fear_sum += float(fear)
        stress_series.append(float(stress))
        fear_series.append(float(fear))
        count += 1
        if max_frames is not None and count >= max_frames:
            break
    avg_stress = stress_sum / count if count else 0.0
    avg_fear = fear_sum / count if count else 0.0
    label = classify(avg_stress, t_good, t_poor)

    # Temporal consistency analysis
    temporal = analyze_temporal_consistency(stress_series, fear_series)

    output = {
        "avg_stress_prob": avg_stress,
        "avg_fear_prob": avg_fear,
        "classification": label,
        "temporal_consistency": temporal,
        "stress_series": stress_series,
        "fear_series": fear_series,
    }
    if os.getenv("DEBUG_FRAMES"):
        output.update({"frames": frames, "timestamps": ts_list})
    return output

