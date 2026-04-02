from typing import Dict, Any, Optional, Tuple
import numpy as np

from .face_detection import CV2FaceDetector
from .tf_emotion_model import EmotionModel
from .preprocessing import crop_and_preprocess
from .features import stress_from_emotions
from .video_emotion_pipeline import run_video_emotion


_detector = None
_model = None


def _get_detector() -> CV2FaceDetector:
    # Lazy initialize OpenCV face detector
    global _detector
    if _detector is None:
        _detector = CV2FaceDetector()
    return _detector


def _get_model(model_dir: Optional[str] = None) -> EmotionModel:
    # Lazy initialize TensorFlow emotion model
    global _model
    if _model is None:
        _model = EmotionModel(model_dir=model_dir)
        _model.load()
    return _model


def infer_emotion_on_frame(
    frame_bgr: np.ndarray, model_dir: Optional[str] = None, return_gradcam: bool = False
) -> Dict[str, Any]:
    """
    Run face detection and emotion inference on a single BGR image frame.
    Returns dictionary with:
      - emotions: dict[label -> probability]
      - stress_metrics: dict with 'stress_prob' and 'fear_prob'
      - gradcam: Optional 2D heatmap array (if return_gradcam=True)
    """
    det = _get_detector()
    faces = det.detect(frame_bgr)
    if not faces:
        return {
            "emotions": {},
            "stress_metrics": {"stress_prob": 0.0, "fear_prob": 0.0},
            "gradcam": None,
        }
    x, y, w, h = faces[0]
    face_img = crop_and_preprocess(frame_bgr, x, y, w, h)
    model = _get_model(model_dir)
    probs = model.predict_proba(face_img)
    metrics = {
        "stress_prob": stress_from_emotions(probs),
        "fear_prob": float(probs.get("fear", 0.0)),
    }
    gradcam_map = None
    if return_gradcam:
        try:
            gradcam_map = model.compute_gradcam(face_img)
        except Exception as e:
            gradcam_map = None

    return {
        "emotions": probs,
        "stress_metrics": metrics,
        "gradcam": gradcam_map,
    }


def run_video_pipeline(video_path: Optional[str] = None, enable_camera: bool = False) -> Dict[str, Any]:
    if not video_path:
        return {"emotion_probs": [], "stress_features": [], "timestamps": []}
    out = run_video_emotion(video_path)
    stress_series = [float(f.get("stress_prob", 0.0)) for f in out.get("frames", [])]
    ts = [float(f.get("timestamp", 0.0)) for f in out.get("frames", [])]
    stress_feats = [{"stress_prob": float(f.get("stress_prob", 0.0)), "fear_prob": float(f.get("fear_prob", 0.0))} for f in out.get("frames", [])]
    return {
        "frames": out.get("frames", []),
        "avg_stress_prob": out.get("avg_stress_prob", 0.0),
        "avg_fear_prob": out.get("avg_fear_prob", 0.0),
        "classification": out.get("classification", "Good"),
        "emotion_probs": stress_series,
        "stress_features": stress_feats,
        "timestamps": ts,
    }
