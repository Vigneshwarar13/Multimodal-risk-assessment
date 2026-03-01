from typing import Dict, Tuple
import numpy as np

from .face_detection import CV2FaceDetector
from .preprocessing import crop_and_preprocess
from .tf_emotion_model import EmotionModel
from .features import stress_from_emotions


class FrameEmotionInferer:
    def __init__(self, model_dir: str | None = None):
        self.detector = CV2FaceDetector()
        self.model = EmotionModel(model_dir=model_dir)
        self.model.load()

    def process(self, frame_bgr: np.ndarray) -> Tuple[Dict[str, float], float, float]:
        faces = self.detector.detect(frame_bgr)
        if not faces:
            return {}, 0.0, 0.0
        probs_list = []
        stress_list = []
        fear_list = []
        for (x, y, w, h) in faces:
            face_img = crop_and_preprocess(frame_bgr, x, y, w, h)
            probs = self.model.predict_proba(face_img)
            probs_list.append(probs)
            stress_list.append(stress_from_emotions(probs))
            fear_list.append(float(probs.get("fear", 0.0)))
        if not probs_list:
            return {}, 0.0, 0.0
        labels = list(probs_list[0].keys())
        avg_probs = {lbl: float(np.mean([p.get(lbl, 0.0) for p in probs_list])) for lbl in labels}
        avg_stress = float(np.mean(stress_list)) if stress_list else 0.0
        avg_fear = float(np.mean(fear_list)) if fear_list else 0.0
        return avg_probs, avg_stress, avg_fear

