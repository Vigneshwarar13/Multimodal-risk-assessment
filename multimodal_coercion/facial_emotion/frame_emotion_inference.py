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
        # Batch all detected faces for a single forward pass
        face_tensors = []
        for (x, y, w, h) in faces:
            face_tensors.append(crop_and_preprocess(frame_bgr, x, y, w, h))  # (1,48,48,1)
        batch = np.concatenate(face_tensors, axis=0)  # (N,48,48,1)
        probs_list = self.model.predict_proba_batch(batch)
        # aggregate
        sum_probs: Dict[str, float] = {}
        stress_sum = 0.0
        fear_sum = 0.0
        for probs in probs_list:
            for lbl, val in probs.items():
                sum_probs[lbl] = sum_probs.get(lbl, 0.0) + float(val)
            stress_sum += stress_from_emotions(probs)
            fear_sum += float(probs.get("fear", 0.0))
        count = len(probs_list)
        if count == 0:
            return {}, 0.0, 0.0
        avg_probs = {lbl: sum_probs[lbl] / count for lbl in sum_probs}
        avg_stress = stress_sum / count
        avg_fear = fear_sum / count
        return avg_probs, avg_stress, avg_fear
