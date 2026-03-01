from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from multimodal_coercion.core.config import project_root


DEFAULT_LABELS = ["neutral", "fear", "anger", "sadness", "happiness"]


def build_cnn(input_shape=(48, 48, 1), num_classes: int = 5) -> tf.keras.Model:
    """
    Small CNN suitable for FER2013-sized grayscale faces.
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


class EmotionModel:
    """
    TensorFlow emotion classifier wrapper.
    Expects Keras model and labels.json in model_dir.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir) if model_dir else None
        self.model: Optional[tf.keras.Model] = None
        self.labels: List[str] = DEFAULT_LABELS.copy()

    def _resolve_paths(self) -> Dict[str, Path]:
        base_dir = self.model_dir
        if base_dir is None:
            base_dir = Path(project_root()) / "models" / "facial_emotion"
        base_dir.mkdir(parents=True, exist_ok=True)
        return {
            "dir": base_dir,
            "keras": base_dir / "emotion_model.keras",
            "labels": base_dir / "labels.json",
            "saved_model": base_dir / "saved_model",
        }

    def load(self):
        paths = self._resolve_paths()
        labels_path = paths["labels"]
        if labels_path.exists():
            with labels_path.open("r", encoding="utf-8") as f:
                self.labels = json.load(f)
        # Try loading Keras model first
        if paths["keras"].exists():
            self.model = tf.keras.models.load_model(paths["keras"])
            return
        # Try saved_model directory
        if paths["saved_model"].exists():
            self.model = tf.keras.models.load_model(paths["saved_model"])
            return
        # If model not found, build untrained model as placeholder
        self.model = build_cnn(num_classes=len(self.labels))

    def predict_proba(self, face_tensor: np.ndarray) -> Dict[str, float]:
        """
        Predict emotion probabilities for a preprocessed face tensor of shape (1,48,48,1).
        Returns dict[label -> probability].
        """
        if self.model is None:
            self.load()
        logits = self.model.predict(face_tensor, verbose=0)
        probs = logits[0].astype("float32")
        probs = probs / (probs.sum() + 1e-8)
        out = {lbl: float(probs[i]) for i, lbl in enumerate(self.labels[: len(probs)])}
        # Ensure all default labels present
        for lbl in self.labels:
            out.setdefault(lbl, 0.0)
        return out
