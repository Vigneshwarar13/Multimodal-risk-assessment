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
        # Light CPU threading hints for faster inference on Windows CPUs
        try:
            tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
            tf.config.threading.set_inter_op_parallelism_threads(0)
        except Exception:
            pass
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

    def predict_proba_batch(self, faces: np.ndarray) -> list[Dict[str, float]]:
        """
        Predict for a batch of faces of shape (N,48,48,1).
        Returns list of dicts[label -> probability] for each face.
        """
        if self.model is None:
            self.load()
        logits = self.model.predict(faces, verbose=0)
        outs: list[Dict[str, float]] = []
        for row in logits:
            probs = row.astype("float32")
            probs = probs / (probs.sum() + 1e-8)
            out = {lbl: float(probs[i]) for i, lbl in enumerate(self.labels[: len(probs)])}
            for lbl in self.labels:
                out.setdefault(lbl, 0.0)
            outs.append(out)
        return outs

    def _get_last_conv_layer(self):
        """Return the last Conv2D layer in the model."""
        if self.model is None:
            raise ValueError("Model is not loaded")
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer
        raise ValueError("No Conv2D layer found in model")

    def compute_gradcam(self, face_tensor: np.ndarray, class_index: Optional[int] = None) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single preprocessed face tensor.

        face_tensor: shape (1, H, W, 1), values in [0, 1]
        class_index: optional class index to explain. If None uses model argmax.

        Returns heatmap numpy array shape (H, W) with values in [0,1].
        """
        if self.model is None:
            self.load()

        if face_tensor.ndim != 4 or face_tensor.shape[0] != 1:
            raise ValueError("face_tensor must be shape (1,H,W,1)")

        # Determine target class
        preds = self.model.predict(face_tensor, verbose=0)
        preds = preds.astype("float32")
        if class_index is None:
            class_index = int(np.argmax(preds[0]))

        # Identify last Conv2D layer
        last_conv_layer = self._get_last_conv_layer()

        # Create a model that outputs conv feature maps and predictions
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [last_conv_layer.output, self.model.output],
        )

        # Compute gradient of class output w.r.t. last conv layer
        with tf.GradientTape() as tape:
            conv_output, model_output = grad_model(face_tensor)
            tape.watch(conv_output)
            class_score = model_output[:, class_index]

        grads = tape.gradient(class_score, conv_output)

        if grads is None:
            raise RuntimeError("Could not compute gradients for Grad-CAM")

        # Global average pooling over spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_output = conv_output[0]  # HxWxC
        pooled_grads = pooled_grads

        # Weight channels by importance
        weighted_features = tf.multiply(conv_output, pooled_grads)
        cam = tf.reduce_sum(weighted_features, axis=-1)

        # ReLU and normalize
        cam = tf.nn.relu(cam)
        cam = cam.numpy()

        if np.max(cam) > 0:
            cam = cam / (np.max(cam) + 1e-8)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        # Resize to input dimensions
        target_size = (face_tensor.shape[1], face_tensor.shape[2])
        cam = tf.image.resize(cam[..., np.newaxis], target_size, method="bilinear").numpy()
        cam = np.squeeze(cam)

        # Ensure [0,1]
        cam = np.clip(cam, 0.0, 1.0).astype("float32")
        return cam

    def overlay_gradcam(
        self,
        face_tensor: np.ndarray,
        gradcam_heatmap: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay a Grad-CAM heatmap on grayscale face tensor (for visualization).

        Returns RGB image shape (H, W, 3) in uint8.
        """
        import cv2

        if face_tensor.ndim != 4 or face_tensor.shape[0] != 1:
            raise ValueError("face_tensor must be shape (1,H,W,1)")

        gray = np.squeeze(face_tensor[0])
        gray_255 = (gray * 255.0).astype("uint8")
        gray_rgb = cv2.cvtColor(gray_255, cv2.COLOR_GRAY2BGR)

        heatmap = (gradcam_heatmap * 255.0).astype("uint8")
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(gray_rgb, 1.0 - alpha, heatmap_color, alpha, 0)
        return overlay
