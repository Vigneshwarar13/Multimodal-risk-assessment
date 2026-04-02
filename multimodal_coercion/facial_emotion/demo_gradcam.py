"""
Demonstrate Grad-CAM for facial emotion model.

1. Load model
2. Generate random face or synthetic face rectangle
3. Compute prediction and gradcam
4. Print summary and optionally save overlay
"""

import os
import numpy as np
import cv2
from multimodal_coercion.facial_emotion.tf_emotion_model import EmotionModel
from multimodal_coercion.facial_emotion.pipeline import infer_emotion_on_frame


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def demo_random_face_gradcam():
    print_header("Grad-CAM Demo: Random Input Face")

    # Create random grayscale face stimulus and convert to BGR
    face = np.random.randint(0, 256, (64, 64), dtype="uint8")
    face_bgr = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
    face_bgr = cv2.resize(face_bgr, (160, 160), interpolation=cv2.INTER_AREA)

    model = EmotionModel()
    model.load()

    # Force single face region (for demo, skip detector by manual crop)
    preprocessed = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
    preprocessed = np.expand_dims(preprocessed, axis=-1)
    preprocessed = np.expand_dims(preprocessed, axis=0)

    proba = model.predict_proba(preprocessed)
    print("Predicted emotion probabilities:")
    for lbl, val in proba.items():
        print(f"  {lbl}: {val:.4f}")

    heatmap = model.compute_gradcam(preprocessed)
    print(f"Grad-CAM heatmap min/max: {float(np.min(heatmap)):.4f}/{float(np.max(heatmap)):.4f}")

    overlay = model.overlay_gradcam(preprocessed, heatmap, alpha=0.5)

    output_dir = os.path.join(os.getcwd(), "gradcam_outputs")
    os.makedirs(output_dir, exist_ok=True)

    overlay_path = os.path.join(output_dir, "gradcam_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    print(f"Overlay saved to: {overlay_path}")


def demo_pipeline_function():
    print_header("Grad-CAM Demo via pipeline.infer_emotion_on_frame")

    # Use black image to trigger no-face path
    blank = np.zeros((200, 200, 3), dtype="uint8")
    result = infer_emotion_on_frame(blank, return_gradcam=True)

    print("Pipeline result structure:")
    print(result)


def main():
    print("Grad-CAM facial emotion demo starting...")
    demo_random_face_gradcam()
    demo_pipeline_function()
    print("\n✓ Grad-CAM demo complete")


if __name__ == "__main__":
    main()
