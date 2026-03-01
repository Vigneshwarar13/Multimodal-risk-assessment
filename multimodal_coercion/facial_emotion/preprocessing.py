import cv2
import numpy as np


def crop_and_preprocess(frame_bgr: np.ndarray, x: int, y: int, w: int, h: int, size: int = 48) -> np.ndarray:
    """
    Crop face region, convert to grayscale, resize to FER2013 size, and normalize to [0,1].
    Returns array of shape (1, size, size, 1) float32.
    """
    h_img, w_img = frame_bgr.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    crop = frame_bgr[y : y + h, x : x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    arr = resized.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=-1)  # (H,W,1)
    arr = np.expand_dims(arr, axis=0)   # (1,H,W,1)
    return arr

