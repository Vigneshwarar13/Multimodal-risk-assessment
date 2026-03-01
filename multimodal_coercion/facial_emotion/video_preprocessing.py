from typing import Iterator, Tuple
from pathlib import Path
import os
import numpy as np


def iterate_video_frames(video_path: str) -> Iterator[Tuple[int, float, np.ndarray]]:
    import cv2

    p = Path(video_path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx = 0
    stride = max(1, int(os.getenv("FRAME_STRIDE", "5")))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                ts = idx / fps
                yield idx, ts, frame
            idx += 1
    finally:
        cap.release()
