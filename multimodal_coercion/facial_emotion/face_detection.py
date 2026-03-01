import cv2
from typing import List, Tuple


class CV2FaceDetector:
    """
    OpenCV Haar-cascade based face detector.
    """

    def __init__(self, cascade_name: str = "haarcascade_frontalface_default.xml"):
        cascade_path = cv2.data.haarcascades + cascade_name
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {cascade_path}")

    def detect(self, frame_bgr) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a BGR frame, return list of (x, y, w, h).
        """
        import os
        h, w = frame_bgr.shape[:2]
        max_w = int(os.getenv("FACE_RESIZE_WIDTH", "640"))
        scale = 1.0
        img = frame_bgr
        if w > max_w > 0:
            scale = w / max_w
            new_h = int(h / scale)
            img = cv2.resize(frame_bgr, (max_w, new_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        out = []
        for (x, y, w0, h0) in faces:
            if scale != 1.0:
                x = int(x * scale)
                y = int(y * scale)
                w0 = int(w0 * scale)
                h0 = int(h0 * scale)
            out.append((x, y, w0, h0))
        return out
