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
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

