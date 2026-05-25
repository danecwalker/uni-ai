from pathlib import Path
from typing import Optional

from companion_ai.emotion.detector import EMOTION_LABELS, EmotionObservation


class FrameEmotionClassifier:
    """Classifies a single decoded BGR frame using the same FER+ ONNX model.

    Decoupled from webcam capture so it can be driven by frames uploaded from
    the browser.
    """

    def __init__(self, emotion_model_path: str):
        self.available = False
        self.cv2 = None
        self.np = None
        self.face_cascade = None
        self.emotion_net = None

        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            print(f"OpenCV unavailable: {exc}")
            return

        self.cv2 = cv2
        self.np = np

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            print("Face cascade failed to load.")
            return
        self.face_cascade = cascade

        if not emotion_model_path:
            print("No EMOTION_MODEL_PATH configured; web emotion disabled.")
            return
        model = Path(emotion_model_path)
        if not model.exists():
            print(f"EMOTION_MODEL_PATH does not exist: {model}.")
            return
        try:
            self.emotion_net = cv2.dnn.readNetFromONNX(str(model))
            self.available = True
            print(f"Web emotion model loaded: {model}")
        except Exception as exc:
            print(f"Could not load emotion ONNX: {exc}")

    def classify_jpeg(self, data: bytes) -> Optional[EmotionObservation]:
        if not self.available:
            return None
        cv2 = self.cv2
        np = self.np
        buf = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64)
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = cv2.resize(gray[y : y + h, x : x + w], (64, 64), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(64, 64))
        self.emotion_net.setInput(blob)
        output = self.emotion_net.forward().reshape(-1)
        shifted = output - np.max(output)
        probs = np.exp(shifted) / np.sum(np.exp(shifted))
        idx = int(np.argmax(probs))
        return EmotionObservation(
            label=EMOTION_LABELS[idx],
            confidence=float(probs[idx]),
            face_box=(int(x), int(y), int(w), int(h)),
        )
