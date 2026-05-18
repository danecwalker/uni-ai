from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ONNX Model Zoo FER+ output order.
# Keep labels short because settings use EMOTION_TRIGGER=sad by default.
EMOTION_LABELS = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]


@dataclass(frozen=True)
class EmotionObservation:
    label: str
    confidence: float


class WebcamEmotionDetector:
    """OpenCV-based webcam emotion detector.

    OpenCV handles webcam access, face detection, and optional emotion-model
    inference. The bundled default is compatible with the ONNX Model Zoo FER+
    model, which expects 64x64 grayscale faces and outputs:
    neutral, happy, surprise, sad, angry, disgust, fear, contempt.
    """

    def __init__(self, camera_index: int = 0, emotion_model_path: str = ""):
        self.camera_index = camera_index
        self.capture = None
        self.emotion_net = None
        self.available = False

        device_path = Path(f"/dev/video{camera_index}")
        if Path("/dev").exists() and not device_path.exists():
            print(f"No webcam found at {device_path}. Webcam emotion detection disabled.")
            return

        self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            print(f"Could not open webcam index {camera_index}. Webcam emotion detection disabled.")
            self.capture.release()
            self.capture = None
            return

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.face_cascade.empty():
            print("OpenCV face cascade could not be loaded. Emotion detection disabled.")
            return

        if emotion_model_path:
            model = Path(emotion_model_path)
            if model.exists():
                self.emotion_net = cv2.dnn.readNetFromONNX(str(model))
                self.available = True
                print(f"OpenCV emotion model loaded: {model}")
            else:
                print(f"EMOTION_MODEL_PATH does not exist: {model}")
        else:
            print(
                "Webcam is available, but no OpenCV emotion model is configured. "
                "Set EMOTION_MODEL_PATH to an ONNX emotion model to enable emotion triggers."
            )

    def read_emotion(self) -> Optional[EmotionObservation]:
        if self.capture is None or self.emotion_net is None:
            return None

        ok, frame = self.capture.read()
        if not ok:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(64, 64),
        )
        if len(faces) == 0:
            return None

        # Use the largest detected face.
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        face = gray[y : y + h, x : x + w]
        face = cv2.resize(face, (64, 64))
        face = face.astype("float32") / 255.0

        blob = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(64, 64))
        self.emotion_net.setInput(blob)
        output = self.emotion_net.forward().reshape(-1)
        probabilities = self._softmax(output)
        index = int(np.argmax(probabilities))

        return EmotionObservation(
            label=EMOTION_LABELS[index],
            confidence=float(probabilities[index]),
        )

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values)
        exp = np.exp(shifted)
        return exp / np.sum(exp)
