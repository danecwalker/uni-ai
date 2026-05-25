from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional, Tuple


# ONNX Model Zoo FER+ output order.
# Keep labels short because settings use EMOTION_TRIGGER=sad by default.
EMOTION_LABELS = [
    "neutral",
    "happy",
    "surprise",
    "sad",
    "angry",
    "disgust",
    "fear",
    "contempt",
]


@dataclass(frozen=True)
class EmotionObservation:
    label: str
    confidence: float
    face_box: Tuple[int, int, int, int]


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
        self.cv2 = None
        self.np = None
        self.webcam_available = False
        self.stop_requested = False
        self.available = False

        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            print(f"OpenCV emotion detection is unavailable: {exc}")
            print("Install optional dependencies, or continue in chat-only mode.")
            return

        self.cv2 = cv2
        self.np = np

        device_path = Path(f"/dev/video{camera_index}")
        if sys.platform.startswith("linux") and not device_path.exists():
            print(f"No webcam found at {device_path}. Webcam emotion detection disabled.")
            return

        if sys.platform == "darwin":
            self.capture = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        else:
            self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            print(f"Could not open webcam index {camera_index}. Webcam emotion detection disabled.")
            self.capture.release()
            self.capture = None
            return
        self.webcam_available = True

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.face_cascade.empty():
            print("OpenCV face cascade could not be loaded. Emotion detection disabled.")
            return

        if emotion_model_path:
            model = Path(emotion_model_path)
            if model.exists():
                try:
                    self.emotion_net = cv2.dnn.readNetFromONNX(str(model))
                    self.available = True
                    print(f"OpenCV emotion model loaded: {model}")
                except Exception as exc:
                    print(f"OpenCV emotion model could not be loaded: {exc}")
                    print("Emotion triggers disabled.")
            else:
                print(f"EMOTION_MODEL_PATH does not exist: {model}. Emotion triggers disabled.")
        else:
            print(
                "Webcam is available, but no OpenCV emotion model is configured. "
                "Set EMOTION_MODEL_PATH to an ONNX emotion model to enable emotion triggers."
            )

    def read_emotion(self, show_frame: bool = False) -> Optional[EmotionObservation]:
        if (
            self.capture is None
            or self.emotion_net is None
            or self.cv2 is None
            or self.np is None
        ):
            return None
        cv2 = self.cv2
        np = self.np

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
            if show_frame:
                self._show_frame(frame, None)
            return None

        # Use the largest detected face.
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        face = gray[y : y + h, x : x + w]
        face = cv2.resize(face, (64, 64), interpolation=cv2.INTER_AREA)

        # FER+ expects grayscale face pixels on the original 0..255 scale.
        blob = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(64, 64))
        self.emotion_net.setInput(blob)
        output = self.emotion_net.forward().reshape(-1)
        probabilities = self._softmax(output)
        index = int(np.argmax(probabilities))

        observation = EmotionObservation(
            label=EMOTION_LABELS[index],
            confidence=float(probabilities[index]),
            face_box=(int(x), int(y), int(w), int(h)),
        )

        if show_frame:
            self._show_frame(frame, observation)

        return observation

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
        if self.cv2 is not None:
            try:
                self.cv2.destroyAllWindows()
            except Exception:
                pass

    def _show_frame(self, frame, observation: Optional[EmotionObservation]) -> None:
        cv2 = self.cv2
        if cv2 is None:
            return

        if observation is None:
            label = "No face detected"
            cv2.putText(
                frame,
                label,
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            x, y, w, h = observation.face_box
            label = f"{observation.label} {observation.confidence:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(
                frame,
                label,
                (x, max(24, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Companion AI Emotion", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.stop_requested = True

    def _softmax(self, values):
        np = self.np
        if np is None:
            return values
        shifted = values - np.max(values)
        exp = np.exp(shifted)
        return exp / np.sum(exp)
