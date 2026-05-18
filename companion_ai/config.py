from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_chat_model: str = "llama-3.1-70b-versatile"
    groq_stt_model: str = "whisper-large-v3-turbo"
    camera_index: int = 0
    emotion_model_path: str = ""
    emotion_trigger: str = "sad"
    emotion_confidence: float = 0.55
    emotion_frames_required: int = 8
    voice_sample_seconds: int = 8
    silence_threshold: float = 0.01
    silence_seconds: float = 1.2
    max_listen_seconds: int = 30


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_chat_model=os.getenv("GROQ_CHAT_MODEL", "llama-3.1-70b-versatile"),
        groq_stt_model=os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo"),
        camera_index=int(os.getenv("CAMERA_INDEX", "0")),
        emotion_model_path=os.getenv("EMOTION_MODEL_PATH", ""),
        emotion_trigger=os.getenv("EMOTION_TRIGGER", "sad"),
        emotion_confidence=float(os.getenv("EMOTION_CONFIDENCE", "0.55")),
        emotion_frames_required=int(os.getenv("EMOTION_FRAMES_REQUIRED", "8")),
        voice_sample_seconds=int(os.getenv("VOICE_SAMPLE_SECONDS", "8")),
        silence_threshold=float(os.getenv("SILENCE_THRESHOLD", "0.01")),
        silence_seconds=float(os.getenv("SILENCE_SECONDS", "1.2")),
        max_listen_seconds=int(os.getenv("MAX_LISTEN_SECONDS", "30")),
    )
