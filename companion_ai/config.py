from dataclasses import dataclass
import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_chat_model: str = "llama-3.1-70b-versatile"
    groq_stt_model: str = "whisper-large-v3-turbo"
    camera_index: int = 0
    emotion_model_path: str = ""
    emotion_trigger: str = "sad"
    emotion_confidence: float = 0.35
    emotion_frames_required: int = 3
    emotion_signal_window: int = 10
    emotion_signal_min_non_neutral: int = 3
    emotion_signal_min_avg_confidence: float = 0.20
    emotion_signal_spike_confidence: float = 0.45
    emotion_signal_spike_frames: int = 2
    idle_decision_cooldown_seconds: float = 10.0
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
        emotion_confidence=float(os.getenv("EMOTION_CONFIDENCE", "0.35")),
        emotion_frames_required=int(os.getenv("EMOTION_FRAMES_REQUIRED", "3")),
        emotion_signal_window=int(os.getenv("EMOTION_SIGNAL_WINDOW", "10")),
        emotion_signal_min_non_neutral=int(os.getenv("EMOTION_SIGNAL_MIN_NON_NEUTRAL", "3")),
        emotion_signal_min_avg_confidence=float(
            os.getenv("EMOTION_SIGNAL_MIN_AVG_CONFIDENCE", "0.20")
        ),
        emotion_signal_spike_confidence=float(
            os.getenv("EMOTION_SIGNAL_SPIKE_CONFIDENCE", "0.45")
        ),
        emotion_signal_spike_frames=int(os.getenv("EMOTION_SIGNAL_SPIKE_FRAMES", "2")),
        idle_decision_cooldown_seconds=float(
            os.getenv("IDLE_DECISION_COOLDOWN_SECONDS", "10")
        ),
        voice_sample_seconds=int(os.getenv("VOICE_SAMPLE_SECONDS", "8")),
        silence_threshold=float(os.getenv("SILENCE_THRESHOLD", "0.01")),
        silence_seconds=float(os.getenv("SILENCE_SECONDS", "1.2")),
        max_listen_seconds=int(os.getenv("MAX_LISTEN_SECONDS", "30")),
    )
