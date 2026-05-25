import io
import wave
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve


# Curated Kokoro v1.0 voices. (af_ = American female, am_ = American male,
# bf_ = British female, bm_ = British male)
VOICES = [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
    "am_michael", "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]
DEFAULT_VOICE = "af_heart"

MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


class KokoroTts:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.kokoro = None
        self.available = False
        try:
            import kokoro_onnx  # noqa: F401
        except ImportError as exc:
            print(f"kokoro-onnx not installed: {exc}. Run: pip install kokoro-onnx")
            return
        try:
            self._ensure_files()
            from kokoro_onnx import Kokoro
            self.kokoro = Kokoro(
                str(self.models_dir / "kokoro-v1.0.onnx"),
                str(self.models_dir / "voices-v1.0.bin"),
            )
            self.available = True
            print(f"Kokoro TTS loaded ({len(VOICES)} voices)")
        except Exception as exc:
            print(f"Kokoro load failed: {exc}")

    def _ensure_files(self) -> None:
        for path, url in [
            (self.models_dir / "kokoro-v1.0.onnx", MODEL_URL),
            (self.models_dir / "voices-v1.0.bin", VOICES_URL),
        ]:
            if not path.exists():
                print(f"[kokoro] downloading {url} ({path.name})")
                urlretrieve(url, path)

    def synth_wav(self, text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> Optional[bytes]:
        if not self.available or not self.kokoro:
            return None
        if voice not in VOICES:
            voice = DEFAULT_VOICE
        lang = "en-gb" if voice.startswith("b") else "en-us"
        samples, sample_rate = self.kokoro.create(text, voice=voice, speed=speed, lang=lang)
        # samples: float32 numpy array in [-1, 1]; encode as 16-bit PCM WAV.
        import numpy as np

        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(int(sample_rate))
            wav.writeframes(pcm.tobytes())
        return buf.getvalue()
