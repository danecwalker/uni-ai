import io
import re
import wave
from pathlib import Path
from typing import Iterator, Optional


VOICES = [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
    "am_michael", "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]
DEFAULT_VOICE = "af_heart"
SAMPLE_RATE = 24000


class KokoroTts:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.pipelines: dict[str, object] = {}
        self.available = False
        try:
            from kokoro import KPipeline  # noqa: F401
        except ImportError as exc:
            print(f"kokoro not installed: {exc}. Run: pip install kokoro torch")
            return
        try:
            self._get_pipeline("a")
            self.available = True
            print(f"Kokoro-82M TTS loaded ({len(VOICES)} voices)")
        except Exception as exc:
            print(f"Kokoro load failed: {exc}")

    def _get_pipeline(self, lang_code: str):
        if lang_code not in self.pipelines:
            from kokoro import KPipeline
            self.pipelines[lang_code] = KPipeline(lang_code=lang_code)
        return self.pipelines[lang_code]

    def _lang_for_voice(self, voice: str) -> str:
        return "b" if voice.startswith("b") else "a"

    def _iter_pcm(self, text: str, voice: str, speed: float) -> Iterator[bytes]:
        import numpy as np

        pipeline = self._get_pipeline(self._lang_for_voice(voice))
        # split on sentence boundaries so KPipeline yields chunks sooner
        chunks = [c.strip() for c in re.split(r"(?<=[.!?])\s+", text) if c.strip()]
        joined = "\n".join(chunks) if chunks else text
        for _, _, audio in pipeline(joined, voice=voice, speed=speed):
            if audio is None:
                continue
            samples = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
            yield pcm.tobytes()

    def stream_pcm(self, text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> Iterator[bytes]:
        if not self.available:
            return iter(())
        if voice not in VOICES:
            voice = DEFAULT_VOICE
        return self._iter_pcm(text, voice, speed)

    def synth_wav(self, text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> Optional[bytes]:
        if not self.available:
            return None
        if voice not in VOICES:
            voice = DEFAULT_VOICE
        pcm = b"".join(self._iter_pcm(text, voice, speed))
        if not pcm:
            return None
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(pcm)
        return buf.getvalue()
