import os
import tempfile
from pathlib import Path

from groq import Groq


class GroqSpeechToTextProvider:
    def __init__(self, api_key: str, model: str, sample_rate: int = 16_000):
        if not api_key:
            raise ValueError("GROQ_API_KEY is required")
        self.client = Groq(api_key=api_key)
        self.model = model
        self.sample_rate = sample_rate

    def listen(self, seconds: int = 8) -> str:
        return self.listen_until_silence(fallback_seconds=seconds)

    def listen_until_silence(
        self,
        fallback_seconds: int = 8,
        silence_threshold: float = 0.01,
        silence_seconds: float = 1.2,
        max_seconds: int = 30,
    ) -> str:
        """Record until the user stops speaking, then transcribe with Groq.

        Uses simple RMS volume detection. Recording starts when speech is first
        detected, stops after `silence_seconds` of quiet, and never exceeds
        `max_seconds`. In WSL/no-audio environments this falls back to typing.
        """
        if _running_in_wsl() and os.getenv("WSL_AUDIO", "0") != "1":
            print("WSL detected: microphone input disabled, so using typed input.")
            print("Set WSL_AUDIO=1 to force-enable Linux audio capture.")
            return input("User: ").strip()

        try:
            import numpy as np
            import sounddevice as sd
            import soundfile as sf
        except OSError as exc:
            print(f"Microphone audio is unavailable: {exc}")
            print("Install PortAudio, or type the user's response below for now.")
            return input("User: ").strip()

        chunk_seconds = 0.1
        chunk_size = int(self.sample_rate * chunk_seconds)
        max_chunks = int(max_seconds / chunk_seconds)
        silence_chunks_required = int(silence_seconds / chunk_seconds)

        frames = []
        has_speech = False
        silent_chunks = 0

        print("Listening... speak when ready. I will stop after you go quiet.")
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=chunk_size,
            ) as stream:
                for _ in range(max_chunks):
                    chunk, _ = stream.read(chunk_size)
                    rms = float(np.sqrt(np.mean(np.square(chunk))))

                    if rms >= silence_threshold:
                        has_speech = True
                        silent_chunks = 0
                    elif has_speech:
                        silent_chunks += 1

                    if has_speech:
                        frames.append(chunk.copy())

                    if has_speech and silent_chunks >= silence_chunks_required:
                        break
        except Exception as exc:
            print(f"Microphone recording failed: {exc}")
            print("Using typed input instead.")
            return input("User: ").strip()

        if not frames:
            print(f"No speech detected. Recording a fixed {fallback_seconds}s sample instead...")
            return self._listen_fixed(fallback_seconds, sf, sd)

        audio = np.concatenate(frames, axis=0)
        return self._transcribe_audio(audio, sf)

    def _listen_fixed(self, seconds: int, sf, sd) -> str:
        try:
            audio = sd.rec(
                int(seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
        except Exception as exc:
            print(f"Microphone recording failed: {exc}")
            print("Using typed input instead.")
            return input("User: ").strip()
        return self._transcribe_audio(audio, sf)

    def _transcribe_audio(self, audio, sf) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
            sf.write(wav.name, audio, self.sample_rate)
            with open(wav.name, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=file,
                    model=self.model,
                )
        return transcription.text.strip()


def _running_in_wsl() -> bool:
    if "WSL_DISTRO_NAME" in os.environ:
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False
