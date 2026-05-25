import os
import tempfile
from pathlib import Path


class GroqSpeechToTextProvider:
    def __init__(self, api_key: str, model: str, sample_rate: int = 16_000):
        self.model = model
        self.sample_rate = sample_rate
        self.client = None

        if not api_key:
            print("GROQ_API_KEY is not set. Speech input will use typed text.")
            return

        try:
            from groq import Groq
        except ImportError:
            print("Groq package is not installed. Speech input will use typed text.")
            return

        try:
            self.client = Groq(api_key=api_key)
        except Exception as exc:
            print(f"Groq speech client could not be initialized: {exc}")
            print("Speech input will use typed text.")

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
        if self.client is None:
            return _typed_input("Speech transcription is unavailable.")

        if _running_in_wsl() and os.getenv("WSL_AUDIO", "0") != "1":
            print("WSL detected: microphone input disabled, so using typed input.")
            print("Set WSL_AUDIO=1 to force-enable Linux audio capture.")
            return _typed_input()

        try:
            import numpy as np
            import sounddevice as sd
            import soundfile as sf
        except Exception as exc:
            print(f"Microphone audio is unavailable: {exc}")
            print("Install PortAudio, or type the user's response below for now.")
            return _typed_input()

        chunk_seconds = 0.05
        chunk_size = int(self.sample_rate * chunk_seconds)
        max_chunks = int(max_seconds / chunk_seconds)
        silence_chunks_required = int(silence_seconds / chunk_seconds)
        calibration_chunks = int(0.5 / chunk_seconds)
        preroll_chunks = int(0.3 / chunk_seconds)

        from collections import deque

        preroll = deque(maxlen=preroll_chunks)
        frames = []
        has_speech = False
        silent_chunks = 0
        noise_samples = []
        active_threshold = silence_threshold
        debug = os.getenv("STT_DEBUG", "0") == "1"

        print("Listening... speak when ready. I will stop after you go quiet.")
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=chunk_size,
            ) as stream:
                for i in range(max_chunks):
                    chunk, _ = stream.read(chunk_size)
                    rms = float(np.sqrt(np.mean(np.square(chunk))))

                    if i < calibration_chunks:
                        noise_samples.append(rms)
                        preroll.append(chunk.copy())
                        if i == calibration_chunks - 1:
                            noise_floor = float(np.median(noise_samples))
                            # Speak when 3x ambient or user-set floor, whichever higher.
                            active_threshold = max(silence_threshold, noise_floor * 3.0)
                            print(
                                f"Mic calibrated: noise_floor={noise_floor:.4f}, "
                                f"speech_threshold={active_threshold:.4f}"
                            )
                        continue

                    if debug:
                        print(f"rms={rms:.4f} thr={active_threshold:.4f} speech={has_speech} silent={silent_chunks}")

                    if rms >= active_threshold:
                        if not has_speech:
                            frames.extend(preroll)
                        has_speech = True
                        silent_chunks = 0
                    elif has_speech:
                        silent_chunks += 1
                    else:
                        preroll.append(chunk.copy())

                    if has_speech:
                        frames.append(chunk.copy())

                    if has_speech and silent_chunks >= silence_chunks_required:
                        break
        except Exception as exc:
            print(f"Microphone recording failed: {exc}")
            print("Using typed input instead.")
            return _typed_input()

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
            return _typed_input()
        return self._transcribe_audio(audio, sf)

    def _transcribe_audio(self, audio, sf) -> str:
        if self.client is None:
            return _typed_input("Speech transcription is unavailable.")

        with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
            sf.write(wav.name, audio, self.sample_rate)
            with open(wav.name, "rb") as file:
                try:
                    transcription = self.client.audio.transcriptions.create(
                        file=file,
                        model=self.model,
                    )
                except Exception as exc:
                    print(f"Speech transcription failed: {exc}")
                    print("Using typed input instead.")
                    return _typed_input()
        return transcription.text.strip()


def _running_in_wsl() -> bool:
    if "WSL_DISTRO_NAME" in os.environ:
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def _typed_input(reason: str = "") -> str:
    if reason:
        print(f"{reason} Type the user's response below.")
    try:
        return input("User: ").strip()
    except EOFError:
        raise KeyboardInterrupt
