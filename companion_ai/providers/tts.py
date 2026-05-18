import os
import shutil
from pathlib import Path


class LocalTextToSpeechProvider:
    """Simple local TTS provider.

    Swap this class later for a cloud TTS provider if you want a more natural voice.
    """

    def __init__(self):
        self.engine = None

        if _running_in_wsl() and os.getenv("WSL_AUDIO", "0") != "1":
            print("WSL detected: text-to-speech audio disabled. Set WSL_AUDIO=1 to force-enable it.")
            return

        if shutil.which("aplay") is None:
            print("Text-to-speech audio is unavailable: 'aplay' was not found.")
            print("Install alsa-utils, or continue with text-only responses.")
            return

        try:
            import pyttsx3

            self.engine = pyttsx3.init()
        except RuntimeError as exc:
            print(f"Text-to-speech audio is unavailable: {exc}")
            print("Install eSpeak/eSpeak-ng, or continue with text-only responses.")
        except Exception as exc:
            print(f"Text-to-speech audio is unavailable: {exc}")
            print("Continuing with text-only responses.")

    def speak(self, text: str) -> None:
        print(f"AI: {text}")
        if self.engine is None:
            return
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as exc:
            print(f"Text-to-speech playback failed: {exc}")
            self.engine = None


def _running_in_wsl() -> bool:
    if "WSL_DISTRO_NAME" in os.environ:
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False
