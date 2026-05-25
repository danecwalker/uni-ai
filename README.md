# Companion AI

A modular companion AI prototype that can:

- Watch the user through a webcam and infer their emotional state.
- Listen to the user via microphone (STT).
- Use Groq for conversational inference.
- Speak responses through TTS.
- Route optional specialised tools through a simple plugin registry.

## Safety note

This is not a medical or crisis-support system. Emotion detection is imperfect and should be treated as a soft signal, not a diagnosis. If you build this into a real product, add explicit consent, privacy controls, data retention rules, and crisis escalation guidance.

## Setup

The app can start with only the Python standard library installed. Missing
optional packages disable the matching feature instead of crashing:

- no `python-dotenv`: environment is read directly from the shell
- no `groq` or `GROQ_API_KEY`: offline text-only replies and typed input
- no `opencv-python` / `numpy`: webcam emotion detection is disabled
- no `sounddevice` / `soundfile` / PortAudio: typed input is used
- no `pyttsx3` / audio backend: responses are printed as text

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

If microphone recording fails with `PortAudio library not found`, install PortAudio:

```bash
# Ubuntu/Debian
sudo apt install portaudio19-dev

# macOS
brew install portaudio
```

Without PortAudio, the app now falls back to typed input so you can keep testing.

Emotion detection now uses OpenCV only. Webcam and face detection work out of the box. This repo is configured to use the downloaded ONNX Model Zoo FER+ model:

```env
EMOTION_MODEL_PATH=models/emotion-ferplus-8.onnx
```

If no model is configured, the app starts in chat-only mode instead of crashing.

If no webcam is available, the app now starts in chat-only mode. On Linux, check cameras with:

```bash
ls /dev/video*
```

If your camera is not `/dev/video0`, set `CAMERA_INDEX` in `.env` to the correct number.

If text-to-speech fails with an eSpeak/eSpeak-ng error, install it:

```bash
# Ubuntu/Debian
sudo apt install espeak-ng
```

Without eSpeak, the app now continues with text-only AI responses.

Microphone input uses silence detection instead of a fixed countdown. Tune it in `.env`:

```env
SILENCE_THRESHOLD=0.01
SILENCE_SECONDS=1.2
MAX_LISTEN_SECONDS=30
```

### WSL notes

WSL usually has no direct webcam at `/dev/video0` and may have no Linux audio devices at `/dev/snd`. In WSL, the app will safely fall back to chat-only / typed-input mode unless you explicitly pass devices through.

Useful WSL options:

- Run the app natively on Windows if you need easy webcam/microphone access.
- For webcam passthrough, use `usbipd-win` and confirm the camera appears as `/dev/video*`.
- For audio playback tools inside WSL, install:

```bash
sudo apt install alsa-utils espeak-ng portaudio19-dev
```

Set your Groq API key in `.env`:

```env
GROQ_API_KEY=your_key_here
```

## Run

```bash
python3 -m companion_ai.main
```

The assistant watches for sustained non-neutral webcam cues, treats them as uncertain, and asks the AI whether a gentle workplace-wellbeing check-in is warranted. If it sees a sustained signal, it may begin with something like:

> I notice you seem a little unhappy. Do you want to talk about what's going on?

## Project layout

```text
companion_ai/
  main.py                 # Application loop
  config.py               # Environment/config loading
  conversation.py         # Conversation orchestration
  emotion/                # Webcam/emotion detection providers
  providers/              # LLM, STT, TTS providers
  tools/                  # Extensible specialised tools
```

## Adding specialised tools

Create a tool function and register it in `companion_ai/tools/registry.py`. Tools are intentionally isolated so later you can add things like calendars, reminders, journaling, wellbeing exercises, or university support resources without rewriting the conversation loop.
