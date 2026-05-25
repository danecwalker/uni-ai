# Deploying with Docker

Packages the FastAPI backend, the Three.js orb UI, and all on-device inference
(Silero VAD in-browser, FER+ emotion, InsightFace recognition, Kokoro TTS) into
a single container. Browser handles webcam + mic + audio playback, so the
container does **not** need a `/dev/video` device.

## 1. Configure `.env`

```bash
cp .env.example .env
# Required:
#   GROQ_API_KEY=gsk_...
#   GROQ_CHAT_MODEL=llama-3.3-70b-versatile
```

| Var | Why |
|---|---|
| `GROQ_API_KEY` | Chat + Whisper STT. console.groq.com → keys. |
| `GROQ_CHAT_MODEL` | `llama-3.3-70b-versatile`, `openai/gpt-oss-120b`, etc. |
| `GROQ_STT_MODEL` | `whisper-large-v3-turbo`. |

All `EMOTION_*` and `IDLE_DECISION_*` vars have sensible defaults in
`companion_ai/config.py`; override here only when tuning.

## 2. Build + run

```bash
docker compose up -d --build
```

First build pulls Python deps + the small FER+ emotion ONNX. Image lands at
~1.5 GB.

On first request to `/`, two larger models lazy-download into the mounted
`./models/` volume (so they survive restarts):

- **InsightFace buffalo_l** bundle (~280 MB) — face detection + recognition
- **Kokoro voices** (~340 MB on first TTS call) — neural TTS

## 3. Open the app

```
http://localhost:8000
```

**HTTPS / non-localhost note:** browsers refuse `getUserMedia` (webcam + mic)
on insecure origins other than `localhost`. To reach the app from another
device on your LAN, run it behind a reverse proxy with TLS (Caddy, Nginx with
Let's Encrypt) — or for quick testing, enable
`chrome://flags/#unsafely-treat-insecure-origin-as-secure` for the LAN URL.

## 4. Useful commands

```bash
# tail logs — face detection, profile updates, LLM tool calls show here
docker compose logs -f companion-ai

# rebuild app layers after a code change
docker compose build && docker compose up -d

# stop
docker compose down

# nuke all stored people (forget every profile)
docker compose down
rm -rf models/faces/chroma_faces
docker compose up -d

# shell into the container
docker compose exec companion-ai bash

# inspect what the app has remembered about people
curl -s http://localhost:8000/api/people | python3 -m json.tool
curl -s http://localhost:8000/api/people/<person_id> | python3 -m json.tool

# forget one specific person
curl -X DELETE http://localhost:8000/api/people/<person_id>
```

## 5. What lives where

| Path inside container | What |
|---|---|
| `/app/baked_models/emotion-ferplus-8.onnx` | FER+ emotion model (baked into image) |
| `/app/models/faces/insightface/`            | InsightFace buffalo_l weights (auto-downloaded) |
| `/app/models/faces/chroma_faces/`           | Chroma vector DB — your enrolled people + profiles |
| `/app/models/kokoro/`                       | Kokoro ONNX + voice bundle (auto-downloaded) |

Only `/app/models` is volume-mounted; the rest is part of the image.

## 6. CPU / RAM

Per webcam snapshot (every 3.5 s while idle):
- InsightFace detect + recognise: ~120 ms

Per chat turn:
- LLM call to Groq: 600–1500 ms (dominant)
- Kokoro TTS for a short sentence: ~600 ms

Silero VAD runs in the browser (WASM/AudioWorklet) — zero server cost.

Steady-state RAM with all models warm: ~700 MB resident.
