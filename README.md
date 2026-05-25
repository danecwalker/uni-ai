# Companion AI

A face-aware, voice-driven companion AI. The browser captures webcam + mic and
talks to a FastAPI backend that runs face recognition, emotion reading, neural
VAD, conversational LLM, and local TTS. The UI is a centred Three.js particle
orb that morphs colour and pulse with conversation state.

## Features

- **Face recognition + persistent profiles** — InsightFace + Chroma vector DB.
  Greets you by name when you arrive, asks for a name when it doesn't know you,
  and remembers per-person profiles (interests, allergies, dietary needs,
  preferences, topics_to_avoid, free-text notes) it can reference in future
  conversations.
- **Browser-side neural VAD** — Silero VAD via `@ricky0123/vad-web` in an
  AudioWorklet. No push-to-talk; the assistant just listens.
- **Webcam emotion read** — FER+ ONNX, treated as a soft companionship cue
  (warmer tone for sadness, gentler for fear, matched energy for happy).
- **Groq LLM** — `llama-3.3-70b-versatile` by default, with tool calling for
  `enroll_person` and `update_person_profile`.
- **Local TTS** — Kokoro v1.0 (28 voices), selectable in the UI.
- **Groq Whisper STT** — turn-by-turn transcription.
- **Animated orb** — particle sphere + morphing neon ring + bloom, driven by
  state (idle / armed / listening / thinking / talking) with audio-amplitude
  reactivity.

## Safety note

This is not a medical or crisis-support system. Face/emotion inferences are
imperfect — soft signals, not diagnoses. The assistant never names emotion
labels out loud and never stores medical/financial/sensitive info unless the
user explicitly asks it to. Add explicit consent, retention rules, and crisis
escalation guidance before deploying to real users.

## Quick start (Docker — recommended)

```bash
cp .env.example .env
# set GROQ_API_KEY=gsk_...
docker compose up -d --build
# open http://localhost:8000
```

Full walkthrough including HTTPS / LAN-access notes, model-volume layout,
and CPU/RAM expectations: see **[DEPLOY.md](./DEPLOY.md)**.

## Quick start (bare-metal Python)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# system libs (Ubuntu/Debian)
sudo apt install espeak-ng libgl1 libglib2.0-0

# emotion model (~34 MB)
curl -L -o models/emotion-ferplus-8.onnx \
  https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx

cp .env.example .env  # then set GROQ_API_KEY
.venv/bin/uvicorn web.app:app --host 0.0.0.0 --port 8000
```

On first run, InsightFace's buffalo_l (~280 MB) and Kokoro voices (~340 MB)
auto-download into `models/`.

## How it works

```
                   browser
  ┌─────────────────────────────────────┐
  │  webcam ▶ /api/vision/snapshot      │     idle: face detect + recognise
  │  webcam ▶ /api/emotion              │     FER+ emotion → companionship cue
  │  mic    ▶ Silero VAD (AudioWorklet) │     local; only sends audio on speech_end
  │           ▶ /api/stt (Groq Whisper) │
  │  POST ◀▶ /api/chat (Groq LLM)       │     scene + profile + emotion injected
  │  speaker ◀ /api/tts (Kokoro)        │
  │  Three.js orb ◀ state + audio amp   │
  └─────────────────────────────────────┘
                   backend
  ┌─────────────────────────────────────┐
  │  FastAPI + uvicorn                  │
  │  ├─ FaceSystem (InsightFace + Chroma) │
  │  ├─ FER+ emotion classifier         │
  │  ├─ Kokoro TTS                      │
  │  └─ Groq client (chat + Whisper)    │
  └─────────────────────────────────────┘
```

## Conversation lifecycle

1. **Idle** — face detector runs every ~3.5 s. When someone is seen:
   - **Known person** (cosine sim ≥ 0.85): LLM crafts a personalised opener
     using their profile and current emotion. Cooldown: 5 min per person.
   - **Unknown face**: assistant gently asks for a name. When the user answers,
     LLM calls `enroll_person` and saves the face. Cooldown: 2 min.
2. **Conversation** — face detection pauses (we already know who's there);
   profile + emotion get injected into every chat turn. As the user volunteers
   info, the LLM calls `update_person_profile` to extend their profile.
3. **Goodbye** — only an explicit farewell from the user (`bye`, `goodbye`,
   `see you later`, `take care`, `talk later`…) returns to idle. The LLM
   itself cannot end the conversation.

## Endpoints

| Method  | Path                          | Purpose                               |
|---------|-------------------------------|---------------------------------------|
| GET     | `/`                           | UI                                    |
| GET     | `/api/state`                  | mode + provider availability          |
| GET     | `/api/voices`                 | Kokoro voice list                     |
| POST    | `/api/chat`                   | text turn → reply + audio side-effect |
| POST    | `/api/stt`                    | audio upload → transcript             |
| POST    | `/api/tts`                    | text → WAV                            |
| POST    | `/api/emotion`                | jpeg frame → emotion observation      |
| POST    | `/api/vision/snapshot`        | jpeg frame → face detections          |
| GET     | `/api/people`                 | list all enrolled people              |
| GET     | `/api/people/{id}`            | full profile for one person           |
| DELETE  | `/api/people/{id}`            | forget one person                     |
| POST    | `/api/reset`                  | force back to idle, clear convo state |

## Project layout

```text
companion_ai/             # Original CLI loop + shared providers / signal logic
  config.py
  conversation.py         # System prompt + helpers reused by the web app
  providers/              # GroqChatProvider, GroqSpeechToTextProvider
  emotion/                # FER+ detector + rolling-window signal aggregator
  tools/                  # Plugin registry for future specialised tools
web/                      # FastAPI app
  app.py                  # Endpoints, chat-turn orchestration, LLM tools
  faces.py                # InsightFace + Chroma + per-person profile schema
  emotion.py              # Frame-level FER+ classifier
  tts.py                  # Kokoro adapter
  static/
    index.html
    app.js                # Browser-side flow: VAD loop, mic / video capture,
                          #   bbox overlay, blob state machine, chat plumbing
    blob.js               # Three.js orb (particle sphere + ring + bloom)
    vad.js                # @ricky0123/vad-web wrapper
    style.css
.github/workflows/
  docker.yml              # Build + push image to ghcr.io
Dockerfile
docker-compose.yml
DEPLOY.md
```

## Tuning

All key thresholds live in `.env` (overrides) and `companion_ai/config.py`
(defaults). The ones you'll touch most:

| Var                                  | Default | What                                           |
|--------------------------------------|---------|------------------------------------------------|
| `GROQ_CHAT_MODEL`                    | `llama-3.3-70b-versatile` | swap to `openai/gpt-oss-120b` for stronger tool calls |
| `EMOTION_SIGNAL_MIN_AVG_CONFIDENCE`  | `0.12`  | lower = more likely to trigger emotion-based check-in |
| `IDLE_DECISION_COOLDOWN_SECONDS`     | `4`     | min seconds between idle-trigger LLM calls     |

Hard-coded knobs worth knowing about:
- `MATCH_SIMILARITY_THRESHOLD` in `web/faces.py` (default `0.85`) — face match strictness
- `KNOWN_GREET_COOLDOWN_SEC` (default `300`) — re-greet window per person
- `UNKNOWN_GREET_COOLDOWN_SEC` (default `120`)
- `positiveSpeechThreshold` / `negativeSpeechThreshold` in `web/static/vad.js`
  — Silero VAD sensitivity

## Privacy

Face embeddings and profiles are stored locally in
`models/faces/chroma_faces/`. To nuke everything:

```bash
docker compose down
rm -rf models/faces/chroma_faces
docker compose up -d
```

Or remove individuals via the API:

```bash
curl http://localhost:8000/api/people | jq .
curl -X DELETE http://localhost:8000/api/people/person_<id>
```

## Adding specialised tools

Tools are wired into the LLM via the JSON-schema list in `web/app.py`
(`UPDATE_PROFILE_TOOL`, `ENROLL_PERSON_TOOL`, `START_CONVERSATION_TOOL`) and
handled in the `/api/chat` tool-call loop. Add a new entry, implement the
side-effect, and the LLM can use it on the next turn.

For larger plugin-style features (calendars, reminders, journaling, university
support resources), the older `companion_ai/tools/registry.py` is still in
place and can be exposed as additional tool definitions.

## License

MIT (see `LICENSE` if present, otherwise add one).
