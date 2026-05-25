import json
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from companion_ai.config import load_settings
from companion_ai.conversation import (
    START_CONVERSATION_TOOL,
    SYSTEM_PROMPT,
    _extract_inline_tool_calls,
    _user_wants_to_end_conversation,
)
from companion_ai.emotion.signal import (
    EmotionSignalSettings,
    IdleDecisionCooldown,
    RollingEmotionSignal,
)
from companion_ai.providers.llm import GroqChatProvider
from companion_ai.tools.registry import default_registry
from web.emotion import FrameEmotionClassifier
from web.faces import FaceSystem


settings = load_settings()
llm = GroqChatProvider(settings.groq_api_key, settings.groq_chat_model)
tools = default_registry()
classifier = FrameEmotionClassifier(settings.emotion_model_path)
faces = FaceSystem(Path(__file__).parent.parent / "models" / "faces")

emotion_signal = RollingEmotionSignal(
    EmotionSignalSettings(
        window=settings.emotion_signal_window,
        min_non_neutral=settings.emotion_signal_min_non_neutral,
        min_avg_confidence=settings.emotion_signal_min_avg_confidence,
        spike_confidence=settings.emotion_signal_spike_confidence,
        spike_frames=settings.emotion_signal_spike_frames,
    )
)
idle_cooldown = IdleDecisionCooldown(settings.idle_decision_cooldown_seconds)

state = {
    "mode": "idle",
    "messages": [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(tools=tools.list_descriptions()),
        }
    ],
    # Latest emotion observation: {label, confidence, ts} or None
    "last_emotion": None,
}


def _reset():
    state["mode"] = "idle"
    state["messages"] = state["messages"][:1]
    emotion_signal.clear()
    idle_cooldown.clear()


app = FastAPI()

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index():
    from fastapi.responses import FileResponse

    return FileResponse(static_dir / "index.html")


@app.get("/favicon.ico")
def favicon():
    from fastapi.responses import Response

    return Response(status_code=204)


@app.get("/test")
def test_page():
    from fastapi.responses import FileResponse

    return FileResponse(static_dir / "test.html")


class ChatIn(BaseModel):
    text: str
    emotion_hint: Optional[str] = None
    confidence: Optional[float] = None


ENROLL_PERSON_TOOL = {
    "type": "function",
    "function": {
        "name": "enroll_person",
        "description": (
            "Save the face of the person currently on the webcam under a name. "
            "Use this only after the user has stated their name (e.g. 'I'm Sara', "
            "'my name is Alex') or explicitly asks you to remember them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The person's preferred name."},
                "notes": {
                    "type": "string",
                    "description": "Optional short profile note (e.g. 'works in design, likes tea').",
                },
            },
            "required": ["name"],
        },
    },
}

UPDATE_PROFILE_TOOL = {
    "type": "function",
    "function": {
        "name": "update_person_profile",
        "description": (
            "Persist new information about a known person to their long-term profile "
            "so future conversations can reference it. Use whenever the user volunteers "
            "something worth remembering — interests, hobbies, allergies, dietary needs, "
            "preferences, topics to avoid, or general notes. Use sparingly and only "
            "with information the user shares about themselves; never store anything "
            "sensitive (medical, financial, sexuality, etc.) without being explicitly "
            "asked to."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "person_id": {
                    "type": "string",
                    "description": "ID of the person — taken verbatim from the system context line 'Person in view: <name> (id=<id>)'.",
                },
                "interests": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Hobbies, fandoms, sports, music, books, games, etc.",
                },
                "allergies": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Food or environmental allergies they tell you about.",
                },
                "dietary": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Dietary preferences: vegan, vegetarian, kosher, halal, gluten-free, etc.",
                },
                "preferences": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Small preferences worth recalling: 'loves dad jokes', 'prefers short answers', 'morning person'.",
                },
                "topics_to_avoid": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Topics they've said they don't want to discuss.",
                },
                "notes": {
                    "type": "string",
                    "description": "Free-form notes that don't fit a category. Will be appended to existing notes.",
                },
            },
            "required": ["person_id"],
        },
    },
}


@app.post("/api/chat")
def chat(payload: ChatIn):
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(400, "empty text")

    hint = ""
    if payload.emotion_hint:
        c = f"{payload.confidence:.2f}" if payload.confidence is not None else "unknown"
        hint = (
            f"\nEmotion hint from webcam: {payload.emotion_hint} ({c}). "
            "Treat this as uncertain."
        )

    # Inject current scene (emotion + recognised person + profile) as a system message.
    context_bits = []
    em = state.get("last_emotion")
    if em and (time.time() - em["ts"] < 30):
        context_bits.append(
            f"Emotion read (uncertain): {em['label']} ({em['confidence']:.2f}). "
            "Use as a soft companionship signal, never as a diagnosis."
        )
    if faces.available:
        s = faces.scene_context_string()
        if s:
            context_bits.append(s)
    if context_bits:
        state["messages"].append({"role": "system", "content": " ".join(context_bits)})

    if state["mode"] != "conversation":
        state["mode"] = "conversation"

    user_wants_end = _user_wants_to_end_conversation(text)
    state["messages"].append({"role": "user", "content": text + hint})

    tools = []
    if faces.available:
        tools.extend([ENROLL_PERSON_TOOL, UPDATE_PROFILE_TOOL])
    result = llm.complete(state["messages"], tools=tools or None)
    response, inline_calls = _extract_inline_tool_calls(result.get("content", ""))
    tool_calls = result.get("tool_calls", []) + inline_calls

    reply = response or ""
    if reply:
        state["messages"].append({"role": "assistant", "content": reply})

    for call in tool_calls:
        name = call.get("name")
        if name == "return_to_idle":
            # Ignored — only an explicit user goodbye ends the conversation.
            print("[chat] LLM tried return_to_idle — ignored.")
            continue
        elif name == "enroll_person":
            args = call.get("arguments", {}) or {}
            person_name = (args.get("name") or "").strip()
            notes_arg = (args.get("notes") or "").strip()
            if person_name and faces.available:
                saved = faces.enroll_from_last_frame(person_name, notes_arg)
                note = (f"[memory] enrolled person {person_name}." if saved
                        else f"[memory] could not enroll {person_name} — no face visible.")
                state["messages"].append({"role": "system", "content": note})
        elif name == "update_person_profile":
            args = call.get("arguments", {}) or {}
            pid = (args.get("person_id") or "").strip()
            if pid and faces.available:
                updates = {k: args.get(k) for k in
                           ("interests", "allergies", "dietary", "preferences", "topics_to_avoid", "notes")
                           if args.get(k)}
                result_obj = faces.update_profile(pid, updates)
                if result_obj is None:
                    note = f"[memory] update failed — unknown person {pid}."
                elif not result_obj.get("added"):
                    note = "[memory] nothing new to add to the profile."
                else:
                    note = f"[memory] updated profile for {result_obj.get('name')}: {result_obj['added']}"
                state["messages"].append({"role": "system", "content": note})

    # If the LLM only called a saving tool and didn't say anything, ask it to
    # produce a natural acknowledgement so the conversation keeps flowing.
    saved_tool_fired = any(
        call.get("name") in ("enroll_person", "update_person_profile")
        for call in tool_calls
    )
    if saved_tool_fired and not reply:
        # Nudge the model to acknowledge in a human way, never mentioning the system.
        state["messages"].append({
            "role": "system",
            "content": (
                "Reply now in one warm, natural sentence acknowledging what you "
                "just remembered — like a friend would ('okay, I'll remember that', "
                "'got it — nice to meet you, X'). Don't mention saving, databases, "
                "embeddings, or that you 'detected' anything."
            ),
        })
        try:
            followup = llm.complete(state["messages"])
            followup_text, _ = _extract_inline_tool_calls(followup.get("content", ""))
            followup_text = followup_text.strip()
            if followup_text:
                state["messages"].append({"role": "assistant", "content": followup_text})
                reply = followup_text
        except Exception as exc:
            print(f"[chat] follow-up after tool failed: {exc}")

    if user_wants_end:
        state["mode"] = "idle"

    return {"reply": reply, "mode": state["mode"]}


@app.post("/api/stt")
async def stt(audio: UploadFile = File(...)):
    if llm.client is None:
        raise HTTPException(503, "Groq not configured")
    # Reuse the existing Groq client from the STT provider for transcription.
    from groq import Groq

    data = await audio.read()
    suffix = "." + (audio.filename.rsplit(".", 1)[-1] if audio.filename and "." in audio.filename else "webm")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        client = Groq(api_key=settings.groq_api_key)
        with open(tmp_path, "rb") as f:
            tr = client.audio.transcriptions.create(file=f, model=settings.groq_stt_model)
        return {"text": tr.text.strip()}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


from web.tts import VOICES as KOKORO_VOICES, DEFAULT_VOICE as KOKORO_DEFAULT, KokoroTts

kokoro_tts = KokoroTts(Path(__file__).parent.parent / "models" / "kokoro")


@app.get("/api/voices")
def voices():
    return {
        "voices": KOKORO_VOICES,
        "default": KOKORO_DEFAULT,
        "available": kokoro_tts.available,
        "provider": "kokoro" if kokoro_tts.available else "browser",
    }


class TtsIn(BaseModel):
    text: str
    voice: Optional[str] = KOKORO_DEFAULT


@app.post("/api/tts")
def tts(payload: TtsIn):
    if not kokoro_tts.available:
        raise HTTPException(503, "kokoro-onnx not installed (pip install kokoro-onnx)")
    from fastapi.responses import Response
    try:
        audio = kokoro_tts.synth_wav(payload.text, payload.voice or KOKORO_DEFAULT)
    except Exception as exc:
        raise HTTPException(502, f"TTS failed: {exc}")
    if not audio:
        raise HTTPException(502, "TTS produced no audio")
    return Response(content=audio, media_type="audio/wav")


@app.post("/api/vision/snapshot")
async def vision_snapshot(frame: UploadFile = File(...)):
    """Face-only snapshot. Detect in idle, freeze last_faces during conversation."""
    data = await frame.read()
    scene: dict = {"available": True, "faces": [], "person_trigger": None}
    if not faces.available:
        return scene
    if state["mode"] == "idle":
        face_results = faces.detect(data)
        scene["faces"] = [
            {"bbox": f["bbox"], "match": f["match"], "det_score": f.get("det_score", 0.0)}
            for f in face_results
        ]
        trigger = faces.pop_greeting_trigger()
        if trigger:
            opener = _person_opener(trigger)
            if opener:
                state["mode"] = "conversation"
                state["messages"].append(
                    {"role": "system", "content": f"Person greeting context: {json.dumps(trigger)}"}
                )
                state["messages"].append({"role": "assistant", "content": opener})
                scene["person_trigger"] = {"opener": opener, **trigger}
    else:
        scene["faces"] = [
            {"bbox": f["bbox"], "match": f["match"], "det_score": f.get("det_score", 0.0)}
            for f in faces.last_faces
        ]
    return scene


def _emotion_hint_for_opener() -> str:
    em = state.get("last_emotion")
    if not em or (time.time() - em["ts"] > 30):
        return ""
    if em["label"] == "neutral":
        return ""
    return (
        f"\n- Their facial expression reads as {em['label']} ({em['confidence']:.2f}) — "
        "uncertain, treat as a soft signal. Acknowledge it gently without naming the "
        "label, and let it shape your tone (warmer for sad, gentler for fear, match "
        "energy for happy, calm for angry/disgust)."
    )


def _person_opener(trigger: dict) -> Optional[str]:
    """Ask the LLM to produce a short, natural opener tailored to who's in view."""
    emotion_line = _emotion_hint_for_opener()
    if trigger["type"] == "known_person":
        p = trigger["person"]
        notes_line = f"\n- Past notes: {p['notes']}" if p.get("notes") else ""
        seen_line = ""
        if p.get("last_seen"):
            mins = max(1, int((time.time() - p["last_seen"]) / 60))
            seen_line = f"\n- Last seen ~{mins} min ago"
        prompt = (
            f"You just recognised {p['name']} on the webcam.{seen_line}{notes_line}{emotion_line}\n"
            "Greet them warmly by name in one or two short sentences — natural, like seeing a friend. "
            "If their expression suggests something, weave a gentle, hedged check-in into the "
            "greeting ('hey Sara, good to see you — you look a bit tired, everything okay?'). "
            "Don't explain that the camera 'recognised' them or 'detected' anything."
        )
    else:
        prompt = (
            "You just noticed a new face on the webcam — someone you don't recognise."
            f"{emotion_line}\n"
            "Greet them gently in one or two short sentences, ask their name so you can remember "
            "them, and if their expression suggests an emotion weave a hedged check-in in too. "
            "Don't be creepy or overly formal."
        )
    decision_messages = state["messages"] + [{"role": "user", "content": prompt}]
    try:
        result = llm.complete(decision_messages)
    except Exception as exc:
        print(f"[faces] opener LLM failed: {exc}")
        return None
    text, _ = _extract_inline_tool_calls(result.get("content", ""))
    return text.strip() or None


@app.get("/api/people")
def people_list():
    return {"available": faces.available, "people": faces.list_people()}


@app.get("/api/people/{person_id}")
def people_get(person_id: str):
    p = faces.get_profile(person_id)
    if not p:
        raise HTTPException(404, "unknown person")
    return p


@app.delete("/api/people/{person_id}")
def people_forget(person_id: str):
    return {"ok": faces.forget(person_id)}


@app.post("/api/emotion")
async def emotion(frame: UploadFile = File(...)):
    if not classifier.available:
        return {"available": False}
    data = await frame.read()
    obs = classifier.classify_jpeg(data)

    response = {"available": True, "observation": None, "trigger": None, "summary": None}
    if obs is None:
        return response

    response["observation"] = {"label": obs.label, "confidence": obs.confidence}
    # Cache for chat context so even non-frontend-attached turns get emotional cues.
    state["last_emotion"] = {
        "label": obs.label,
        "confidence": float(obs.confidence),
        "ts": time.time(),
    }

    if state["mode"] == "conversation":
        emotion_signal.clear()
        idle_cooldown.clear()
        return response

    summary = emotion_signal.add(obs)
    response["summary"] = {
        "dominant": summary.dominant_emotion,
        "avg": summary.average_confidence,
        "max": summary.max_confidence,
        "non_neutral": summary.non_neutral_frames,
        "total": summary.total_frames,
        "threshold_met": summary.threshold_met,
    }

    now = time.monotonic()
    if summary.threshold_met and idle_cooldown.can_consider(now):
        opener = _consider(summary)
        emotion_signal.clear()
        if opener:
            idle_cooldown.clear()
            response["trigger"] = {"opener": opener, "dominant": summary.dominant_emotion}
        else:
            idle_cooldown.mark_staying_idle(now)
    return response


def _consider(summary) -> Optional[str]:
    prompt = (
        "Idle mode webcam observation:\n"
        f"- Dominant non-neutral emotion label: {summary.dominant_emotion}\n"
        f"- Average non-neutral confidence: {summary.average_confidence:.2f}\n"
        f"- Maximum non-neutral confidence: {summary.max_confidence:.2f}\n"
        f"- Non-neutral frames: {summary.non_neutral_frames}/{summary.total_frames}\n"
        f"- Recent non-neutral distribution: {summary.distribution_text or 'none'}\n\n"
        "Strong bias toward starting a conversation. If warranted, call "
        "start_conversation with a short, gentle, hedged opener. Otherwise call no tool "
        "and produce no user-facing text."
    )
    decision_messages = state["messages"] + [{"role": "user", "content": prompt}]
    result = llm.complete(decision_messages, tools=[START_CONVERSATION_TOOL])
    for call in result.get("tool_calls", []):
        if call.get("name") == "start_conversation":
            opener = call.get("arguments", {}).get("opener") or (
                f"I might be misreading this, but you seem a little {summary.dominant_emotion}. "
                "Want to talk about it?"
            )
            state["mode"] = "conversation"
            state["messages"].append({"role": "user", "content": prompt})
            state["messages"].append({"role": "assistant", "content": opener})
            return opener
    return None


@app.post("/api/reset")
def reset():
    _reset()
    return {"ok": True, "mode": state["mode"]}


@app.get("/api/state")
def get_state():
    return {
        "mode": state["mode"],
        "emotion_available": classifier.available,
        "llm_available": llm.client is not None,
    }
