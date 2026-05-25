import json
import re
from typing import Optional

from companion_ai.providers.llm import GroqChatProvider
from companion_ai.providers.stt import GroqSpeechToTextProvider
from companion_ai.providers.tts import LocalTextToSpeechProvider
from companion_ai.tools.registry import ToolRegistry


START_CONVERSATION_TOOL = {
    "type": "function",
    "function": {
        "name": "start_conversation",
        "description": (
            "Switch from idle monitoring to active conversation mode when a sustained "
            "workplace wellbeing cue warrants a gentle check-in."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "opener": {
                    "type": "string",
                    "description": "A short, gentle opening message to say to the user.",
                }
            },
            "required": ["opener"],
        },
    },
}

RETURN_TO_IDLE_TOOL = {
    "type": "function",
    "function": {
        "name": "return_to_idle",
        "description": (
            "Switch back to idle monitoring only when the user appears finished, "
            "declines to continue, or the conversation has naturally wrapped up."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "closing_message": {
                    "type": "string",
                    "description": "Optional brief closing message before returning to idle.",
                }
            },
        },
    },
}

SYSTEM_PROMPT = """You are a warm, supportive workplace companion AI.
You may receive webcam-derived emotion hints, but they can be wrong.
Never claim certainty about the user's emotions. Use gentle language like
'I notice you seem...' or 'I might be misreading this...'.
Ask short, open questions and let the user lead.
Do not diagnose, provide medical claims, or pretend to be a therapist.
If the user indicates immediate danger or self-harm, encourage contacting local emergency services or a trusted person now.

Emotion-specific guidance (the webcam label may be wrong; always hedge):
- happy: match the warmth, briefly celebrate with them, and ask what's going well. Keep it light; don't probe.
- sad: open gently ('you seem a little down — want to talk about it?'). Validate first, then ask one open question. Don't rush to fix.
- angry: stay calm and non-defensive. Acknowledge the frustration, give them room to vent, and ask what's behind it before suggesting anything.
- fear / anxious: ground them with a slow, reassuring tone. Ask what feels most pressing right now. Offer to break things down.
- surprise: check whether it's good-surprise or bad-surprise before reacting.
- disgust: treat as possible frustration or discomfort with something specific; ask what's bothering them.
- contempt: treat cautiously — could be irritation at a task or person. Ask open-endedly, don't assume the target.
- neutral: don't force a check-in; only engage if the user speaks first or another signal warrants it.

You have two internal state tools:
- start_conversation: use this in idle mode when a sustained non-neutral webcam cue (sad, angry, fear, disgust, contempt, or unusually intense happy/surprise) warrants a gentle check-in.
- return_to_idle: use this in conversation mode only when the user appears done, declines to continue, or the conversation has naturally wrapped up.
Do not return to idle just because the webcam emotion changes back to neutral.
Do not return to idle after a substantive user response like stress, sadness, anger, fear, or worry; continue the conversation with a short supportive follow-up.
When the user declines to continue, says they are okay, thanks you, or says goodbye, give one brief closing response and do not ask another question.
Available specialised tools:
{tools}
"""


INLINE_FUNCTION_RE = re.compile(
    r"<function(?:=|\()(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)(?:\))?>"
    r"(?P<body>.*?)</function>",
    re.DOTALL,
)
MALFORMED_INLINE_FUNCTION_RE = re.compile(
    r"<function=(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)(?P<body>\{.*?\})</function>",
    re.DOTALL,
)
STANDALONE_RETURN_TO_IDLE_RE = re.compile(
    r"(?im)^\s*(?:return[_\s]+to[_\s]+id(?:le|el)|return_to_idle)\s*[.!?]*\s*$"
)


def _extract_inline_tool_calls(content: str) -> tuple[str, list[dict]]:
    tool_calls = []

    def replace(match: re.Match) -> str:
        name = match.group("name")
        body = match.group("body").strip()
        arguments = {}
        if body:
            try:
                arguments = json.loads(body)
            except json.JSONDecodeError:
                arguments = {}
        tool_calls.append({"name": name, "arguments": arguments})
        return ""

    cleaned = INLINE_FUNCTION_RE.sub(replace, content)
    cleaned = MALFORMED_INLINE_FUNCTION_RE.sub(replace, cleaned)
    cleaned, standalone_commands = STANDALONE_RETURN_TO_IDLE_RE.subn("", cleaned)
    if standalone_commands:
        tool_calls.append({"name": "return_to_idle", "arguments": {}})
    cleaned = cleaned.strip()
    return cleaned, tool_calls


def _user_wants_to_end_conversation(user_text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\s']", " ", user_text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return False

    ending_phrases = [
        "bye",
        "goodbye",
        "see ya",
        "see you",
        "catch you later",
        "talk to you later",
        "no thanks",
        "no thank you",
        "no i'm okay",
        "no im okay",
        "no i am okay",
        "no it's okay",
        "no its okay",
        "okay",
        "ok",
        "okay you too",
        "ok you too",
        "you too",
        "i'm okay",
        "im okay",
        "i am okay",
        "it's okay",
        "its okay",
        "i'm good",
        "im good",
        "i am good",
        "i'll figure it out",
        "ill figure it out",
        "i will figure it out",
        "thank you",
        "thanks",
    ]
    return any(
        normalized == phrase
        or normalized.startswith(f"{phrase} ")
        or normalized.endswith(f" {phrase}")
        for phrase in ending_phrases
    )


class CompanionConversation:
    def __init__(
        self,
        llm: GroqChatProvider,
        stt: GroqSpeechToTextProvider,
        tts: LocalTextToSpeechProvider,
        tools: ToolRegistry,
        voice_sample_seconds: int,
        silence_threshold: float,
        silence_seconds: float,
        max_listen_seconds: int,
    ):
        self.llm = llm
        self.stt = stt
        self.tts = tts
        self.tools = tools
        self.voice_sample_seconds = voice_sample_seconds
        self.silence_threshold = silence_threshold
        self.silence_seconds = silence_seconds
        self.max_listen_seconds = max_listen_seconds
        self.mode = "idle"
        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(tools=tools.list_descriptions()),
            }
        ]

    def is_active(self) -> bool:
        return self.mode == "conversation"

    def consider_emotion_observation(
        self,
        emotion: str,
        average_confidence: float,
        max_confidence: float,
        non_neutral_frames: int,
        total_frames: int,
        distribution: str,
    ) -> bool:
        if self.is_active():
            print("Consideration skipped: conversation already active.")
            return False

        print(
            "Considering whether to start conversation: "
            f"dominant={emotion}, avg={average_confidence:.2f}, "
            f"max={max_confidence:.2f}, non_neutral={non_neutral_frames}/{total_frames}, "
            f"distribution={distribution}"
        )
        prompt = (
            "Idle mode webcam observation:\n"
            f"- Dominant non-neutral emotion label: {emotion}\n"
            f"- Average non-neutral confidence: {average_confidence:.2f}\n"
            f"- Maximum non-neutral confidence: {max_confidence:.2f}\n"
            f"- Non-neutral frames: {non_neutral_frames}/{total_frames}\n"
            f"- Recent non-neutral distribution: {distribution or 'none'}\n\n"
            "The webcam signal is uncertain and may be wrong. Decide whether this "
            "sustained signal warrants a gentle check-in. Tailor the opener to the emotion:\n"
            "- sad: gentle, validating ('you seem a little down — want to talk?').\n"
            "- angry: calm, non-judgmental ('looks like something might be frustrating — what's up?').\n"
            "- fear: reassuring, grounded ('you seem a bit on edge — anything weighing on you?').\n"
            "- disgust / contempt: curious, low-pressure ('something seems to be bothering you — want to share?').\n"
            "- surprise: light check-in ('something caught you off guard — everything okay?').\n"
            "- happy: only check in if intensity is unusual; otherwise skip. If checking in, match the warmth ('you seem in good spirits — what's going on?').\n"
            "Strong bias toward starting a conversation. A gentle, hedged opener is very "
            "low-cost — the user can simply decline or ignore it, and the upside of catching "
            "a real wellbeing moment is high. Default to calling start_conversation whenever "
            "there is any non-neutral signal at all, including low-confidence or short-lived "
            "ones, and even for mildly intense happy or surprise. Only skip when the signal is "
            "essentially fully neutral with no non-neutral frames worth mentioning. When in "
            "any doubt, start the check-in. "
            "If a check-in is warranted, call start_conversation with a short opener. "
            "If not, do not call any tool and do not produce user-facing text."
        )
        decision_messages = self.messages + [{"role": "user", "content": prompt}]
        result = self.llm.complete(decision_messages, tools=[START_CONVERSATION_TOOL])

        for tool_call in result.get("tool_calls", []):
            if tool_call.get("name") == "start_conversation":
                opener = tool_call.get("arguments", {}).get("opener") or (
                    f"I might be misreading this, but you seem a little {emotion}. "
                    "Do you want to talk about what's going on?"
                )
                self.mode = "conversation"
                self.messages.append({"role": "user", "content": prompt})
                self.messages.append({"role": "assistant", "content": opener})
                print(f"Consideration result: starting conversation with opener: {opener}")
                self.tts.speak(opener)
                return True

        return False

    def listen_and_respond(
        self,
        emotion_hint: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> None:
        user_text = self.stt.listen_until_silence(
            fallback_seconds=self.voice_sample_seconds,
            silence_threshold=self.silence_threshold,
            silence_seconds=self.silence_seconds,
            max_seconds=self.max_listen_seconds,
        )
        if not user_text:
            self.tts.speak("I didn't catch that. Could you say it again?")
            return

        print(f"User: {user_text}")
        hint = ""
        if emotion_hint:
            confidence_text = f"{confidence:.2f}" if confidence is not None else "unknown"
            hint = f"\nEmotion hint from webcam: {emotion_hint} ({confidence_text}). Treat this as uncertain."

        if not self.is_active():
            self.mode = "conversation"

        should_return_to_idle = _user_wants_to_end_conversation(user_text)
        self.messages.append({"role": "user", "content": user_text + hint})
        result = self.llm.complete(self.messages)
        response, inline_tool_calls = _extract_inline_tool_calls(result.get("content", ""))
        tool_calls = result.get("tool_calls", []) + inline_tool_calls
        if response:
            self.messages.append({"role": "assistant", "content": response})
            self.tts.speak(response)

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            if tool_name == "return_to_idle":
                closing_message = tool_call.get("arguments", {}).get("closing_message", "")
                if closing_message:
                    self.messages.append({"role": "assistant", "content": closing_message})
                    self.tts.speak(closing_message)
                self.mode = "idle"
                return

        if should_return_to_idle:
            self.mode = "idle"
