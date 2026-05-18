from companion_ai.providers.llm import GroqChatProvider
from companion_ai.providers.stt import GroqSpeechToTextProvider
from companion_ai.providers.tts import LocalTextToSpeechProvider
from companion_ai.tools.registry import ToolRegistry


SYSTEM_PROMPT = """You are a warm, supportive companion AI.
You may receive webcam-derived emotion hints, but they can be wrong.
Never claim certainty about the user's emotions. Use gentle language like
'I notice you seem...' or 'I might be misreading this...'.
Ask short, open questions and let the user lead.
Do not diagnose, provide medical claims, or pretend to be a therapist.
If the user indicates immediate danger or self-harm, encourage contacting local emergency services or a trusted person now.
Available specialised tools:
{tools}
"""


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
        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(tools=tools.list_descriptions()),
            }
        ]

    def start_emotion_check_in(self, emotion: str, confidence: float) -> None:
        opener = (
            f"I might be misreading this, but you seem a little {emotion}. "
            "Do you want to talk about what's going on?"
        )
        self.tts.speak(opener)
        self.messages.append({"role": "assistant", "content": opener})
        self.listen_and_respond(emotion_hint=emotion, confidence=confidence)

    def listen_and_respond(self, emotion_hint: str | None = None, confidence: float | None = None) -> None:
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
            hint = f"\nEmotion hint from webcam: {emotion_hint} ({confidence:.2f}). Treat this as uncertain."

        self.messages.append({"role": "user", "content": user_text + hint})
        response = self.llm.reply(self.messages)
        self.messages.append({"role": "assistant", "content": response})
        self.tts.speak(response)
