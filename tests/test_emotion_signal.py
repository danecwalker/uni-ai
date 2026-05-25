import contextlib
import io
import unittest

from companion_ai.conversation import (
    CompanionConversation,
    _extract_inline_tool_calls,
    _user_wants_to_end_conversation,
)
from companion_ai.emotion.detector import EmotionObservation
from companion_ai.emotion.signal import (
    EmotionSignalSettings,
    IdleDecisionCooldown,
    RollingEmotionSignal,
)


def observation(label: str, confidence: float) -> EmotionObservation:
    return EmotionObservation(label=label, confidence=confidence, face_box=(0, 0, 1, 1))


class RollingEmotionSignalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = EmotionSignalSettings()

    def test_low_confidence_sustained_sadness_triggers(self) -> None:
        signal = RollingEmotionSignal(self.settings)

        for _ in range(8):
            summary = signal.add(observation("sad", 0.35))
        for _ in range(4):
            summary = signal.add(observation("neutral", 0.40))

        self.assertTrue(summary.threshold_met)
        self.assertTrue(summary.sustained_threshold_met)
        self.assertEqual(summary.dominant_emotion, "sad")
        self.assertEqual(summary.non_neutral_frames, 8)
        self.assertAlmostEqual(summary.average_confidence, 0.35)

    def test_mixed_non_neutral_signal_uses_dominant_label_and_distribution(self) -> None:
        signal = RollingEmotionSignal(self.settings)

        for _ in range(7):
            summary = signal.add(observation("sad", 0.36))
        for _ in range(5):
            summary = signal.add(observation("disgust", 0.36))

        self.assertTrue(summary.threshold_met)
        self.assertEqual(summary.dominant_emotion, "sad")
        self.assertEqual(summary.distribution, {"sad": 7, "disgust": 5})
        self.assertEqual(summary.distribution_text, "sad=7 disgust=5")

    def test_neutral_heavy_window_does_not_trigger(self) -> None:
        signal = RollingEmotionSignal(self.settings)

        for _ in range(7):
            summary = signal.add(observation("sad", 0.50))
        for _ in range(5):
            summary = signal.add(observation("neutral", 0.50))

        self.assertFalse(summary.threshold_met)
        self.assertEqual(summary.non_neutral_frames, 7)

    def test_spike_case_triggers_before_window_is_full(self) -> None:
        signal = RollingEmotionSignal(self.settings)

        for _ in range(4):
            summary = signal.add(observation("angry", 0.65))

        self.assertTrue(summary.threshold_met)
        self.assertTrue(summary.spike_threshold_met)
        self.assertEqual(summary.total_frames, 4)


class IdleDecisionCooldownTests(unittest.TestCase):
    def test_ai_decline_starts_cooldown_and_blocks_repeat_consideration(self) -> None:
        cooldown = IdleDecisionCooldown(seconds=30.0)

        self.assertTrue(cooldown.can_consider(100.0))
        cooldown.mark_staying_idle(100.0)

        self.assertFalse(cooldown.can_consider(129.9))
        self.assertAlmostEqual(cooldown.seconds_remaining(129.9), 0.1)
        self.assertTrue(cooldown.can_consider(130.0))


class FakeLlm:
    def __init__(self, result: dict):
        self.result = result
        self.messages = []
        self.tools = []

    def complete(self, messages, tools=None, tool_choice="auto"):
        self.messages.append(messages)
        self.tools.append(tools)
        return self.result


class FakeTts:
    def __init__(self):
        self.spoken = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


class FakeStt:
    def __init__(self, text: str = ""):
        self.text = text

    def listen_until_silence(self, **kwargs) -> str:
        return self.text


class FakeTools:
    def list_descriptions(self) -> str:
        return ""


class ConversationDecisionTests(unittest.TestCase):
    def test_ai_starts_conversation_and_mode_becomes_conversation(self) -> None:
        opener = "I might be misreading this, but you seem tense. Want to pause for a minute?"
        conversation = CompanionConversation(
            llm=FakeLlm(
                {
                    "content": "",
                    "tool_calls": [
                        {"name": "start_conversation", "arguments": {"opener": opener}}
                    ],
                }
            ),
            stt=FakeStt(),
            tts=FakeTts(),
            tools=FakeTools(),
            voice_sample_seconds=8,
            silence_threshold=0.01,
            silence_seconds=1.2,
            max_listen_seconds=30,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            started = conversation.consider_emotion_observation(
                emotion="sad",
                average_confidence=0.39,
                max_confidence=0.50,
                non_neutral_frames=8,
                total_frames=12,
                distribution="sad=6 disgust=2",
            )

        self.assertTrue(started)
        self.assertEqual(conversation.mode, "conversation")
        self.assertEqual(conversation.tts.spoken, [opener])

    def test_ai_declines_and_mode_stays_idle(self) -> None:
        conversation = CompanionConversation(
            llm=FakeLlm({"content": "", "tool_calls": []}),
            stt=FakeStt(),
            tts=FakeTts(),
            tools=FakeTools(),
            voice_sample_seconds=8,
            silence_threshold=0.01,
            silence_seconds=1.2,
            max_listen_seconds=30,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            started = conversation.consider_emotion_observation(
                emotion="sad",
                average_confidence=0.39,
                max_confidence=0.50,
                non_neutral_frames=8,
                total_frames=12,
                distribution="sad=6 disgust=2",
            )

        self.assertFalse(started)
        self.assertEqual(conversation.mode, "idle")
        self.assertEqual(conversation.tts.spoken, [])

    def test_active_conversation_does_not_use_tool_calling(self) -> None:
        llm = FakeLlm({"content": "That sounds like a lot. What's the biggest pressure?", "tool_calls": []})
        conversation = CompanionConversation(
            llm=llm,
            stt=FakeStt("Yeah, I'm just really stressed out with work at the moment."),
            tts=FakeTts(),
            tools=FakeTools(),
            voice_sample_seconds=8,
            silence_threshold=0.01,
            silence_seconds=1.2,
            max_listen_seconds=30,
        )
        conversation.mode = "conversation"

        with contextlib.redirect_stdout(io.StringIO()):
            conversation.listen_and_respond(emotion_hint="sad", confidence=0.39)

        self.assertIsNone(llm.tools[0])
        self.assertEqual(conversation.mode, "conversation")

    def test_user_initiated_reply_moves_idle_to_conversation_without_tools(self) -> None:
        llm = FakeLlm({"content": "Tell me what has been weighing on you.", "tool_calls": []})
        conversation = CompanionConversation(
            llm=llm,
            stt=FakeStt("I'm having a rough day."),
            tts=FakeTts(),
            tools=FakeTools(),
            voice_sample_seconds=8,
            silence_threshold=0.01,
            silence_seconds=1.2,
            max_listen_seconds=30,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            conversation.listen_and_respond()

        self.assertIsNone(llm.tools[0])
        self.assertEqual(conversation.mode, "conversation")

    def test_clear_decline_returns_to_idle_after_reply(self) -> None:
        llm = FakeLlm(
            {
                "content": (
                    "That's okay. I'm here if you need anything later. "
                    "Take care of yourself."
                ),
                "tool_calls": [],
            }
        )
        conversation = CompanionConversation(
            llm=llm,
            stt=FakeStt("No, it's okay."),
            tts=FakeTts(),
            tools=FakeTools(),
            voice_sample_seconds=8,
            silence_threshold=0.01,
            silence_seconds=1.2,
            max_listen_seconds=30,
        )
        conversation.mode = "conversation"

        with contextlib.redirect_stdout(io.StringIO()):
            conversation.listen_and_respond()

        self.assertEqual(conversation.mode, "idle")
        self.assertEqual(
            conversation.tts.spoken,
            ["That's okay. I'm here if you need anything later. Take care of yourself."],
        )

    def test_no_im_okay_returns_to_idle_after_reply(self) -> None:
        llm = FakeLlm(
            {
                "content": "Take care of yourself, feel free to reach out if you need anything.",
                "tool_calls": [],
            }
        )
        conversation = CompanionConversation(
            llm=llm,
            stt=FakeStt("No, I'm okay."),
            tts=FakeTts(),
            tools=FakeTools(),
            voice_sample_seconds=8,
            silence_threshold=0.01,
            silence_seconds=1.2,
            max_listen_seconds=30,
        )
        conversation.mode = "conversation"

        with contextlib.redirect_stdout(io.StringIO()):
            conversation.listen_and_respond()

        self.assertEqual(conversation.mode, "idle")
        self.assertEqual(
            conversation.tts.spoken,
            ["Take care of yourself, feel free to reach out if you need anything."],
        )

    def test_plain_return_to_idle_response_is_not_spoken(self) -> None:
        llm = FakeLlm({"content": "return_to_idle", "tool_calls": []})
        conversation = CompanionConversation(
            llm=llm,
            stt=FakeStt("Okay, you too."),
            tts=FakeTts(),
            tools=FakeTools(),
            voice_sample_seconds=8,
            silence_threshold=0.01,
            silence_seconds=1.2,
            max_listen_seconds=30,
        )
        conversation.mode = "conversation"

        with contextlib.redirect_stdout(io.StringIO()):
            conversation.listen_and_respond()

        self.assertEqual(conversation.mode, "idle")
        self.assertEqual(conversation.tts.spoken, [])

    def test_closing_plus_return_to_idle_only_speaks_closing(self) -> None:
        llm = FakeLlm(
            {
                "content": "Take care of yourself, goodbye.\nreturn_to_idle",
                "tool_calls": [],
            }
        )
        conversation = CompanionConversation(
            llm=llm,
            stt=FakeStt("Bye."),
            tts=FakeTts(),
            tools=FakeTools(),
            voice_sample_seconds=8,
            silence_threshold=0.01,
            silence_seconds=1.2,
            max_listen_seconds=30,
        )
        conversation.mode = "conversation"

        with contextlib.redirect_stdout(io.StringIO()):
            conversation.listen_and_respond()

        self.assertEqual(conversation.mode, "idle")
        self.assertEqual(conversation.tts.spoken, ["Take care of yourself, goodbye."])

    def test_goodbye_returns_to_idle_after_reply(self) -> None:
        llm = FakeLlm({"content": "Bye for now.", "tool_calls": []})
        conversation = CompanionConversation(
            llm=llm,
            stt=FakeStt("Okay, see ya"),
            tts=FakeTts(),
            tools=FakeTools(),
            voice_sample_seconds=8,
            silence_threshold=0.01,
            silence_seconds=1.2,
            max_listen_seconds=30,
        )
        conversation.mode = "conversation"

        with contextlib.redirect_stdout(io.StringIO()):
            conversation.listen_and_respond()

        self.assertEqual(conversation.mode, "idle")

    def test_inline_return_to_idle_tag_is_not_spoken(self) -> None:
        llm = FakeLlm(
            {
                "content": (
                    '<function(return_to_idle)>{"closing_message": '
                    '"I am here if you need anything. Take care of yourself."}</function>'
                ),
                "tool_calls": [],
            }
        )
        conversation = CompanionConversation(
            llm=llm,
            stt=FakeStt("No, it's okay. I'll talk to them."),
            tts=FakeTts(),
            tools=FakeTools(),
            voice_sample_seconds=8,
            silence_threshold=0.01,
            silence_seconds=1.2,
            max_listen_seconds=30,
        )
        conversation.mode = "conversation"

        with contextlib.redirect_stdout(io.StringIO()):
            conversation.listen_and_respond()

        self.assertEqual(conversation.mode, "idle")
        self.assertEqual(
            conversation.tts.spoken,
            ["I am here if you need anything. Take care of yourself."],
        )

    def test_empty_inline_return_to_idle_tag_returns_to_idle_silently(self) -> None:
        llm = FakeLlm({"content": "<function(return_to_idle)></function>", "tool_calls": []})
        conversation = CompanionConversation(
            llm=llm,
            stt=FakeStt("Thank you."),
            tts=FakeTts(),
            tools=FakeTools(),
            voice_sample_seconds=8,
            silence_threshold=0.01,
            silence_seconds=1.2,
            max_listen_seconds=30,
        )
        conversation.mode = "conversation"

        with contextlib.redirect_stdout(io.StringIO()):
            conversation.listen_and_respond()

        self.assertEqual(conversation.mode, "idle")
        self.assertEqual(conversation.tts.spoken, [])

    def test_extract_inline_tool_call_removes_markup_from_response(self) -> None:
        content = (
            'Okay. <function(return_to_idle)>{"closing_message": '
            '"Take care."}</function>'
        )

        cleaned, tool_calls = _extract_inline_tool_calls(content)

        self.assertEqual(cleaned, "Okay.")
        self.assertEqual(
            tool_calls,
            [{"name": "return_to_idle", "arguments": {"closing_message": "Take care."}}],
        )

    def test_extract_inline_tool_call_handles_groq_failed_generation_shape(self) -> None:
        content = '<function=return_to_idle{"closing_message": "Take care."}</function>'

        cleaned, tool_calls = _extract_inline_tool_calls(content)

        self.assertEqual(cleaned, "")
        self.assertEqual(
            tool_calls,
            [{"name": "return_to_idle", "arguments": {"closing_message": "Take care."}}],
        )

    def test_extract_plain_return_to_idle_command(self) -> None:
        cleaned, tool_calls = _extract_inline_tool_calls("return_to_idle")

        self.assertEqual(cleaned, "")
        self.assertEqual(tool_calls, [{"name": "return_to_idle", "arguments": {}}])

    def test_extract_closing_plus_return_to_idle_command(self) -> None:
        cleaned, tool_calls = _extract_inline_tool_calls(
            "Take care of yourself, goodbye.\nreturn_to_idle"
        )

        self.assertEqual(cleaned, "Take care of yourself, goodbye.")
        self.assertEqual(tool_calls, [{"name": "return_to_idle", "arguments": {}}])

    def test_extract_return_to_idel_typo_command(self) -> None:
        cleaned, tool_calls = _extract_inline_tool_calls("return to idel")

        self.assertEqual(cleaned, "")
        self.assertEqual(tool_calls, [{"name": "return_to_idle", "arguments": {}}])

    def test_user_wants_to_end_conversation_detects_clear_exits_only(self) -> None:
        self.assertTrue(_user_wants_to_end_conversation("No, it's okay."))
        self.assertTrue(_user_wants_to_end_conversation("No, I'm okay."))
        self.assertTrue(_user_wants_to_end_conversation("Okay, you too."))
        self.assertTrue(_user_wants_to_end_conversation("Okay, see ya"))
        self.assertTrue(_user_wants_to_end_conversation("Bye."))
        self.assertFalse(
            _user_wants_to_end_conversation(
                "I don't know, I'm just stressed out at work."
            )
        )
        self.assertFalse(
            _user_wants_to_end_conversation(
                "It's just the workload, it's a lot and I'm the only developer."
            )
        )


if __name__ == "__main__":
    unittest.main()
